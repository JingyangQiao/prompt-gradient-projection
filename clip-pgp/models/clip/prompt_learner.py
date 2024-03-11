import torch
import torch.nn as nn

from models.clip import clip
from models.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer


_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.backbonename
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)  # Embedding Prompts
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module): # 该类根据给定的类别名称和配置参数，生成一组文本提示，这些提示将用于训练分类模型
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames) # 类别数量
        n_ctx = cfg.NCTX  # 上下文向量的数量，用于生成文本提示。这是一个超参数，用于控制提示的复杂性。
        ctx_init = cfg.CTXINIT  # 一个字符串，表示初始上下文（context）。如果提供了初始上下文，则将使用它作为提示的一部分。
        dtype = clip_model.dtype     
        ctx_dim = clip_model.ln_final.weight.shape[0]   # 上下文向量的维度

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1:1+n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if cfg.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)   # 上下文向量，可以是类别特定的或通用的，取决于配置参数。
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)  # 上下文向量，可以是类别特定的或通用的，取决于配置参数。
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx) # 引号中是空格

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        device = clip_model.token_embedding.weight.device
        self.ctx = nn.Parameter(ctx_vectors).to(device)  # 一个可学习的参数，代表上下文向量，它将在训练中进行优化
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]   # 类别名称的长度，用于确定上下文的数量。 
        # _tokenizer.encode 函数通常用于将文本标记化（tokenization）后的文本转换为模型可以处理的数值形式
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)   # 标记化的文本提示，将用于训练分类模型
        # clip.tokenize 是 OpenAI 的 CLIP 模型中用于将文本标记化的函数。
        # 它的作用是将输入的文本字符串转换为模型可以理解的标记序列，这些标记通常是整数或特殊的编码
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            # clip_model.token_embedding 是 CLIP 模型中的一个成员
            # 它表示 CLIP 模型的文本嵌入层。这一层负责将文本标记（token）映射为高维的嵌入向量

        self.register_buffer("token_prefix", embedding[:, :1, :])   # 前缀，用于构建完整的文本提示。对应这里是前面prefix中的一个空格
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])   # 后缀，用于构建完整的文本提示
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.CLASS_TOKEN_POSITION    # 配置参数，指示类别标记（class token）的位置

    def forward(self):  #  forward 方法是 PromptLearner 类的前向传播方法，用于生成分类任务的文本提示（prompts）
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix  # 前缀
        suffix = self.token_suffix  # 后缀
        if self.class_token_position == "end":  # 根据类别标记（class token）的位置，构建不同类型的文本提示。
                                                # CLASS_TOKEN_POSITION 参数指的是在 suffix（后缀标记）中类别标记（class token）的位置
            prompts = torch.cat([prefix, ctx, suffix], dim=1)   # pre和su分别是之前XXX CLS.中的第一个空格和除了16个X外的所有的embedding信息，然后把可学习的16个ctx放进去作为learnable的prompt
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i+1, :, :]
                class_i = suffix[i:i+1, :name_len, :]
                suffix_i = suffix[i:i+1, name_len:, :]
                ctx_i_half1 = ctx[i:i+1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i:i+1, half_n_ctx:, :]
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i+1, :, :]
                class_i= suffix[i:i+1, :name_len, :]
                suffix_i = suffix[i:i+1, name_len:, :]
                ctx_i = ctx[i:i+1, :, :]
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError
        return prompts

class cfgc(object):
    backbonename = 'ViT-B/16'
    NCTX = 16
    CTXINIT = ''
    CSC = False
    CLASS_TOKEN_POSITION = 'end'