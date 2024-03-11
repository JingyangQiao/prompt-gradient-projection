import torch
import torch.nn as nn
import copy
import math

from models.clip.prompt_learner import cfgc, load_clip_to_cpu, TextEncoder, PromptLearner
from datasets.class_names import cifar100_classnames


class Ticlip(nn.Module):
    def __init__(self, args):
        super(Ticlip, self).__init__()
        self.args = args
        self.cfg = cfgc()
        clip_model = load_clip_to_cpu(self.cfg)
        self.clip_moedl = clip_model
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        class_names = self.generate(args)
        self.text_prompt_pool = nn.ModuleList([
            PromptLearner(self.cfg, class_names[i], self.clip_moedl)
            for i in range(args["total_tasks"])
        ])  # Text Prompt Pool
        self.image_prompt_pool = nn.ModuleList([
            nn.Linear(args["embed_dim"], args["prompt_length"], bias=False)
            # for i in range(args["total_tasks"])
        ])  # Image Prompt Pool

        for linear in self.image_prompt_pool:
            nn.init.kaiming_uniform_(linear.weight, a=math.sqrt(5))

        self.class_num = len(class_names[0])
        self.task = 0

    @property
    def feature_dim(self):
        return self.image_encoder.output_dim

    def forward(self, image):
        logits = []
        # image_features = self.image_encoder(image.type(self.dtype), self.image_prompt_pool[self.task - 1].weight)
        image_features = self.image_encoder(image.type(self.dtype), self.image_prompt_pool[0].weight)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_prompts = self.text_prompt_pool[self.task - 1]
        tokenized_prompts = text_prompts.tokenized_prompts
        text_features = self.text_encoder(text_prompts(), tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits.append(logit_scale * image_features @ text_features.t())
        return torch.cat(logits, dim=1)

    def interface(self, images, tasks):
        logits = []
        tasks = tasks.cpu().tolist()
        selects = [0 for i in range(len(tasks))]
        image_prompts = torch.stack([prompt.weight for prompt in self.image_prompt_pool], 0)[selects, :, :]
        image_features = self.image_encoder(images.type(self.dtype), image_prompts)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        for text_prompt in self.text_prompt_pool:
            tokenized_prompts = text_prompt.tokenized_prompts
            text_features = self.text_encoder(text_prompt(), tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            logits.append(logit_scale * image_features @ text_features.t())
        logits = torch.cat(logits,1)
        selectedlogit = []
        if self.args["mode"] == "CIL":
            cur_id = max(tasks)
        for idx, ii in enumerate(tasks):
            if self.args["mode"] == "TIL":
                selectedlogit.append(logits[idx][self.class_num*ii:self.class_num*ii+self.class_num])
            if self.args["mode"] == "CIL":
                selectedlogit.append(logits[idx][:self.class_num * cur_id + self.class_num])
        selectedlogit = torch.stack(selectedlogit)

        return selectedlogit

    def query(self, images, tasks):
        tasks = tasks.cpu().tolist()
        selects = [0 for i in range(len(tasks))]
        image_prompts = torch.stack([prompt.weight for prompt in self.image_prompt_pool], 0)[selects, :, :]
        _ = self.image_encoder(images.type(self.dtype), image_prompts)
        representations = self.image_encoder.act['rep']
        return representations

    def update_fc(self):
        self.task += 1

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def generate(self, args):
        temp_names = list(cifar100_classnames.values())
        class_names = []
        for i in range(args["total_tasks"]):
            class_names.append(temp_names[10 * i:10 * i + 10])
        return class_names
