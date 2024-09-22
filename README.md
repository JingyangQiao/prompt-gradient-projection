# prompt-gradient-projection

Official Pytorch implementation for "**Prompt Gradient Projection for Continual Learning**", **ICLR 2024 (Spotlight)**.

#######################################################################################
## Our novel project page about continual instruction tuning now is available at (https://jingyangqiao.github.io/)

Based on the Exponential Moving Average (EMA) method, we have proposed work about how to resist forgetting in various efficient-parameter include **prompt, prefix, LoRA** with **LLaVA-1.5** backbone.

## Our project page about unified continual image classification framework now is available at (https://dmcv-ecnu-pegp.github.io/)

Recently, based on prompt gradient projection method, we have proposed an unified work about how to resist forgetting by gradient projection in various efficient-parameter include **prompt, prefix, adapter, LoRA** with **ViT, CLIP** backbones.

We have validated our work on more incremental settings, e.g. CIL, TIL, DIL, OIL.

Detailed information please kindly refer to the above website.

#######################################################################################

~~**Code will come soon!**~~

L2P-PGP is published in 2/27/2024

~~DualPrompt-PGP and CLIP-PGP will be published in 3/2024~~

DualPrompt-PGP is published in 3/10/2024

~~CLIP-PGP will be published in 3/2024~~

CLIP-PGP is published in 3/11/2024

## Abstract 

Prompt-tuning has demonstrated impressive performance in continual learning by querying relevant prompts for each input instance, which can avoid the introduction of task identifier. Its forgetting is therefore reduced as this instance-wise query mechanism enables us to select and update only relevant prompts. In this paper, we further integrate prompt-tuning with gradient projection approach. Our observation is: prompt-tuning releases the necessity of task identifier for gradient projection method; and gradient projection provides theoretical guarantees against forgetting for prompt-tuning. This inspires a new **p**rompt **g**radient **p**rojection approach (PGP) for continual learning. In PGP, we deduce that reaching the orthogonal condition for prompt gradient can effectively prevent forgetting via the self attention mechanism in vision-transformer. The condition equations are then realized by conducting Singular Value Decomposition (SVD) on an element-wise sum space between input space and prompt space. We validate our method on diverse datasets and experiments demonstrate the efficiency of reducing forgetting both in class incremental, online class incremental, and task incremental settings.

## Thanks

The baseline code of L2P and DualPrompt are from (https://github.com/JH-LEE-KR/l2p-pytorch) and (https://github.com/JH-LEE-KR/dualprompt-pytorch) and the baseline code of CLIP is from (https://github.com/iamwangyabin/s-prompts)


## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Citation

```
@inproceedings{qiao2024PGP,
  title={Prompt Gradient Projection for Continual Learning},
  author={Jingyang Qiao, Zhizhong Zhang, Xin Tan, Chengwei Chen, Yanyun Qu, Yong Peng, Yuan Xie},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```
