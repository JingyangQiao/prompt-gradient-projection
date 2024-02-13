# prompt-gradient-projection

Official Pytorch implementation for "**Prompt Gradient Projection for Continual Learning**", **ICLR 2024 (Spotlight)**. 

**Code will come soon!**

## Abstract 

Prompt-tuning has demonstrated impressive performance in continual learning by querying relevant prompts for each input instance, which can avoid the introduction of task identifier. Its forgetting is therefore reduced as this instance-wise query mechanism enables us to select and update only relevant prompts. In this paper, we further integrate prompt-tuning with gradient projection approach. Our observation is: prompt-tuning releases the necessity of task identifier for gradient projection method; and gradient projection provides theoretical guarantees against forgetting for prompt-tuning. This inspires a new **p**rompt **g**radient **p**rojection approach (PGP) for continual learning. In PGP, we deduce that reaching the orthogonal condition for prompt gradient can effectively prevent forgetting via the self attention mechanism in vision-transformer. The condition equations are then realized by conducting Singular Value Decomposition (SVD) on an element-wise sum space between input space and prompt space. We validate our method on diverse datasets and experiments demonstrate the efficiency of reducing forgetting both in class incremental, online class incremental, and task incremental settings.

## Usage

## Experiments

This repository currently contains experiments reported in the paper for 10-split-CIFAR100, 20-split-CIFAR100, 10-Split-TinyImageNet, 10-Split-ImageNet-R and 5-Split-CUB200 datasets under the settings of class incremental learning, online class incremental learning and task incremental learning with the baselines of [L2P](https://github.com/JH-LEE-KR/l2p-pytorch), [DualPrompt](https://github.com/JH-LEE-KR/dualprompt-pytorch) and [CLIP](https://github.com/iamwangyabin/s-prompts).

## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Citation

```

```
