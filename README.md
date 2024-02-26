# prompt-gradient-projection

Official Pytorch implementation for "**Prompt Gradient Projection for Continual Learning**", **ICLR 2024 (Spotlight)**. 

~~**Code will come soon!**~~

L2P-PGP is published in 2/27/2024

DualPrompt-PGP and CLIP-PGP will be published in 3/2024

## Abstract 

Prompt-tuning has demonstrated impressive performance in continual learning by querying relevant prompts for each input instance, which can avoid the introduction of task identifier. Its forgetting is therefore reduced as this instance-wise query mechanism enables us to select and update only relevant prompts. In this paper, we further integrate prompt-tuning with gradient projection approach. Our observation is: prompt-tuning releases the necessity of task identifier for gradient projection method; and gradient projection provides theoretical guarantees against forgetting for prompt-tuning. This inspires a new **p**rompt **g**radient **p**rojection approach (PGP) for continual learning. In PGP, we deduce that reaching the orthogonal condition for prompt gradient can effectively prevent forgetting via the self attention mechanism in vision-transformer. The condition equations are then realized by conducting Singular Value Decomposition (SVD) on an element-wise sum space between input space and prompt space. We validate our method on diverse datasets and experiments demonstrate the efficiency of reducing forgetting both in class incremental, online class incremental, and task incremental settings.

## Usage

First, clone our repository:

```
git clone https://github.com/JingyangQiao/prompt-gradient-projection/l2p-pgp
cd l2p-pgp
```

Then, install the packages below:

```
pytorch==1.13.1
torchvision==0.14.1
numpy==1.25.0
timm==0.6.7
sklearn==1.3.0
matplotlib
```

or you can install these packages with ```requirements.txt``` by: 

```
pip install -r requirements.txt
```

## Data preparation

If you already have CIFAR-100 or ImageNet-R or CUB200 or TinyImageNet, pass your dataset path to  `--data-path` in each 
```configs/*.py``` file.

If the datasets aren't ready, just run the training command and the datasets will be downloaded automatically in the `--data-path`.

## Training

To train a model via command line:

**For CIL (Class Incremental Learning) Settings:**

10-Split-CIFAR100

```
python main.py 10cifar100_l2p_pgp --model vit_base_patch16_224 --output_dir ./output --epochs 5
```

20-Split-CIFAR100

```
python main.py 20cifar100_l2p_pgp --model vit_base_patch16_224 --output_dir ./output --epochs 5
```

10-Split-ImageNet-R

```
python main.py imr_l2p_pgp --model vit_base_patch16_224 --output_dir ./output --epochs 5
```

10-Split-TinyImageNet

```
python main.py tinyimagenet_l2p_pgp --model vit_base_patch16_224 --output_dir ./output --epochs 5
```

5-Split-CUB200

```
python main.py cub200_l2p_pgp --model vit_base_patch16_224 --output_dir ./output --epochs 5
```

**For OIL (Online Incremental Learning) settings:**

For 10-Split-CIFAR100

```
python main.py 10cifar100_l2p_pgp --model vit_base_patch16_224 --output_dir ./output --epochs 1
```

For 20-Split-CIFAR100

```
python main.py 20cifar100_l2p_pgp --model vit_base_patch16_224 --output_dir ./output --epochs 1 --seed 42
```

For 10-Split-TinyImageNet

```
python main.py tinyimagenet_l2p_pgp --model vit_base_patch16_224 --output_dir ./output --epochs 1 --seed 50
```

**We also keep the original L2P method in our codebase and command line is like:**

10-Split-CIFAR100

```
python main.py 10cifar100_l2p_pgp --model vit_base_patch16_224 --output_dir ./output --epochs 5 --no_pgp
```

20-Split-CIFAR100

```
python main.py 20cifar100_l2p_pgp --model vit_base_patch16_224 --output_dir ./output --epochs 5 --no_pgp
```

10-Split-ImageNet-R

```
python main.py imr_l2p_pgp --model vit_base_patch16_224 --output_dir ./output --epochs 5 --pgp False --no_pgp
```

10-Split-TinyImageNet

```
python main.py tinyimagenet_l2p_pgp --model vit_base_patch16_224 --output_dir ./output --epochs 5 --pgp False --no_pgp
```

5-Split-CUB200

```
python main.py cub200_l2p_pgp --model vit_base_patch16_224 --output_dir ./output --epochs 5 --pgp False --no_pgp
```

## Evaluation

To evaluate a trained model:

```
python main.py <10cifar100_l2p_pgp or 20cifar100_l2p_pgp or imr_l2p_pgp or tinyimagenet_l2p_pgp or cub200_l2p_pgp> --eval
```

## Thanks

The baseline code of L2P and DualPrompt are from (https://github.com/JH-LEE-KR/l2p-pytorch) and (https://github.com/JH-LEE-KR/dualprompt-pytorch)


## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Citation

```
@inproceedings{qiao2024PGP,
  title={Prompt Gradient Projection for Continual Learning},
  author={Jingyang Qiao & Zhizhong Zhang, Xin Tan, Chengwei Chen, Yanyun Qu, Yong Peng, Yuan Xie},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```
