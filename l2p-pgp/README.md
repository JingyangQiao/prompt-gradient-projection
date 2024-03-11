## Usage

First, clone our repository:

```
git clone https://github.com/JingyangQiao/prompt-gradient-projection
cd prompt-gradient-projection/l2p-pgp
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
python main.py imr_l2p_pgp --model vit_base_patch16_224 --output_dir ./output --epochs 50
```

10-Split-TinyImageNet

```
python main.py tinyimagenet_l2p_pgp --model vit_base_patch16_224 --output_dir ./output --epochs 5
```

5-Split-CUB200

```
python main.py cub200_l2p_pgp --model vit_base_patch16_224 --output_dir ./output --epochs 5
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
python main.py imr_l2p_pgp --model vit_base_patch16_224 --output_dir ./output --epochs 50 --no_pgp
```

10-Split-TinyImageNet

```
python main.py tinyimagenet_l2p_pgp --model vit_base_patch16_224 --output_dir ./output --epochs 5 --no_pgp
```

5-Split-CUB200

```
python main.py cub200_l2p_pgp --model vit_base_patch16_224 --output_dir ./output --epochs 5 --no_pgp
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
  author={Jingyang Qiao, Zhizhong Zhang, Xin Tan, Chengwei Chen, Yanyun Qu, Yong Peng, Yuan Xie},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```
