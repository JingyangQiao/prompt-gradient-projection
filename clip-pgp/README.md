## Usage

First, clone our repository:

```
git clone https://github.com/JingyangQiao/prompt-gradient-projection
cd prompt-gradient-projection/clip-pgp
```

Then, install the packages with ```requirements.txt``` by: 

```
pip install -r requirements.txt
```

## Data preparation

If you already have CIFAR-100, pass your dataset path to  `data_path` in ```configs/cifar100_split.json``` file.

If the datasets aren't ready, just run the training command and the datasets will be downloaded automatically in the `data_path`.

## Training

To train a model via command line:

**For CIL (Class Incremental Learning) Settings:**

10-Split-CIFAR100

Change `"mode": "CIL"` in ```configs/cifar100_split.json``` file.
```
python main.py --config configs/cifar100_split.json
```

**For TIL (Task Incremental Learning) settings:**

10-Split-CIFAR100

Change `"mode": "TIL"` in ```configs/cifar100_split.json``` file.
```
python main.py --config configs/cifar100_split.json
```

## Thanks

The baseline code is from (https://github.com/iamwangyabin/s-prompts)


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
