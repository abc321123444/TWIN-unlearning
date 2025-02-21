# TWIN-unlearning

## Getting Started

### Installation

**1. Prepare the code and the environment**

Git clone our repository, creating a python environment and activate it via the following command

```bash
git clone https://github.com/abc321123444/TWIN-unlearning
cd TWIN-unlearning
conda env create -f env_yaml
conda activate unlearning
```

**2. Download datasets**
You can dowanload dataset cifar10 on https://blog.csdn.net/qq_41185868/article/details/82793025
**Train an original model**
```
python train.py --dataset cifar10
```
**Train a gold model**
python train_gm.py --dataset cifar10 --target_label 0
**Prepare for unlearning**
***Train a model from scatch for 2 epochs for c loss***
python train_loss.py --dataset cifar10 --target_label 0
***Finetune original model on val dataset for the other two losses***
python train_test_data.py --dataset cifar10 --target_label 0
**Train binary classifier and unlearning**
python train_generalization_binary_classifier.py --dataset cifar10 --target_label 0
