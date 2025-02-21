# TWIN-unlearning
This is the offical implementation of paper **Towards Aligned Data Forgetting via Twin Machine Unlearning**.


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

You can dowanload dataset cifar10 on http://www.cs.toronto.edu/~kriz/cifar.html

### Run the code

**Train an original model**
```
python train.py --dataset cifar10
```
**Train a gold model**
```
python train_gm.py --dataset cifar10 --target_label 0
```
**Prepare for unlearning**

***Train a model from scratch for 2 epochs for loss feature***
```
python train_loss.py --dataset cifar10 --target_label 0
```
***Finetune original model on val dataset for the other two features***
```
python train_test_data.py --dataset cifar10 --target_label 0
```
**Train binary classifier and unlearning**
```
python train_generalization_binary_classifier.py --dataset cifar10 --target_label 0
```
