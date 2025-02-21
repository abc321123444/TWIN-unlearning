import copy

from torch.utils.data import ConcatDataset

import config
import torch
from dataloader import get_dataloader, get_dataset, Customer_cifar10
from un import class_un
from util import get_model, eval, Custom_dataset, train, eval_all_classes, eval_adverarial_attack, \
     train_for_loss
from un import eval_forget
import os
import numpy as np
import random

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():



    opt = config.get_arguments().parse_args()
    opt.save_dir = './exp/'
    if opt.dataset in ["mnist", "cifar10", "vgg"]:
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    elif opt.dataset == "imagenet":
        opt.num_classes = 100
    elif opt.dataset == "celeba":
        opt.num_classes = 8
    elif opt.dataset == "vgg":
        opt.num_classes = 10
    else:
        raise Exception("Invalid Dataset")

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    elif opt.dataset == "imagenet":
        opt.input_height = 224
        opt.input_width = 224
        opt.input_channel = 3
    elif opt.dataset == "celeba":
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
    elif opt.dataset == "vgg":
        opt.input_height = 224
        opt.input_width = 224
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    opt.bs = 128

    netC, optimizerC, schedulerC = get_model(opt)

    test_dl = get_dataloader(opt, False)

    dataset = Customer_cifar10(keep=f'./exp/keep.npy')

    dataset_test = get_dataset(opt, train=False)
    dataset_test = Custom_dataset(dataset_test)
    dataset_test.filter([i for i in range(opt.num_classes) if i != opt.target_label])

    dataset_con = ConcatDataset((dataset, dataset_test))

    train_dl = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    eval(netC, test_dl, opt)

    for epoch in range(2):
        print("Epoch {}:".format(epoch + 1))

        train(netC, optimizerC, schedulerC, train_dl)

        if epoch % 1 == 0:
            eval(netC, train_dl, opt)
            eval(netC, test_dl, opt)

    torch.save(netC.state_dict(), os.path.join(opt.save_dir, f'{opt.target_label}/Curriculum_model.pth'))


if __name__ == "__main__":
    main()