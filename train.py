import copy
import os

from torch.utils.data import ConcatDataset, Subset

import config
import torch
from dataloader import get_dataloader, get_dataset, Customer_cifar10
from util import get_model, eval, train, Custom_dataset, eval_all_classes
import random
import numpy as np

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

    dataset = Customer_cifar10()


    train_dl = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    eval(netC, test_dl, opt)

    for epoch in range(10):
        print("Epoch {}:".format(epoch + 1))

        train(netC, optimizerC, schedulerC, train_dl)

        if epoch % 10 == 0:
            eval(netC, train_dl, opt)
            eval(netC, test_dl, opt)

    #np.save(os.path.join(opt.save_dir, 'keep.npy'), keep)
    torch.save(netC.state_dict(), os.path.join(opt.save_dir, 'model.pth'))



if __name__ == "__main__":
    main()