import copy

from torch.utils.data import ConcatDataset, Subset

import config
import torch
from dataloader import get_dataloader, get_dataset, Customer_cifar10
from un import class_un
from util import get_model, eval, Custom_dataset, train, eval_all_classes, eval_adverarial_attack
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

    opt.num_classes = 10

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "cifar100":
        opt.input_height = 224
        opt.input_width = 224
        opt.input_channel = 3
    elif opt.dataset == "vgg":
        opt.input_height = 224
        opt.input_width = 224
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    dataset_train = Customer_cifar10(keep='./exp/keep.npy')
    forget_indices = dataset_train.forget_indices(opt.target_label)
    ratio = 0.9
    set_seed(0)
    shuffled_indices = np.random.permutation(forget_indices[0])
    split_point = int(len(shuffled_indices) * ratio)

    # 分割索引数组
    part1_indices = shuffled_indices[:split_point]
    part2_indices = shuffled_indices[split_point:]

    np.save(f'./exp/{opt.target_label}/forget_indices.npy', np.where(dataset_train.bool_array == True)[0][part2_indices])
    #exit()
    dataset_t_nf = Subset(dataset_train, part1_indices)
    dataset_t_f = Subset(dataset_train, part2_indices)

    # 创建剩余数据的子集
    all_indices = set(range(len(dataset_train)))
    remaining_indices = list(all_indices - set(forget_indices[0]))
    dataset_t_r = Subset(dataset_train, remaining_indices)

    dataset_con = ConcatDataset((dataset_t_nf, dataset_t_r))

    # prepare model
    netC, optimizerC, schedulerC = get_model(opt)
    dl_train_con = torch.utils.data.DataLoader(dataset_con, batch_size=64, num_workers=opt.num_workers, shuffle=True)

    test_dataloader = get_dataloader(opt, False)

    for epoch in range(10):
        print("Epoch {}:".format(epoch + 1))
        train(netC , optimizerC, schedulerC, dl_train_con)

        if (epoch+1) % 10 == 0:
            eval(netC, test_dataloader, opt)
            eval(netC, dataset_t_f, opt)
            eval(netC, dl_train_con , opt)

    opt.save_dir = './exp/'
    torch.save(netC.state_dict(), os.path.join(opt.save_dir, f'{opt.target_label}/Gmodel.pth'))


if __name__ == "__main__":
    main()