from torch.utils.data import ConcatDataset
import config
import torch
from dataloader import get_dataset, get_dataloader, Customer_cifar10
from util import get_model, eval, Custom_dataset, train, eval_adverarial_attack
import os
import copy
import randoml
import numpy as np
import random

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main():
    opt = config.get_arguments().parse_args()

    if opt.dataset in ["mnist", "cifar10"]:
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    elif opt.dataset == "imagenet":
        opt.num_classes = 100
    elif opt.dataset == "celeba":
        opt.num_classes = 8
    elif opt.dataset == "vgg":
        opt.num_classes = 10
    elif opt.dataset == "cifar100":
        opt.num_classes = 100
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
        opt.input_height = 112
        opt.input_width = 112
        opt.input_channel = 3
    elif opt.dataset == "cifar100":
        opt.input_height = 224
        opt.input_width = 224
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    netC, optimizerC, schedulerC = get_model(opt)
    netC.load_state_dict(torch.load('./exp/model.pth'))

    #optimizerC = torch.optim.SGD(netC.parameters(), 0.01, momentum=0.9, weight_decay=5e-4)
    optimizerC = torch.optim.Adam(netC.parameters(), lr=1e-4)
    #schedulerC = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerC, 120)



    dataset = Customer_cifar10(keep='./exp/keep.npy')

    data_list, label_list = dataset.get_left()

    data_tensor = torch.tensor(data_list, dtype=torch.float)
    label_tensor = torch.tensor(label_list)
    from torch.utils.data import TensorDataset
    # 使用TensorDataset将数据和标签组合成一个Dataset对象
    dataset_left = TensorDataset(data_tensor, label_tensor)
    dataset_test = Custom_dataset(dataset_left)
    dataset_test.filter([i for i in range(opt.num_classes) if i != opt.target_label])

    dataset_con = ConcatDataset((dataset, dataset_test))

    train_dl = torch.utils.data.DataLoader(dataset_con, batch_size=64, shuffle=True)

    for epoch in range(10):
        print("Epoch {}:".format(epoch + 1))
        from util import train
        train(netC, optimizerC, schedulerC, train_dl)

        if True:
            eval(netC, dataset_test)
            eval(netC, dataset)

    opt.save_dir = './exp/'
    torch.save(netC.state_dict(), os.path.join(opt.save_dir, f'{opt.target_label}/test_finetune.pth'))


if __name__ == "__main__":
    main()