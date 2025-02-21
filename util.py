import sys
import time
import os

import torch
from torch.utils.data import Dataset, ConcatDataset, Subset, DataLoader
from classifier_models import PreActResNet18
from networks.models import NetC_MNIST
from typing import Callable, Iterable, Tuple
import torch.nn as nn
from torchvision import models
import kornia.augmentation as A
import random
import kornia
import random
from classifier_models.resnet import ResNet18

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0,
                 activation_fn=nn.ReLU, batch_norm=True, transpose=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        model = []
        if not transpose:
#             model += [ConvStandard(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
#                                 )]
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=not batch_norm)]
        else:
            model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         output_padding=output_padding, bias=not batch_norm)]
        if batch_norm:
            model += [nn.BatchNorm2d(out_channels, affine=True)]
        model += [activation_fn()]
        super(Conv, self).__init__(*model)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self,x):
        return x.view(x.size(0), -1)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
class AllCNN(nn.Module):
    def __init__(self, filters_percentage=1., n_channels=3, num_classes=10, dropout=False, batch_norm=True):
        super(AllCNN, self).__init__()
        n_filter1 = int(96 * filters_percentage)
        n_filter2 = int(192 * filters_percentage)
        self.features = nn.Sequential(
            Conv(n_channels, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),  # 14
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=1, stride=1, batch_norm=batch_norm),
            nn.AvgPool2d(8),
            Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_filter2, num_classes),
        )

    def forward(self, x):
        features = self.features(x)
        #print(features.shape)
        output = self.classifier(features)
        return output

    def get_feature(self, x):
        features = self.features(x)
        return features



def get_model(opt):
    netC = None

    if opt.dataset == "cifar10" or opt.dataset == "gtsrb" or opt.dataset =="vgg":
        #from torchvision.models import resnet18
        #print(opt.a)
        #netC = ViT().cuda()
        #if opt.a:

        from classifier_models.wide_resnet import WideResNet
        from classifier_models.resnet import ResNet18
        #else:
        from un import ViT
        #netC = ViT(num_classes=10).cuda()
        #netC = resnet18().cuda()
        netC = ResNet18().cuda()
        #netC = WideResNet(depth=28, num_classes=10).cuda()
        #from models.vit_small import ViT
        from models.simple_vit import SimpleViT
        from models.vit_small import ViT

        #netC = AllCNN().cuda()
        #netC =ResNet(BasicBlock, [2,2,2,2]).to(opt.device)
        '''
        netC = ViT(
            image_size=32,
            patch_size=4,
            num_classes=10,
            dim=int(512),
            depth=6,
            heads=8,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1
        ).cuda()
        
        
        netC =  ViT(
            image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = 512,
            depth = 4,
            heads = 6,
            mlp_dim = 256,
            dropout = 0.1,
            emb_dropout = 0.1
        ).cuda()
        '''
        optimizerC = torch.optim.Adam(netC.parameters(), lr=1e-4)
        schedulerC = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerC, 120)
        #vit
        #optimizerC = torch.optim.SGD(netC.parameters(), 0.0001, momentum=0.9, weight_decay=5e-4)
        #schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)
        #from vit_pytorch import ViT, SimpleViT

        #optimizerC = torch.optim.SGD(netC.parameters(), 0.01, momentum=0.9, weight_decay=5e-4)
        #schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)


    if opt.dataset == "cifar100" :
        from sp20_model import ResNet18
        netC = ResNet18(num_classes = 20, pretrained = True).to(opt.device)
        optimizerC = torch.optim.Adam(netC.parameters(), lr=1e-3)
        schedulerC = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerC, 120)
        #from vit_pytorch import ViT
        #netC = ViT()
        #netC = ResNet18(num_classes=opt.num_classes).to(opt.device)
    if opt.dataset == "celeba"  :
        netC = ResNet18(opt.num_classes).to(opt.device)
    if opt.dataset == "mnist":
        netC = NetC_MNIST().to(opt.device)
        num_ftrs = netC.linear9.in_features
        netC.linear9 = nn.Linear(num_ftrs, opt.num_classes)
        netC.to("cuda")

    if opt.dataset == 'imagenet':
        netC = models.resnet18(num_classes=1000, pretrained="imagenet")
        # netC = models.resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = netC.fc.in_features
        netC.fc = nn.Linear(num_ftrs, 100)
        netC.cuda()

    if opt.dataset == 'tini_imagenet':
        netC = models.resnet18(num_classes=1000, pretrained="imagenet")
        # netC = models.resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = netC.fc.in_features
        netC.fc = nn.Linear(num_ftrs, 197)
        netC.cuda()


    # Optimizer


    return netC, optimizerC, schedulerC

def evalb(netC, test_dl, return_flag=True):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    total_t = 0
    t= 0
    total_f = 0
    f = 0
    flag = []

    for batch_idx, (inputs,  targets) in enumerate(test_dl):
        #print(targets)
        with torch.no_grad():
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            total_targets = targets
            #print(targets)
            bs = inputs.shape[0]
            total_sample += bs

            t += torch.sum(targets)
            f += len(targets) - torch.sum(targets)
            # Evaluate Clean
            total_preds = netC(inputs)
            #total_preds = torch.sigmoid_(total_preds)
            #print(total_preds)

            for i in range(len(inputs)):
                if total_preds[i] >= 0.5 and targets[i] == 1 :
                    total_t += 1
                    total_clean_correct += 1
                elif total_preds[i] <= 0.5 and targets[i] == 0:
                    total_f += 1
                    total_clean_correct += 1
                else:
                    pass

            for i in range(len(inputs)):
                if total_preds[i] > 0.5 :
                    flag.append(torch.tensor(1))
                elif total_preds[i] <= 0.5 :
                    flag.append(torch.tensor(0))
                else:
                    pass


    acc_clean = total_clean_correct * 100.0 / total_sample
   # print(total_t)
    #print(t)
    acc_t = total_t * 100.0 /t
    #print(total_f)
    #print(f)
    acc_f = total_f * 100.0 /f

    info_string = "Clean Acc: {:.4f} total sample : {} T_ACC: {} F_ACC :{}".format(
        acc_clean, total_sample, acc_t, acc_f
    )
    print(info_string)
    if return_flag:
        return torch.stack(flag, dim=-1)
    else:
        return acc_clean, acc_t, acc_f


def getF(netC, test_dl, return_flag=True):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    flag = []
    if isinstance(test_dl, Dataset):
        test_dl = DataLoader(test_dl, batch_size=64, shuffle=False)
    else:
        pass

    for batch_idx, (inputs,  targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            bs = inputs.shape[0]
            total_sample += bs
            total_preds = netC(inputs)

            for i in range(len(inputs)):
                if torch.argmax(total_preds[i]) == targets[i] :
                    flag.append(torch.tensor(1))
                else:
                    flag.append(torch.tensor(0))

    if return_flag:
        return torch.stack(flag, dim=-1)


def eval(netC, test_dl, setreturn = True, reloss = True):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    dataset_T = []
    dataset_F = []
    output = []
    criterion = torch.nn.CrossEntropyLoss()
    loss = []

    if isinstance(test_dl, Dataset):
        test_dl = DataLoader(test_dl, batch_size=64, shuffle=False)
    else:
        pass

    for batch_idx, (inputs,  targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            total_preds = netC(inputs)
            if reloss:
                for i in range(len(inputs)):
                    loss.append(criterion(total_preds[i], targets[i]))

            for i in range(len(inputs)):
                if torch.argmax(total_preds[i]) == targets[i]:

                    dataset_T.append((inputs[i].cpu(),  int(targets[i]))) #,int(old_label[i]),
                    total_clean_correct += 1
                    output.append(1)
                else:
                    output.append(0)
                    dataset_F.append((inputs[i].cpu(),  int(targets[i])))


    acc_clean = total_clean_correct * 100.0 / total_sample

    info_string = "Clean Acc: {:.4f} total sample : {}".format(
        acc_clean, total_sample
    )
    print(info_string)
    if reloss:
        loss = torch.stack(loss)
        print("loss is :", torch.mean(loss))
    #print(output)
    if setreturn:
        return dataset_T, dataset_F
    else:
        return acc_clean


def eval2(netC, test_dl, setreturn = True, reloss = True):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    dataset_T = []
    dataset_F = []
    output = []
    criterion = torch.nn.CrossEntropyLoss()
    loss = []

    if isinstance(test_dl, Dataset):
        test_dl = DataLoader(test_dl, batch_size=64, shuffle=False)
    else:
        pass

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            total_preds = netC(inputs)
            if reloss:
                for i in range(len(inputs)):
                    loss.append(criterion(total_preds[i], targets[i]))

            for i in range(len(inputs)):
                if torch.argmax(total_preds[i]) == targets[i]:

                    dataset_T.append((inputs[i].cpu(),  int(targets[i]))) #,int(old_label[i]),
                    total_clean_correct += 1
                    output.append(1)
                else:
                    output.append(0)
                    dataset_F.append((inputs[i].cpu(),  int(targets[i])))


    acc_clean = total_clean_correct * 100.0 / total_sample

    info_string = "Clean Acc: {:.4f} total sample : {}".format(
        acc_clean, total_sample
    )
    print(info_string)
    if reloss:
        loss = torch.stack(loss)
        print("loss is :", torch.mean(loss))
    #print(output)
    if setreturn:
        return dataset_T, dataset_F
    else:
        return acc_clean

def eval_bbbb(netC, test_dl, setreturn = True):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    dataset_T = []
    dataset_F = []
    output = []
    j = 0

    for batch_idx, (inputs,  targets) in enumerate(test_dl):
        #print(targets)
        with torch.no_grad():
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            total_targets = targets
            #print(targets)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            total_preds = netC(inputs)
            #total_preds = torch.sigmoid_(total_preds)

            for i in range(len(inputs)):
                if j>=50:

                    j += 1
                    dataset_T.append((inputs[i].cpu(),  int(targets[i].cpu()))) #,int(old_label[i]),
                    total_clean_correct += 1
                    output.append(1)
                else:
                    j += 1
                    output.append(0)
                    dataset_F.append((inputs[i].cpu(),  int(targets[i].cpu())))


    acc_clean = total_clean_correct * 100.0 / total_sample

    info_string = "Clean Acc: {:.4f} total sample : {}".format(
        acc_clean, total_sample
    )
    if setreturn:
        return dataset_T, dataset_F
    else:
        return acc_clean


def eval_rl(netC, test_dl):
    print(" Eval:")
    netC.eval()
    n = 0
    total_sample = 0
    total_clean_correct = 0
    dataset_T = []
    dataset_F = []
    criterion = torch.nn.CrossEntropyLoss()
    loss = []
    if isinstance(test_dl, Dataset):
        test_dl = DataLoader(test_dl, batch_size=64, shuffle=False)
    else:
        pass

    for batch_idx, (inputs,  targets) in enumerate(test_dl):
        #print(targets)
        with torch.no_grad():
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            total_targets = targets
            #print(targets)
            bs = inputs.shape[0]
            total_sample += bs


            total_preds = netC(inputs)

            for i in range(len(inputs)):
                loss.append(criterion(total_preds[i], targets[i]))
                if torch.argmax(total_preds[i]) == targets[i]:
                    total_clean_correct += 1
            n += len(targets)

    acc_clean = total_clean_correct * 100.0 / total_sample

    info_string = "Clean Acc: {:.4f} total sample : {}".format(
        acc_clean, total_sample
    )
    print(info_string)
    loss = torch.stack(loss)
    return loss

def get_TF(netC, test_dl,):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    Flag = []

    for batch_idx, (inputs,  targets) in enumerate(test_dl):
        #print(targets)
        with torch.no_grad():
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            total_targets = targets
            #print(targets)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            total_preds = netC(inputs)
            #total_preds = torch.sigmoid_(total_preds)

            for i in range(len(inputs)):
                if torch.argmax(total_preds[i]) == targets[i]:
                    print(torch.argmax(total_preds[i]))
                    print(targets[i])
                    Flag.append(torch.tensor(1)) #,int(old_label[i]),
                    total_clean_correct += 1
                else:
                    Flag.append(torch.tensor(0))
                # print(total_preds[i])
                #if total_preds[i] > int(total_targets[i]) * 0.5 and total_preds[i] <= int(total_targets[i]) * 0.5 + 0.5:
                    #total_clean_correct += 1


    acc_clean = total_clean_correct * 100.0 / total_sample

    info_string = "Clean Acc: {:.4f} total sample : {}".format(
        acc_clean, total_sample
    )
    print(info_string)
    Flag = torch.stack(Flag, dim=-1)
    return Flag

def eval_label(netC, test_dl, opt=None ):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    dataset_T = []
    dataset_F = []
    label = []

    for batch_idx, (inputs,  targets) in enumerate(test_dl):
        #print(targets)
        with torch.no_grad():
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            total_targets = targets
            #print(targets)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            total_preds = netC(inputs)
            #total_preds = torch.sigmoid_(total_preds)

            for i in range(len(inputs)):
                if torch.argmax(total_preds[i]) == targets[i]:
                    label.append(1)
                else:
                    #print(torch.argmax(total_preds[i]))
                    label.append(0)
                # print(total_preds[i])
                #if total_preds[i] > int(total_targets[i]) * 0.5 and total_preds[i] <= int(total_targets[i]) * 0.5 + 0.5:
                    #total_clean_correct += 1


    acc_clean = total_clean_correct * 100.0 / total_sample
    label = torch.cat(label)

    info_string = "Clean Acc: {:.4f} total sample : {}".format(
        acc_clean, total_sample
    )
    print(info_string)
    return label



def eval_all_classes(netC, test_dl, opt, ):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = torch.zeros(10).cuda()

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            preds_clean = netC(inputs)
            for i in range(10):
                total_clean_correct[i] += torch.sum(
                    torch.argmax(preds_clean, 1) == (torch.ones_like(targets) * i).cuda())

    print("in test dl total sample are {}".format(total_sample))
    total_clean_correct = total_clean_correct  / total_sample

    info_string = "10 classes predict rate: {} ".format(
        total_clean_correct
    )
    print(info_string)

    return total_clean_correct

def trainb(netC, optimizerC, train_dl, opt=None):
    criterion = torch.nn.BCELoss()

    for batch_idx, (inputs, targets) in enumerate(train_dl):


        optimizerC.zero_grad()


        total_inputs, total_targets = inputs.to('cuda'), targets.to('cuda')

        #print(total_inputs.shape)

        total_preds = netC(total_inputs)
        #print(total_preds.shape)
        total_preds = torch.sigmoid_(total_preds)
        #print(total_preds)

        loss_ce = 1 * criterion(total_preds.squeeze(1).float(), total_targets.float())

        loss = loss_ce

        loss.backward()

        optimizerC.step()

    #sch.step()
    print(loss)



def train(netC, optimizerC,  schedulerC, train_dl):
    #print(" Train:")
    netC.train()
    criterion = torch.nn.CrossEntropyLoss()
    #transform = PostTensorTransform(opt)
    total_clean_correct = 0
    if isinstance(train_dl, Dataset):
        train_dl = DataLoader(train_dl, batch_size=64, shuffle=True)
    else:
        pass
    bs = 0

    for batch_idx, (inputs,  targets) in enumerate(train_dl):
        #print(inputs.shape)
        optimizerC.zero_grad()
        total_inputs, total_targets = inputs.to('cuda'), targets.to('cuda')

        #total_inputs = transform(total_inputs)

        total_preds = netC(total_inputs)

        total_clean_correct += torch.sum(torch.argmax(total_preds, 1) == total_targets)
        bs += inputs.shape[0]
        # print(total_clean_correct / bs)

        loss_ce = 1 * criterion(total_preds, total_targets)

        loss = loss_ce

        loss.backward()

        optimizerC.step()

    schedulerC.step()


def train_noaug(netC, optimizerC,  schedulerC, train_dl, opt):
    #print(" Train:")
    netC.train()
    criterion = torch.nn.CrossEntropyLoss()
    transform = PostTensorTransform(opt)
    total_clean_correct = 0
    bs = 0

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        optimizerC.zero_grad()
        total_inputs, total_targets = inputs.to(opt.device), targets.to(opt.device)

        #total_inputs = transform(total_inputs)

        total_preds = netC(total_inputs)

        total_clean_correct += torch.sum(torch.argmax(total_preds, 1) == total_targets)
        bs += inputs.shape[0]
        # print(total_clean_correct / bs)

        loss_ce = 1 * criterion(total_preds, total_targets)

        #print(loss_ce)
        #print(total_targets)
        #loss =

        loss = loss_ce

        loss.backward()

        optimizerC.step()

    #schedulerC.step()

def train_for_loss(netC, optimizerC, schedulerC, train_dl, opt):
    #print(" Train:")
    netC.train()
    criterion = torch.nn.CrossEntropyLoss()
    transform = PostTensorTransform(opt)
    total_clean_correct = 0
    bs = 0

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        optimizerC.zero_grad()
        total_inputs, total_targets = inputs.to(opt.device), targets.to(opt.device)

        #total_inputs = transform(total_inputs)

        total_preds = netC(total_inputs)

        total_clean_correct += torch.sum(torch.argmax(total_preds, 1) == total_targets)
        bs += inputs.shape[0]
        # print(total_clean_correct / bs)

        loss_ce = 1 * criterion(total_preds, total_targets)

        #print(loss_ce)
        #print(total_targets)
        #loss =

        loss = loss_ce

        loss.backward()

        optimizerC.step()

    # print(total_clean_correct/bs)
    if schedulerC:
        schedulerC.step()


class LabelSortedDataset(ConcatDataset):
    def __init__(self, dataset: Dataset):
        self.orig_dataset = dataset
        self.by_label = {}
        for i, (_, y) in enumerate(dataset):
            self.by_label.setdefault(y, []).append(i)

        self.n = len(self.by_label)
        assert set(self.by_label.keys()) == set(range(self.n))
        self.by_label = [Subset(dataset, self.by_label[i]) for i in range(self.n)]
        super().__init__(self.by_label)

    def subset(self, labels: Iterable[int]) -> ConcatDataset:
        if isinstance(labels, int):
            labels = [labels]
        return ConcatDataset([self.by_label[i] for i in labels])


class Custom_dataset_for_unlearning(Dataset):
    def __init__(self, dataset, Flag = 0):
        self.dataset = dataset
        self.F = Flag
        self.get_flag()

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def get_flag(self):
        dataset_ = list()
        for i in range(len(self.dataset)):
            img, label = self.dataset[i]
            dataset_.append((img, label, self.F))
        self.dataset = dataset_

class Custom_dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.full_dataset = self.dataset

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def filter(self, target):
        dataset_ = list()
        for i in range(len(self.full_dataset)):
            img, label = self.full_dataset[i]
            if label in target:
                continue
            dataset_.append((img, torch.tensor(label)))
        self.dataset = dataset_

    def get_flag(self, target):
        dataset_ = list()
        count = 0
        for i in range(len(self.full_dataset)):
            img, label = self.full_dataset[i]
            if label == target and count<100:
                dataset_.append((img, label, 1))
                count += 1
            else:
                dataset_.append((img, label, 0))
        self.dataset = dataset_


    def changelabel(self, target=1):
        dataset_ = list()
        for i in range(len(self.full_dataset)):
            img, label = self.full_dataset[i]
            dataset_.append((img, target))
        self.dataset = dataset_

    def randomlabel(self, target_label):
        label_nf = list(range(10))
        label_nf.remove(target_label)
        dataset_ = list()
        for i in range(len(self.full_dataset)):
            img, label = self.full_dataset[i]
            import random

            target = random.choice(label_nf)
            dataset_.append((img, torch.tensor(target)))
        self.dataset = dataset_

    def resetlabel(self):
        dataset_ = list()
        for i in range(len(self.full_dataset)):
            img, label = self.full_dataset[i]
            dataset_.append((img, label))
        self.dataset = dataset_

    def changelabelnf(self, target = 100):
        dataset_ = list()
        for i in range(len(self.dataset)):
            img,   label = self.dataset[i]
            dataset_.append((img,  target))
        self.dataset = dataset_

    def reset(self):

        self.dataset = self.full_dataset

    def spilt(self, ind):
        dataset_1 = list()
        dataset_2 = list()
        print(len(self.full_dataset))
        for i in range(len(self.full_dataset)):
            img,   label = self.full_dataset[i]
            if ind[i] == 1:
                dataset_1.append((img,  label))
            else:
                dataset_2.append((img,  label))
        return dataset_1, dataset_2

    def remove_oneclass(self, target_label, gm=True):
        dataset_r = []
        dataset_f = []
        for i in range(len(self.full_dataset)):
            img, label = self.full_dataset[i]
            if label == target_label:
                #print(label)
                dataset_f.append((img, label))
            else:
                if label > target_label and gm :
                    label = label - 1
                dataset_r.append((img, label))

        return dataset_r, dataset_f

    def adv_maker(self, netC, target_label):

        netC.eval()

        total_preds_logits = torch.zeros(10).to('cuda')

        total_ad_correct = 0

        dataset_ = list()

        loss_l = torch.nn.CrossEntropyLoss()

        for i in range(len(self.full_dataset)):

            inputs, targets = self.full_dataset[i]

            inputs, t_targets = inputs.to('cuda'), torch.tensor(targets).to('cuda')
            total_inputs = inputs.unsqueeze(0)



            total_inputs_orig = inputs.clone().detach()
            total_inputs.requires_grad = True

            eps = 8. / 255
            alpha = eps / 1

            for iteration in range(1):
                optimx = torch.optim.SGD([total_inputs], lr=1.)
                optimx.zero_grad()
                output = netC(total_inputs)
                #target = torch.argmax(output)

                loss = -loss_l(output, t_targets.unsqueeze(0))

                loss.backward()

                total_inputs.grad.data.copy_(alpha * torch.sign(total_inputs.grad))
                optimx.step()
                total_inputs = torch.min(total_inputs, total_inputs_orig + eps)
                total_inputs = torch.max(total_inputs, total_inputs_orig - eps)
                total_inputs = total_inputs.clone().detach()
                total_inputs.requires_grad = True

            total_inputs.requires_grad = False
            adv_inputs = total_inputs.clone().detach()

            preds = netC(adv_inputs)
            import torch.nn.functional as F
            logits = F.softmax(preds, dim=1)

            total_ad_correct += torch.sum(torch.argmax(preds, 1) == t_targets.unsqueeze(0))

            total_preds_logits += logits.squeeze(0)

            progress_bar(i, len(self.full_dataset))

            preds[:, target_label] = 0
            logits[:, target_label] = 0

            vals, inds = torch.topk(logits, k=3, dim=1, largest=True)



            t = int(torch.argmax(preds, 1))
            # print(t)
            # print(targets)

            # dataset_.append((adv_inputs.squeeze(0).clone().detach().cpu(), t))
            dataset_.append((inputs.clone().detach().cpu(), t))

        acc_ad = total_ad_correct * 100.0 / 5000

        total_preds_logits = total_preds_logits / 5000


        print(total_preds_logits)

        print(acc_ad)


        self.dataset = dataset_






def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(" [")
    for i in range(cur_len):
        sys.stdout.write("=")
    sys.stdout.write(">")
    for i in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write("]")

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    if msg:
        L.append(" | " + msg)

    msg = "".join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(" ")

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write("\b")
    sys.stdout.write(" %d/%d " % (current + 1, total))

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()

class Normalize:
    def __init__(self,  expected_values, variance):
        self.n_channels = 3
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = (x[:, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone





def my_norm(image):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.261)
    normalize = Normalize(mean, std)

    return normalize(image)

class Denormalize:
    def __init__(self,  expected_values, variance):
        self.n_channels = 3
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = x[:, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone

def eval_adverarial_attack(netC, test_dl, opt=None, eps= None):
    #print(" Eval:")
    netC.eval()

    total_ad_correct = 0
    total_sample = 0

    maxiter = 5
    eps = 4/255
    alpha = eps / 3
    ad_ro = []

    if isinstance(test_dl, Dataset):
        test_dl = DataLoader(test_dl, batch_size=64, shuffle=False)
    else:
        pass

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        if True:
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            bs = inputs.shape[0]
            total_sample += bs
            total_targets = targets
            total_inputs = inputs
            netC.eval()
            total_inputs_orig = total_inputs.clone().detach()

            total_inputs.requires_grad = True
            labels = total_targets
            
            for iteration in range(maxiter):
                optimx = torch.optim.SGD([total_inputs], lr=1.)
                optim = torch.optim.SGD(netC.parameters(), lr=1.)
                optimx.zero_grad()
                optim.zero_grad()
                output = netC(total_inputs)
                pgd_loss = -1 * torch.nn.functional.cross_entropy(output, labels)
                pgd_loss.backward()

                total_inputs.grad.data.copy_(alpha * torch.sign(total_inputs.grad))
                optimx.step()
                total_inputs = torch.min(total_inputs, total_inputs_orig + eps)
                total_inputs = torch.max(total_inputs, total_inputs_orig - eps)
                # total_inputs = th.clamp(total_inputs, min=-1.9895, max=2.1309)
                total_inputs = total_inputs.clone().detach()
                total_inputs.requires_grad = True
            
            optimx.zero_grad()
            optim.zero_grad()

            with torch.no_grad():
                preds_ad = netC(total_inputs)

                preds_ad = preds_ad.to("cpu")
                preds_cl = netC(total_inputs_orig)
                preds_cl = preds_cl.to("cpu")
                for i in range(len(preds_ad)):
                    ad_ro.append(torch.nn.functional.cross_entropy(preds_ad[i].unsqueeze(0), preds_cl[i].unsqueeze(0)))

                total_ad_correct += torch.sum(torch.argmax(preds_ad, 1) == total_targets.cpu())


    acc_ad = total_ad_correct * 100.0 / total_sample

    info_string = "Adversarial attack Acc: {:.4f} ".format(
        acc_ad
    )
    print(info_string)
    ad_ro = torch.stack(ad_ro, dim=0)
    return ad_ro

def actv_dist(model1, model2, dataloader, device = 'cuda'):
    sftmx = nn.Softmax(dim = 1)
    distances = []
    import torch.nn.functional as F
    if isinstance(dataloader, Dataset):
        dataloader = DataLoader(dataloader, batch_size=64, shuffle=False)
    else:
        pass
    for batch in dataloader:
        x, _ = batch
        x = x.to(device)
        model1_out = model1(x)
        model2_out = model2(x)
        diff = torch.sqrt(torch.sum(torch.square(F.softmax(model1_out, dim = 1) - F.softmax(model2_out, dim = 1)), axis = 1))
        diff = diff.detach().cpu()
        distances.append(diff)
    distances = torch.cat(distances, axis = 0)
    return distances.mean()

