import copy
import sys

import torch.utils.data as data
from torch.utils.data import Dataset, Subset
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import os
import csv
import kornia.augmentation as A
import numpy as np

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
from torchvision.datasets import CIFAR100
import random


class ToNumpy:
    def __call__(self, x):
        x = np.array(x)
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=2)
        return x


class ProbTransform(torch.nn.Module):
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):  # , **kwargs):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x




def get_transform(opt, train=True, pretensor_transform=False):
    transforms_list = []
    # transforms_list.append(transforms.ToPILImage())
    #transforms.RandomCrop(32, padding=4),
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    if pretensor_transform:
        if train:
            #transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop))
            #transforms_list.append(transforms.RandomRotation(opt.random_rotation))
            if opt.dataset == "cifar10":
                pass
                #transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))

    transforms_list.append(transforms.ToTensor())
    if opt.dataset == "cifar10" or opt.dataset == "cifar100":
        pass
        #transforms_list.append(transforms.Normalize([0.5], [0.5]))
        transforms_list.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
    elif opt.dataset == "mnist":
        #pass
        transforms_list.append(transforms.Normalize([0.5], [0.5]))
    elif opt.dataset == "gtsrb" or opt.dataset == "celeba" or opt.dataset == "imagenet" or opt.dataset == 'tini_imagenet' or opt.dataset == "vgg":
        pass
    else:
        raise Exception("Invalid Dataset")
    return transforms.Compose(transforms_list)


class GTSRB(data.Dataset):
    def __init__(self, opt, train, transforms):
        super(GTSRB, self).__init__()
        if train:
            self.data_folder = os.path.join(opt.data_root, "GTSRB/Train")
            self.images, self.labels = self._get_data_train_list()
        else:
            self.data_folder = os.path.join(opt.data_root, "GTSRB/Test")
            self.images, self.labels = self._get_data_test_list()

        self.transforms = transforms

    def _get_data_train_list(self):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                images.append(prefix + row[0])
                labels.append(int(row[7]))
                #if int(row[7]) == 2:
                    #images.append(prefix + row[0])
                    #labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def _get_data_test_list(self):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            images.append(self.data_folder + "/" + row[0])
            labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image)
        label = self.labels[index]
        return image, label


class CelebA_attr(data.Dataset):
    def __init__(self, opt, split, transforms):
        self.dataset = torchvision.datasets.CelebA(root=opt.data_root, split=split, target_type="attr", download=True)
        self.list_attributes = [18, 31, 21]
        self.transforms = transforms
        self.split = split

    def _convert_attributes(self, bool_attributes):
        return (bool_attributes[0] << 2) + (bool_attributes[1] << 1) + (bool_attributes[2])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input, target = self.dataset[index]
        input = self.transforms(input)
        #print(self.transforms)
        #print(target)
        #print(target[self.list_attributes])
        target = self._convert_attributes(target[self.list_attributes])
        return (input, target)



class Customer_cifar10(Dataset):
    def __init__(self, rate = 0.75, keep = None):
        self.inputs = np.load("tf_data/x_train.npy").transpose(0, 3, 1, 2)
        self.labels = np.load("tf_data/y_train.npy")

        member_size = int(len(self.labels) * rate)
        if keep:
            self.bool_array = np.load(keep)
        else:
            self.bool_array = np.array([True] * member_size + [False] * (len(self.labels) - member_size))
            np.random.shuffle(self.bool_array)

        self.inputs_ft = self.inputs[self.bool_array]
        self.labels_ft = self.labels[self.bool_array]

    def __getitem__(self, item):
        img = torch.tensor(self.inputs_ft[item], dtype=torch.float)
        label = torch.tensor(self.labels_ft[item])

        return img, label

    def __len__(self):
        return len(self.labels_ft)

    def get_keep(self):
        return self.bool_array

    def get_left(self):
        return self.inputs[~self.bool_array], self.labels[~self.bool_array]

    def forget_indices(self, target_label):
        indices_with_targetlabel = np.where(self.labels_ft == target_label)
        return indices_with_targetlabel




class Customer_super20(Dataset):
    def __init__(self, rate = 0.75, keep = None):
        self.inputs = np.load("tf_data/x_train.npy").transpose(0, 3, 1, 2)
        self.labels = np.load("tf_data/y_train.npy")

        member_size = int(len(self.labels) * rate)
        self.coarse_map = {
            0: [4, 30, 55, 72, 95],
            1: [1, 32, 67, 73, 91],
            2: [54, 62, 70, 82, 92],
            3: [9, 10, 16, 28, 61],
            4: [0, 51, 53, 57, 83],
            5: [22, 39, 40, 86, 87],
            6: [5, 20, 25, 84, 94],
            7: [6, 7, 14, 18, 24],
            8: [3, 42, 43, 88, 97],
            9: [12, 17, 37, 68, 76],
            10: [23, 33, 49, 60, 71],
            11: [15, 19, 21, 31, 38],
            12: [34, 63, 64, 66, 75],
            13: [26, 45, 77, 79, 99],
            14: [2, 11, 35, 46, 98],
            15: [27, 29, 44, 78, 93],
            16: [36, 50, 65, 74, 80],
            17: [47, 52, 56, 59, 96],
            18: [8, 13, 48, 58, 90],
            19: [41, 69, 81, 85, 89]
        }
        if keep:
            self.bool_array = np.load(keep)
        else:
            self.bool_array = np.array([True] * member_size + [False] * (len(self.labels) - member_size))
            np.random.shuffle(self.bool_array)

        self.inputs_ft = self.inputs[self.bool_array]
        self.labels_ft = self.labels[self.bool_array]

    def __getitem__(self, item):
        img = torch.tensor(self.inputs_ft[item], dtype=torch.float)
        label = torch.tensor(self.labels_ft[item])
        for i in range(20):
            for j in self.coarse_map[i]:
                if label == j:
                    coarse_y = i
                    break
        return img, label, coarse_y

    def __len__(self):
        return len(self.labels_ft)

    def get_keep(self):
        return self.bool_array

    def get_left(self):
        return self.inputs[~self.bool_array], self.labels[~self.bool_array]

    def forget_indices(self, target_label):
        indices_with_targetlabel = np.where(self.labels_ft == target_label)
        return indices_with_targetlabel



transform_train_vits = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])



def get_dataloader(opt, train=True, pretensor_transform=False):
    transform = get_transform(opt, train, pretensor_transform)
    if opt.dataset == "gtsrb":
        dataset = GTSRB(opt, train, transform)
    elif opt.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(opt.data_root, train, transform, download=True)
    elif opt.dataset == "cifar10":
        if train:
            dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform, download=True)
        else:
            dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform, download=True)

    elif opt.dataset == "cifar100":
        dataset = torchvision.datasets.CIFAR100('.', train, transform=transform, download=True)


    elif opt.dataset == "celeba":
        if train:
            split = "train"
        else:
            split = "test"
        dataset = CelebA_attr(opt, split, transform)

    elif opt.dataset == "imagenet":
        if train:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
            ])
            dataset = datasets.ImageFolder('/data/imgnet-100/train', transform_train)
            print(len(dataset))
            # dataset = datasets.ImageFolder('./imagenette2/train', transform_train)
            # length = len(dataset)
            # print(length)
            # train_size, validate_size = int(0.05 * length), int(0.95 * length) + 1
            # print(train_size)
            # print(validate_size)
            # dataset, validate_set = torch.utils.data.random_split(dataset, [train_size, validate_size])
            # print((dataset)[0])
        else:
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
                # 需要更多数据预处理，自己查
            ])
            dataset = datasets.ImageFolder('/data/imgnet-100/val', transform_test)
            # length = len(dataset)
            # print(length)
            # train_size, validate_size = int(0.2 * length), int(0.8 * length) #+ 1
            # print(train_size)
            # print(validate_size)
            # dataset, validate_set = torch.utils.data.random_split(dataset, [train_size, validate_size])

    elif opt.dataset == "vgg":
        if train:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
            ])
            dataset = datasets.ImageFolder('/data/vggface/train', transform)
        else:

            dataset = datasets.ImageFolder('/data/vggface/eval', transform)

    elif opt.dataset == "tini_imagenet":
        if train:
            transform_train = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
            ])
            dataset = datasets.ImageFolder('./data/tiny-imagenet-197/train', transform_train)
        else:
            transform_test = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
                # 需要更多数据预处理，自己查
            ])
            dataset = datasets.ImageFolder('./data/tiny-imagenet-197/val', transform_test)

    else:
        raise Exception("Invalid dataset")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=True)
    return dataloader




def get_dataset(opt, train=True):
    transform = get_transform(opt, train, False)
    if opt.dataset == "gtsrb":
        dataset = GTSRB(opt, train, transform)
    elif opt.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(opt.data_root, train, transform, download=True)

    elif opt.dataset == "cifar10":
        if train:
            dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform, download=True)
        else:
            dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform, download=True)

    elif opt.dataset == "cifar100":
        dataset = torchvision.datasets.CIFAR100('.', train, transform=transform, download=True)


    elif opt.dataset == "celeba":
        if train:
            split = "train"
        else:
            split = "test"
        dataset = CelebA_attr(
            opt,
            split,
            transforms=transforms.Compose([transforms.Resize((opt.input_height, opt.input_width)), ToNumpy()]),
        )
    elif opt.dataset == "imagenet":
        if train:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
            ])
            dataset = datasets.ImageFolder('/data/imgnet-100/train', transform_train)

            length = len(dataset)

            train_size, validate_size = int(0.1 * length), int(0.9 * length)

            dataset, validate_set = torch.utils.data.random_split(dataset, [train_size, validate_size])
            # print((dataset)[0])
        else:
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
                # 需要更多数据预处理，自己查
            ])
            dataset = datasets.ImageFolder('/data/imgnet-100/test', transform_test)


    elif opt.dataset == "vgg":

        if train:

            transform_train = transforms.Compose([

                transforms.RandomResizedCrop(224),

                transforms.RandomVerticalFlip(),

                transforms.ToTensor(),

                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理

            ])

            dataset = datasets.ImageFolder('/data/vgg10/train', transform)

        else:

            dataset = datasets.ImageFolder('/data/vgg10/eval', transform)

    elif opt.dataset == "tini_imagenet":
        if train:
            transform_train = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
            ])
            dataset = datasets.ImageFolder('./data/tiny-imagenet-200/train', transform_train)
        else:
            transform_test = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
                # 需要更多数据预处理，自己查
            ])
            dataset = datasets.ImageFolder('./data/tiny-imagenet-200/val', transform_test)

    elif opt.dataset == "vgg":
        if train:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
            ])
            dataset = datasets.ImageFolder('/data/vgg10/train', transform)
        else:

            dataset = datasets.ImageFolder('/data/vgg10/eval', transform)

    else:
        raise Exception("Invalid dataset")
    return dataset


class Custom_dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.full_dataset = self.dataset

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def filter(self, filter_index):
        dataset_ = list()
        for i in range(len(self.full_dataset)):
            # print(len(self.full_dataset))
            img, label, flag = self.full_dataset[i]
            if filter_index[i]:
                continue
            dataset_.append((img, label, flag))
        self.dataset = dataset_

    def addLabel(self):
        dataset_ = list()
        for i in range(len(self.dataset)):
            img, label = self.dataset[i]
            dataset_.append((img, label, 0))
        self.dataset = dataset_

    def random_filter(self, filter_index):
        dataset_ = list()
        for i in range(len(self.full_dataset)):
            img, label, flag = self.full_dataset[i]
            if filter_index[i]:
                continue
            if random.random() < 0.5 and flag == 1:
                continue
            dataset_.append((img, label, flag))
        self.dataset = dataset_

    def delete_poison(self, b):
        dataset_ = list()
        random.seed(2)
        for i in range(len(self.dataset)):
            img, label, flag = self.dataset[i]
            if random.random() < b and flag == 1:
                continue
            dataset_.append((img, label, flag))
        self.dataset = dataset_

    def semi_prepare(self):
        dataset_ = list()
        for i in range(len(self.dataset)):
            img, label, flag = self.dataset[i]
            dataset_.append((img, 1000, flag))
        self.dataset = dataset_

    def filter_2(self, filter_index):
        dataset_ = list()
        for i in range(len(self.full_dataset)):
            img, label = self.full_dataset[i]
            # img = np.array(img)
            if filter_index[i]:
                continue
            dataset_.append((img, label))
        self.dataset = dataset_

    def augfilter(self, filter_index, opt):
        transform = get_transform(opt, True, True)
        dataset_ = list()
        for i in range(len(self.full_dataset)):
            img, label, bd_label = self.full_dataset[i]
            if filter_index[i]:
                continue
            img_PIL = transforms.ToPILImage()(img)
            dataset_.append((transform(img_PIL), label, bd_label))
            dataset_.append((img, label, bd_label))
        self.dataset = dataset_

    def set_label(self, CL=True):
        dataset_ = list()
        for i in range(len(self.dataset)):
            img, label, flag = self.dataset[i]
            if CL:
                dataset_.append((img, label, 0))
            else:
                dataset_.append((img, label, 1))
        self.dataset = dataset_

    def oder_filter(self, filter_index):
        ds = torch.utils.data.Subset(self.dataset, filter_index)
        self.dataset = ds

    def set_fulldata(self, full_dataset):
        self.full_dataset = full_dataset

    def aug(self, b):
        dataset_ = list()
        for i in range(len(self.dataset)):
            img, label, flag = self.dataset[i]
            for j in range(b):
                dataset_.append((img, label, flag))
        self.dataset = dataset_

    def deleterepeat(self):
        dataset_ = list()
        for i in range(len(self.dataset)):
            img, label, flag = self.dataset[i]
            for j in range(len(dataset_)):
                if torch.equal(self.dataset[i][0], dataset_[j][0]):
                    continue
            dataset_.append(self.dataset[i])
        self.dataset = dataset_

    def shuff(self):
        from torch import randperm
        lenth = randperm(len(self.dataset)).tolist()  # 生成乱序的索引
        # print(lenth)
        ds = torch.utils.data.Subset(self.dataset, lenth)
        self.dataset = ds
        self.full_dataset = ds


class C(Dataset):
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
            # print(len(self.full_dataset))
            img, label = self.full_dataset[i]
            if label != target:
                continue
            dataset_.append((img, label))
        self.dataset = dataset_



class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(self.train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt


transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

class CustomCIFAR100(CIFAR100):
    def __init__(self, root, train, download, transform):
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.coarse_map = {

            0: [4, 30, 55, 72, 95],
            1: [1, 32, 67, 73, 91],
            2: [54, 62, 70, 82, 92],
            3: [9, 10, 16, 28, 61],
            4: [0, 51, 53, 57, 83],     #fruit
            5: [22, 39, 40,  86, 87],    #,
            6: [5, 20, 25, 84, 94],
            7: [6, 7, 14, 18, 24],
            8: [3, 42, 43, 88, 97],
            9: [12, 17, 37, 68, 76],
            10: [23, 33, 49, 60, 71],  #sea 71
            11: [15, 19, 21, 31, 38],
            12: [34, 63, 64, 66, 75],
            13: [26, 45, 77, 79, 99],
            14: [2, 11, 35, 46, 98], #baby
            15: [27, 29, 44, 78, 93],
            16: [36, 50, 65, 74, 80],
            17: [47, 52, 56, 59, 96],
            18: [8, 13, 48, 58, 90],
            19: [41, 69, 81, 85, 89]  #rocket
        }

    # def __len__(self):
    #    len(self.main_dataset)

    def __getitem__(self, index):

        x, y = super().__getitem__(index)
        coarse_y = None
        for i in range(20):
            for j in self.coarse_map[i]:
                if y == j:
                    coarse_y = i
                    break
        return x, y, coarse_y
