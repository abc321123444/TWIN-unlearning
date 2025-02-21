from torch.utils.data import ConcatDataset, Subset, Dataset
import torch.utils.data as Data
import config
import torch
from classifier_models import pure_model
from dataloader import get_dataloader, get_dataset, Customer_cifar10
from un import kl_un, class_un, loss_approach, noise_train
from util import get_model, eval, Custom_dataset, train, trainb, evalb, eval_adverarial_attack, \
    eval_rl, Custom_dataset_for_unlearning
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import random


def outline(data):
    Q1 = torch.quantile(data, 0.25)
    Q3 = torch.quantile(data, 0.75)
    IQR = Q3 - Q1

    # 设定上下限
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 找到离群值的索引
    outliers_low = data < lower_bound
    outliers_high = data > upper_bound

    # 限制离群值
    data = torch.where(data < lower_bound, lower_bound, data)
    data = torch.where(data > upper_bound, upper_bound, data)

    return data


i = 0


def plot(x1, x2, l1, l2, opt):
    global i
    # Choosing color palette
    # https://seaborn.pydata.org/generated/seaborn.color_palette.html
    palette = np.array(sns.color_palette("pastel", 10))
    # pastel, husl, and so on
    # print(np.sum(l1))
    # print(np.sum(l2))

    # Create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    ind1 = np.where(l1 == 1)
    ind0 = np.where(l1 == 0)
    # print*()
    # sc = ax.scatter(x1[ind1,0], x1[ind1,1], lw=0, s=40, marker = 'o',facecolors = 'none',  edgecolors='green')
    # sc = ax.scatter(x1[ind0, 0], x1[ind0, 1], lw=0, s=40, marker = 'o', facecolors = 'none',  edgecolors='blue')
    ax.scatter(x1[ind1, 0], x1[ind1, 1], marker='o', facecolors='none', edgecolors='red',
               label='easy sample in $D_{test}$')
    ax.scatter(x1[ind0, 0], x1[ind0, 1], marker='o', facecolors='none', edgecolors='green',
               label='hard sample in $D_{test}$')

    ind1 = np.where(l2 == 1)
    ind0 = np.where(l2 == 0)
    sc = ax.scatter(x2[ind1, 0], x2[ind1, 1], lw=0, s=40, c='red', marker='.', label='easy sample in $D_{f}$')
    sc = ax.scatter(x2[ind0, 0], x2[ind0, 1], lw=0, s=40, c='green', marker='.', label='hard sample in $D_{f}$')
    # Add the labels for each digit.
    txts = []
    '''
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
        txts.append(txt)
    '''
    # plt.legend()
    plt.legend(loc=4, prop={"size": 18})
    path = './pt/digits_tsne-pastel' + str(opt.target_label) + '.png'
    plt.savefig(path, dpi=80, pad_inches=0)

    i += 1
    return f, ax, txts


class Denormalize:
    def __init__(self, expected_values, variance):
        self.n_channels = 3
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = x[:, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone


def caculate_features(model, model_g, ):
    pass


def datasettotensor(dataset):
    x = []
    y = []
    for i in range(len(dataset)):
        x.append(dataset[i][0])
        y.append(dataset[i][1])
    # print(len(x))
    x = torch.stack(x, dim=0)
    y = torch.tensor(y)
    x = np.array(x)
    y = np.array(y)
    # print(x.shape)
    # print(y.shape)
    return x, y


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

    dataset_test = get_dataset(opt, train=False)
    dataset_test = Custom_dataset(dataset_test)
    dataset_test.filter([i for i in range(opt.num_classes) if i != opt.target_label])

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
    np.save(f'./exp/{opt.target_label}/retrain_indices.npy', np.where(dataset_train.bool_array == True)[0][part1_indices])
    #exit()
    dataset_t_nf = Subset(dataset_train, part1_indices)
    dataset_t_f = Subset(dataset_train, part2_indices)

    # 创建剩余数据的子集
    all_indices = set(range(len(dataset_train)))
    remaining_indices = list(all_indices - set(forget_indices[0]))
    dataset_t_r = Subset(dataset_train, remaining_indices)

    from un import get_feature
    # prepare model

    netG, _, _ = get_model(opt)
    netG.load_state_dict(torch.load(f'./exp/model.pth'))

    netC, optimizerC, schedulerC = get_model(opt)
    netC.load_state_dict(torch.load(f'./exp/{opt.target_label}/test_finetune.pth'))

    netGG, _, _ = get_model(opt)
    # real gold model
    netGG.load_state_dict(torch.load(f'./exp/{opt.target_label}/Gmodel.pth'))

    model = pure_model().to('cuda')

    optimizer = torch.optim.Adam(model.parameters(), 0.01, betas=(0.9, 0.999), eps=1e-8)

    netO, _, _ = get_model(opt)
    netO.load_state_dict(torch.load(f'./exp/{opt.target_label}/Curriculum_model.pth'))

    netTO, _, _ = get_model(opt)
    netTO.load_state_dict(torch.load(f'./exp/{opt.target_label}/Curriculum_model.pth'))

    netUL, _, _ = get_model(opt)

    print("here are evaluations")

    dataset_T, dataset_F = eval(netG, dataset_test)
    '''
    dataset_nF = []
    for tensor, label in dataset_F:
        # 将原始张量添加到新列表中
        dataset_nF.append((tensor, label))
        if(random.random()>0.2):
        # 计算张量的镜像翻转（水平翻转）
            mirrored_tensor = torch.flip(tensor, [1])

        # 将镜像翻转的张量添加到新列表中
            dataset_nF.append((mirrored_tensor, label))

    dataset_F = dataset_nF
    '''
    preds_R = get_feature(netC, dataset_t_nf, logits=False)
    preds_R = torch.nn.functional.normalize(preds_R, p=2, dim=1)

    ds_T = Custom_dataset(dataset_T)
    preds_T = get_feature(netC, ds_T, logits=False)
    label_T = torch.ones(len(preds_T))

    td_t = torch.utils.data.DataLoader(ds_T, batch_size=128, num_workers=0,
                                       shuffle=False)
    loss_t = eval_rl(netTO, td_t)

    ds_F = Custom_dataset(dataset_F)
    preds_F = get_feature(netC, ds_F, logits=False)
    label_F = torch.zeros(len(preds_F))
    td_f = torch.utils.data.DataLoader(ds_F, batch_size=128, num_workers=0,
                                       shuffle=False)
    loss_f = eval_rl(netTO, td_f)

    preds_con = torch.cat((preds_T, preds_F), dim=0)
    label_con = torch.cat((label_T, label_F), dim=0)

    print(torch.sum(label_con))

    d_F = []
    L_F = []
    for i in range(len(preds_F)):
        preds_t = preds_F[i].unsqueeze(0)
        # print(preds_t.shape)
        preds_t = preds_t.repeat(len(preds_R), 1)
        preds_t = torch.nn.functional.normalize(preds_t, p=2, dim=1)
        dis = torch.pairwise_distance(preds_t, preds_R, p=2)
        ind = torch.argsort(dis)
        dis = torch.mean(dis[ind[:5]])
        d_F.append(dis)
        L_F.append(torch.tensor(0))

    d_F = torch.stack(d_F)
    L_F = torch.stack(L_F)
    print(torch.mean(d_F))

    d_T = []
    L_T = []
    for i in range(len(preds_T)):
        preds_t = preds_T[i].unsqueeze(0)
        # print(preds_t.shape)
        preds_t = preds_t.repeat(len(preds_R), 1)
        preds_t = torch.nn.functional.normalize(preds_t, p=2, dim=1)
        dis = torch.pairwise_distance(preds_t, preds_R, p=2)
        ind = torch.argsort(dis)
        dis = torch.mean(dis[ind[:5]])
        d_T.append(dis)
        L_T.append(torch.tensor(1))

    d_T = torch.stack(d_T)
    L_T = torch.stack(L_T)
    print(torch.mean(d_T))

    adro_T = eval_adverarial_attack(netC, td_t, eps=8 / 255)
    print(torch.mean(adro_T))
    adro_F = eval_adverarial_attack(netC, td_f, eps=8 / 255)
    print(torch.mean(adro_F))
    # exit()

    d_1 = torch.cat((d_T, d_F), dim=0).cpu()
    d_2 = torch.cat((adro_T, adro_F), dim=0).cpu()
    d_3 = torch.cat((loss_t, loss_f), dim=0).cpu()

    ind = torch.argsort(d_1, descending=True)

    d_1 = outline(d_1)
    d_2 = outline(d_2)
    d_3 = outline(d_3)

    max, min = torch.max(d_1), torch.min(d_1)
    d_1 = (d_1 - min) / (max - min)
    max, min = torch.max(d_2), torch.min(d_2)
    d_2 = (d_2 - min) / (max - min)
    max, min = torch.max(d_3), torch.min(d_3)
    d_3 = (d_3 - min) / (max - min)

    d1_con = d_1
    d2_con = d_2
    d3_con = d_3

    # d1_con = torch.cat((d_1[:int(3.5 * lenth)], d_1[-lenth:]), dim=0)
    # d2_con = torch.cat((d_2[:int(3.5 * lenth)], d_2[-lenth:]), dim=0)
    # d3_con = torch.cat((d_3[:int(3.5 * lenth)], d_3[-lenth:]), dim=0)

    feature_con = torch.stack((d1_con.cpu(), d3_con.cpu()), dim=-1)

    # feature_con = torch.stack((d2_con.cpu(), d3_con.cpu()), dim=-1)
    # feature_con = d3_con.cpu().unsqueeze(-1)

    print(feature_con.shape)
    print(label_con.shape)

    dataset_train = Data.TensorDataset(feature_con, label_con)
    train_dl = torch.utils.data.DataLoader(dataset_train, batch_size=128, num_workers=0,
                                           shuffle=True)

    # dataset_T, dataset_F = eval(netGG, dataset_t_f)

    # acc_f_gm = 100.0 * len(dataset_T) / (len(dataset_T) + len(dataset_F))
    # print("ACC Forget set GM:{}".format(acc_f_gm))

    from util import getF
    label_test = getF(netGG, dataset_t_f)
    print(torch.sum(label_test))

    preds_test = get_feature(netG, dataset_t_f, logits=False)

    loss_test = eval_rl(netO, dataset_t_f)

    d_test = []
    for i in range(len(preds_test)):
        preds_t = preds_test[i].unsqueeze(0)
        # print(preds_t.shape)
        preds_t = preds_t.repeat(len(preds_R), 1)
        preds_t = torch.nn.functional.normalize(preds_t, p=2, dim=1)
        dis = torch.pairwise_distance(preds_t, preds_R, p=2)
        ind = torch.argsort(dis)
        dis = torch.mean(dis[ind[:5]])
        d_test.append(dis)
    d_test = torch.stack(d_test)

    adro_test = eval_adverarial_attack(netG, dataset_t_f, eps=8 / 255)

    d_test_1 = d_test.cpu()
    d_test_2 = adro_test.cpu()
    d_test_3 = loss_test.cpu()

    d_test_1 = outline(d_test_1)
    d_test_2 = outline(d_test_2)
    d_test_3 = outline(d_test_3)

    max, min = torch.max(d_test_1), torch.min(d_test_1)
    d_test_1 = (d_test_1 - min) / (max - min)
    max, min = torch.max(d_test_2), torch.min(d_test_2)
    d_test_2 = (d_test_2 - min) / (max - min)

    max, min = torch.max(d_test_3), torch.min(d_test_3)
    d_test_3 = (d_test_3 - min) / (max - min)

    d1_test_con = d_test_1
    d2_test_con = d_test_2

    feature_test_con = torch.stack((d1_test_con.cpu(), d_test_3.cpu()), dim=-1)
    label_test_con = label_test

    dataset_test = Data.TensorDataset(feature_test_con, label_test_con)
    test_dl = torch.utils.data.DataLoader(dataset_test, batch_size=128, num_workers=0,
                                          shuffle=False)

    if os.path.exists(f'./exp/{opt.target_label}/classifier.pth'):
        model.load_state_dict(torch.load(f'./exp/{opt.target_label}/classifier.pth'))
    else:
        for epoch in range(200):
            print("epoch is {}".format(epoch))
            trainb(model, optimizer, train_dl, opt)
            if epoch % 1 == 0:
                acc_clean, acc_t, acc_f = evalb(model, train_dl, False)
                acc_clean2, acc_t2, acc_f2 = evalb(model, test_dl, False)

        import csv
        with open('model.csv', 'a+') as file:
            writer = csv.writer(file)
            writer.writerow(
                [opt.target_label, acc_clean, float(acc_t), float(acc_f), acc_clean2, float(acc_t2),
                 float(acc_f2)])

        torch.save(model.state_dict(), f'./exp/{opt.target_label}/classifier.pth')

    torch.save(model.state_dict(), f'./exp/{opt.target_label}/classifier.pth')

    Flag = evalb(model, test_dl)

    Flag = np.array(Flag)

    dataset_per_f1 = Custom_dataset(dataset_t_f)
    ds_gT, ds_gF = dataset_per_f1.spilt(Flag)
    ds_gF = Custom_dataset(ds_gF)
    ds_gF.randomlabel(opt.target_label)

    ds_gT = Custom_dataset(ds_gT)
    ds_gF = Custom_dataset(ds_gF)

    shuffled_indices = np.random.permutation(len(dataset_t_r))
    split_point = int(len(shuffled_indices) * 0.2)

    # 分割索引数组
    part1_indices = shuffled_indices[:split_point]
    dataset_t_r_percent = Subset(dataset_t_r, part1_indices)

    ds_gTu = Custom_dataset_for_unlearning(ds_gT, Flag=1)
    ds_gFu = Custom_dataset_for_unlearning(ds_gF, Flag=2)
    print(len(ds_gT))
    print(len(ds_gF))

    dataset_gcon = ConcatDataset(
        [ds_gTu, ds_gFu, Custom_dataset_for_unlearning(dataset_t_nf), Custom_dataset_for_unlearning(dataset_t_r)])
    print(len(dataset_gcon))
    dl_train_gcon = torch.utils.data.DataLoader(dataset_gcon, batch_size=64, num_workers=opt.num_workers, shuffle=True)
    td = get_dataloader(opt, False)

    #optimizerC = torch.optim.SGD(netC.parameters(), 0.001, momentum=0.9, weight_decay=5e-4)
    optimizerC = torch.optim.SGD(netC.parameters(), 1e-2, momentum=0.9, weight_decay=5e-4)

    loss_easy = torch.mean(eval_rl(netC, ds_T))
    loss_hard = torch.mean(eval_rl(netC, ds_F))
    print(loss_easy)
    print(loss_hard)
    eval(netC, dataset_t_r)
    model.eval()
    eval(netC, ds_T, reloss=True)
    eval(netC, ds_F, reloss=True)
    eval(netC, ds_gT, reloss=True)
    for epoch in range(5):
        print("Epoch {}:".format(epoch + 1))
        # loss_easy = torch.mean(eval_rl(netC, ds_T))

        # loss_approach(netC, optimizerC, loss_easy, loss_hard, dl_train_gcon)
        noise_train(netC, optimizerC, dl_train_gcon)
        # train(netC, optimizerC, schedulerC, dl_train_gcon)

        if epoch % 1 == 0:
            eval(netC, dataset_t_nf)
            eval(netC, dataset_t_r)
            eval(netC, ds_gT)
            eval(netC, ds_gF)

    path = f'./exp/{opt.target_label}/Unlearned_model.pth'
    lists = []
    for i in range(len(dataset_t_f)):
        lists.append(dataset_t_f[i][0])

    # ts = torch.stack(lists)
    # numpy_array = ts.numpy()
    # np.save('./exp/forget_set.npy', numpy_array)

    eval(netC, ds_T, reloss=True)
    eval(netC, ds_gF, reloss=True)
    eval(netC, ds_gT, reloss=True)
    eval(netC, dataset_t_nf, reloss=True)
    eval(netC, dataset_t_r, reloss=True)

    torch.save(netC.state_dict(), path)
    '''
    import csv
    with open('data.csv', 'a+') as file:
        writer = csv.writer(file)
        writer.writerow(
            [opt.target_label, float(acc_f_gm), float(acc_test_gm), float(best_acc_f), float(best_acc_r), float(ar),
             len(ds_gT), len(ds_gF)])

    '''


if __name__ == "__main__":
    main()