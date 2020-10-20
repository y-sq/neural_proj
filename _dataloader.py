import torch
from torch.utils.data import *
from matplotlib import pyplot as plt
import numpy as np


def read_data(start, end, data_path, label_path, TS, P_N, DIM, error_thre = -1, thre_type = "l2"):

    N = end - start

    #  N * (P_N*DIM)
    data = torch.ones([N*TS, P_N, DIM])
    label = torch.ones([N*TS, P_N, DIM])
    for i in range(N):
        data_f = open(data_path(i+start), "r")
        label_f = open(label_path(i+start), "r")
        for t in range(TS):
            for j in range(P_N):
                for k in range(DIM):
                    data[i*TS + t, j, k] = float(data_f.readline())
                    label[i*TS + t, j, k] = float(label_f.readline())
        data_f.close()
        label_f.close()
    
    # remove data with high difference between label and data
    if (error_thre > 0):
        if (thre_type == "l2"):
            errors = np.sum((data.numpy()-label.numpy())**2, axis=(1,2))
        elif (thre_type == "l1"): 
            errors = np.sum((np.abs(data.numpy()-label[:, :P_N, :].numpy())), axis=(1,2))
        print(errors.size)
        index = np.argwhere(errors < error_thre)
        index = index[:, 0]
        print(index.shape)
        data = data[index, :, :]
        label = label[index, :, :]

    sum_diff = 0
    for i in range(data.shape[0]):
        c = (data[i, :, :] - label[i, :, :])**2
        sum_diff += sum(sum(c))
    print(sum_diff / data.shape[0])
    print(len(data))
    return data, label


def read_data_with_samples(start, end, data_path, label_path, TS, P_N, DIM, error_thre = -1, thre_type = "l2", PN_A=-1, sample_idx=[]):

    print("Read sampled data.")
    
    N = end - start

    data = torch.ones([N*TS, P_N, DIM])
    label = torch.ones([N*TS, P_N, DIM])
    for i in range (N):
        data_f = open(data_path(i+start), "r") 
        label_f = open(label_path(i+start), "r") 
        for t in range (TS):
            t_j = 0
            for j in range (PN_A):
                if (j in sample_idx):
                    for k in range (DIM):
                        # print(t, j, t_j)
                        data[i*TS + t, t_j, k] = float(data_f.readline())
                        label[i*TS + t, t_j, k] = float(label_f.readline())
                    t_j += 1
                else:
                    for k in range (DIM):
                        data_f.readline()
                        label_f.readline()
        data_f.close()
        label_f.close()
    
    # remove data with high difference between label and data
    if (error_thre > 0):
        if (thre_type == "l2"):
            errors = np.sum((data.numpy()-label.numpy())**2, axis=(1,2))
        elif (thre_type == "l1"): 
            errors = np.sum((np.abs(data.numpy()-label[:, :P_N, :].numpy())), axis=(1,2))
        print(errors.size)
        index = np.argwhere(errors < error_thre)
        index = index[:, 0]
        print(index.shape)
        data = data[index, :, :]
        label = label[index, :, :]

    sum_diff = 0
    for i in range(data.shape[0]):
        c = (data[i, :, :] - label[i, :, :])**2
        sum_diff += sum(sum(c))
    print(sum_diff / data.shape[0])
    print(len(data))
    return data, label



def get_data_loader(start, end, data_path, label_path, TS, P_N, DIM, split=0.8, error_thre = -1, thre_type = "l2", PN_A=-1, sample_idx=[]):
    if (PN_A < 0):
        d, l = read_data(start=start, end=end, data_path=data_path,
                         label_path=label_path, TS=TS, P_N=P_N, DIM=DIM, error_thre = error_thre, thre_type = thre_type)
    else:
        d, l = read_data_with_samples(start=start, end=end, data_path=data_path,
                                      label_path=label_path, TS=TS, P_N=P_N, DIM=DIM, error_thre = error_thre, thre_type = thre_type,
                                      PN_A = PN_A, sample_idx = sample_idx)

    num = d.shape[0]
    idx = int(num * split)
    d_tr = d[0:idx, :, :].cuda()
    d_val = d[idx:len(d), :, :].cuda()
    print(d_tr.shape)
    print(d_val.shape)
    l_tr = l[0:idx, :, :].cuda()
    l_val = l[idx:len(d), :, :].cuda()
    train_ds = TensorDataset(d_tr, l_tr)
    val_ds = TensorDataset(d_val, l_val)
    return train_ds, val_ds


def plot(d, l):
    plt.scatter(d[:, 0, 0], d[:, 0, 1], c='r')
    plt.scatter(l[:, 0, 0], l[:, 0, 1], c='b')
    plt.show()


