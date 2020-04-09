import os
import torch
from itertools import combinations
import six
import collections
from tqdm import tqdm
import numpy as np

np.set_printoptions(threshold=10010)

def check_uniqueness(iterable_list):
    flag = 0
    for array1, array2 in combinations(iterable_list, 2):
        for i in range(len(array1)):
            for j in range(len(array2)):
                if array1[i] == array2[j]:
                    print("Ouch! ", i, "th element of array 1 with value ", str(array1[i]), " and ", str(j), "th element of array 2 with value ", str(array2[j]))
                    flag += 1

    if flag == 0:
        print("All is well!")
    if flag != 0:
        print(str(flag) + " repetitions... Reformulate data points!")

def flatten_grad(grad):
    tuple_to_list = []
    for tensor in grad:
        tuple_to_list.append(tensor.view(-1))
    all_flattened = torch.cat(tuple_to_list)
    return all_flattened

def find_hessian(loss, model):
    grad1 = torch.autograd.grad(loss, model.parameters(), create_graph=True) #create graph important for the gradients

    grad1 = flatten_grad(grad1)
    list_length = grad1.size(0)
    hessian = torch.zeros(list_length, list_length)

    for idx in range(list_length):
            print("{} / {}".format(idx, list_length))
            grad2rd = torch.autograd.grad(grad1[idx], model.parameters(), create_graph=True)
            cnt = 0
            for g in grad2rd:
                g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
                cnt = 1
            hessian[idx] = g2.detach().cpu()
            del g2

    H = hessian.cpu().data.numpy()
    # calculate every element separately -> detach after calculating all 2ndgrad from this 1grad
    return H

def find_heigenvalues(loss, model):
    H = find_hessian(loss, model)
    eigenvalues = np.linalg.eigvalsh(H)
    return eigenvalues

def save_to_file(list, filename, folder_name = None):
    if folder_name is None:
        folder_name = 'tmp'

    if isinstance(list, torch.Tensor):
        if list.requires_grad is True:
            list = list.detach().numpy()

    list_to_string = np.array2string(np.asarray(list), separator=' ', max_line_width=np.inf)
    list_wo_brackets = list_to_string.translate({ord(i): None for i in '[]'})
    file = open(folder_name + '/' + filename, 'w')
    file.write(list_wo_brackets)
    file.close()

def append_to_file(list, filename, folder_name = None, delimiter = ' '):
    if folder_name is None:
        folder_name = 'tmp'

    if isinstance(list, torch.Tensor):
        if list.requires_grad is True:
            list = list.detach().numpy()

    list_to_string = np.array2string(np.asarray(list), separator=delimiter, max_line_width=np.inf)
    list_wo_brackets = list_to_string.translate({ord(i): None for i in '[]'})

    file = open(folder_name + '/' + filename, 'a')
    file.write("\n")
    file.close()

    file = open(folder_name + '/' + filename, 'a')
    file.write(list_wo_brackets)
    file.close()

def remove_slash(path):
    return path[:-1] if path[-1] == '/' else path

def create_progressbar(end, desc='', stride=1, start=0):
    return tqdm(six.moves.range(int(start), int(end), int(stride)), desc=desc, leave=False)

# nD list to 1D list
def flatten(list):
    if isinstance(list, collections.Iterable):
        return [a for i in list for a in flatten(i)]
    else:
        return [list]
