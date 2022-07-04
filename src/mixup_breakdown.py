import train_utils
import data
import numpy as np
import torch


def Break(l, teacher_pred, mixture_lenghts):
    M, N = teacher_pred.shape[2], teacher_pred.shape[3]

    flat_estimate = train_utils.remove_pad_and_flat(teacher_pred.data, mixture_lengths)

    s1 = flat_estimate[0][0]
    s2 = flat_estimate[0][1]

    s1 = l * s1
    s2 = (1 - l) * s2

    sources_arr = []
    sources_arr.append(np.array(torch.reshape(torch.Tensor(s1), (1, M, N))))  # собираем 2 сигнала в 1 для pad source
    sources_arr.append(np.array(torch.reshape(torch.Tensor(s2), (1, M, N))))
    pad_value = 0
    sources_pad = data.pad_list([torch.from_numpy(s).float() for s in sources_arr], pad_value)

    new_padded_source = sources_pad.permute((1, 0, 2, 3)).contiguous()
    return new_padded_source


def Mix(l, s1, s2):
    return l * s1 + (1 - l) * s2
