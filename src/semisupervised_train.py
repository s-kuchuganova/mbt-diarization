import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import mixup_breakdown
import pit_criterion
import tasnet
import train_utils
# TODO:Add evaluation in every train script
from src.dataset import LabeledDataset

device = 'cuda:1'
EPS = 1e-8
EPOCHS = 10
ckpt_save_step = 10
save_folder = '/second_4tb/kuchuganova/other/'
mix_data = pd.read_csv('/second_4tb/kuchuganova/other/MiniLibriMix/metadata/mixture_train_mix_clean.csv')
mix_data_list = mix_data.to_dict('records')

dataset = LabeledDataset(mix_data_list)
dataloader = DataLoader(dataset, batch_size=1)

student_model = tasnet.TasNet(1, 100, 100, 4, bidirectional=True, nspk=2)
teacher_model = tasnet.TasNet(1, 100, 100, 4, bidirectional=True, nspk=2)

student_model.to(device)
teacher_model.to(device)
student_optimizer = torch.optim.Adam(student_model.parameters(),
                                     lr=1e-3,
                                     weight_decay=0.0)

lmbd = np.random.beta(1, 1)

total_loss = 0
for epoch in range(EPOCHS):

    start = time.time()
    total_correctness_loss = 0
    total_consist_loss = 0
    max_norm = 5
    print_freq = 1

    print('-------STUDENT LEARNING STEP {} ---------'.format(epoch))

    for i, (data) in enumerate((dataloader)):
        student_model.eval()
        padded_mixture, mixture_lengths, padded_source = data['mix'], data['length'], data['sources']
        padded_mixture = padded_mixture.to(device)

        student_pred = student_model(padded_mixture, mixture_lengths)

        mixture_lengths = mixture_lengths.to(device)
        padded_source = padded_source.to(device)
        correctness_loss, max_snr, estimate_source, reorder_estimate_source = \
            pit_criterion.cal_loss(padded_source, student_pred, mixture_lengths)

        # student_optimizer.zero_grad()
        # correctness_loss.backward()
        # torch.nn.utils.clip_grad_norm_(student_model.parameters(),
        #                                max_norm)
        # student_optimizer.step()

        total_correctness_loss += np.array(correctness_loss.item())

        print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
              'Correctness Loss {3:.6f} | {4:.1f} ms/batch'.format(epoch + 1, i + 1, total_correctness_loss / (i + 1),
                                                                   np.array(correctness_loss.item()),
                                                                   1000 * (time.time() - start) / (i + 1)), flush=True)

        print('-------TEACHER LEARNING STEP {} ---------'.format(epoch))
        mixture_lengths = mixture_lengths.cpu()
        teacher_model.eval()
        teacher_pred = teacher_model(padded_mixture, mixture_lengths)
        teacher_pred = teacher_pred.cpu().detach().numpy()
        #MIXUP BREAKDOWN
        braked, mix = mixup_breakdown.mixup_breakdown_op(lmbd, teacher_pred)
        mix = mix.to(device)
        braked = braked.to(device)
        #STUDENT ON UNLABELED
        student_model.train()
        student_on_mix = student_model(mix, mixture_lengths)
        mixture_lengths = mixture_lengths.to(device)

        consist_loss, max_snr, estimate_source, reorder_estimate_source = pit_criterion.cal_loss(braked,
                                                                                                 student_on_mix,
                                                                                                 mixture_lengths)

        total_consist_loss += consist_loss
        print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
              'Consist Loss {3:.6f} | {4:.1f} ms/batch'.format(epoch + 1, i + 1, total_consist_loss / (i + 1),
                                                               np.array(consist_loss.item()),
                                                               1000 * (time.time() - start) / (i + 1)), flush=True)
        LOSS = 1 / len(dataloader) * total_correctness_loss + np.exp(epoch + 1 / EPOCHS - 1) / (
                len(dataloader) + len(dataloader)) * total_consist_loss
        student_optimizer.zero_grad()
        LOSS.backward()
        # torch.nn.utils.clip_grad_norm_(student_model.parameters(),
        #                                max_norm)
        # update weights
        student_optimizer.step()

        total_loss += np.array(LOSS.item())

        train_utils.update_ema_variables(student_model, teacher_model, 0.999, epoch + 1)

    if epoch % ckpt_save_step == 0:
        file_path = os.path.join(
            save_folder, 'student_epoch%d.pth' % (epoch + 1))
        torch.save(student_model.serialize(student_model, student_optimizer, epoch + 1),
                   file_path)
        torch.save(teacher_model, file_path)
        print('Saving checkpoint model to %s' % file_path)
    # LOSS to cpu
