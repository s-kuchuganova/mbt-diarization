import os
import time

import numpy as np
import torch
from tqdm import tqdm

import data
import mixup_breakdown
import pit_criterion
import tasnet
import train_utils

# TODO:Add evaluation in every train script

# run this bash command
# !cp "TasNet/src/tasnet.py" .
# import TasNet

# !cp "TasNet/src/pit_criterion.py" .
# import pit_criterion
device = 'cuda:0'
EPS = 1e-8
EPOCHS = 10
labeled_dataset = data.AudioDataset('wav8k/train', batch_size=1)
labeled_loader = data.AudioDataLoader(labeled_dataset, pin_memory=True, num_workers=2)

unlabeled_dataset = data.EvalDataset('wav8k/test', 'mix', batch_size=1)
l_unlabeled_dataset = data.EvalDataset('wav8k/train', 'mix',
                                       batch_size=1)  # на самом деле он labeled, но мы берем только mix, без s1 s2
unlabeled_dataset = unlabeled_dataset + l_unlabeled_dataset
unlabeled_loader = data.EvalDataLoader(unlabeled_dataset, shuffle=True, pin_memory=True, num_workers=2)

student_model = tasnet.TasNet(40, 100, 100, 4, bidirectional=True, nspk=2)
teacher_model = tasnet.TasNet(40, 100, 100, 4, bidirectional=True, nspk=2)

student_model.to(device)
teacher_model.to(device)
student_optimizer = torch.optim.Adam(student_model.parameters(),
                                     lr=1e-3,
                                     weight_decay=0.0)

save_folder = '/media/storage1/kuchuganova/MBT_checkpoints/'

lmbd = np.random.beta(1, 1)

total_loss = 0
for epoch in range(EPOCHS):

    start = time.time()
    total_correctness_loss = 0
    max_norm = 5  # уточнить что это
    print_freq = 1

    print('-------SUPERVISED LEARNING EPOCH {} ---------'.format(epoch))

    student_model.train()
    for i, (data) in enumerate(tqdm(labeled_loader)):
        padded_mixture, mixture_lengths, padded_source = data
        padded_mixture = padded_mixture.to(device)

        student_pred = student_model(padded_mixture, mixture_lengths)

        mixture_lengths = mixture_lengths.to(device)
        padded_source = padded_source.to(device)
        correctness_loss, max_snr, estimate_source, reorder_estimate_source = pit_criterion.cal_loss(padded_source,
                                                                                                     student_pred,
                                                                                                     mixture_lengths)

        student_optimizer.zero_grad()
        correctness_loss.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(),
                                       max_norm)
        student_optimizer.step()

        total_correctness_loss += np.array(correctness_loss.item())

    print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
          'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(epoch + 1, i + 1, total_correctness_loss / (i + 1),
                                                           np.array(correctness_loss.item()),
                                                           1000 * (time.time() - start) / (i + 1)), flush=True)

    total_consist_loss = 0
    teacher_model.eval()

    print("-------UNSUPERVISED LEARNING EPOCH {} ------------".format(epoch))

    for i, (data) in enumerate(tqdm(unlabeled_loader)):
        padded_mixture, mixture_lengths, padded_source = data
        padded_mixture = padded_mixture.to(device)

        teacher_pred = teacher_model(padded_mixture, mixture_lengths)
        # TODO: Break возвращает два элемента: сигнал и шум, которые определяются по PSNR
        new_padded_source = mixup_breakdown.Break(lmbd, teacher_pred, mixture_lengths)
        M, N = teacher_pred.shape[2], teacher_pred.shape[3]

        # "-------BREAKDOWN----------"
        flat_estimate = train_utils.remove_pad_and_flat(teacher_pred.data,
                                                        mixture_lengths)  # каким-то образом разделяем выход сети
        # здесь я беру .data чтобы не брать градиента, лучше перевести
        # модель в режим evaluate?
        s1 = flat_estimate[0][0]
        s2 = flat_estimate[0][1]

        # "---------MIXUP---------"
        Mix_s1_s2 = mixup_breakdown.Mix(lmbd, s1, s2)
        Mix_s1_s2 = torch.reshape(torch.Tensor(Mix_s1_s2), (1, mixture_lengths, N))
        Mix_s1_s2 = Mix_s1_s2.to(device)
        # mixture_lengths = mixture_lengths.cpu()

        student_estimate_source = student_model(Mix_s1_s2, mixture_lengths)

        new_padded_source = new_padded_source.to(device)
        mixture_lengths = mixture_lengths.to(device)

        consist_loss, max_snr, estimate_source, reorder_estimate_source = pit_criterion.cal_loss(new_padded_source,
                                                                                                 student_estimate_source,
                                                                                                 mixture_lengths)

        total_consist_loss += consist_loss

    LOSS = 1 / len(labeled_loader) * total_correctness_loss + np.exp(epoch + 1 / EPOCHS - 1) / (
            len(labeled_loader) + len(unlabeled_loader)) * total_consist_loss
    # if not cross_valid:
    student_optimizer.zero_grad()
    LOSS.backward()
    torch.nn.utils.clip_grad_norm_(student_model.parameters(),
                                   max_norm)
    # update weights
    student_optimizer.step()

    total_loss += np.array(LOSS.item())

    if i % 10 == 0:
        file_path = os.path.join(
            save_folder, 'epoch%d.pth.tar' % (epoch + 1))
        torch.save(student_model.serialize(student_model, student_optimizer, epoch + 1),
                   file_path)
        print('Saving checkpoint model to %s' % file_path)
    # LOSS to cpu
    train_utils.update_ema_variables(student_model, teacher_model, 0.999, epoch + 1)
    if i % print_freq == 0:
        print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
              'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
            epoch + 1, i + 1, total_loss / (i + 1),
            np.array(LOSS.item()), 1000 * (time.time() - start) / (i + 1)),
            flush=True)
