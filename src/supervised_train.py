# TODO: конфиг и парсер аргументов
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import LabeledDataset
from pit_criterion import cal_loss
from tasnet import TasNet

# TODO: exps with hidden size
mix_data = pd.read_csv('/second_4tb/kuchuganova/other/MiniLibriMix/metadata/mixture_train_mix_clean.csv')
mix_data_list = mix_data.to_dict('records')

device = 'cuda:0'

model = TasNet(L=1)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
print_freq = 1
max_norm = 5

dataset = LabeledDataset(mix_data_list)
data_loader = DataLoader(dataset, batch_size=1)


def run_one_epoch(epoch, cross_valid=False):
    start = time.time()
    total_loss = 0

    for i, (data) in enumerate(data_loader):
        padded_mixture, mixture_lengths, padded_source = data['mix'], data['length'], data['sources']
        padded_mixture = padded_mixture.to(device)
        padded_source = padded_source.to(device)
        estimate_source = model(padded_mixture, mixture_lengths)
        mixture_lengths = mixture_lengths.to(device)

        loss, max_snr, estimate_source, reorder_estimate_source = \
            cal_loss(padded_source, estimate_source, mixture_lengths)
        if not cross_valid:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm)
            optimizer.step()

        total_loss += loss.item()

        if i % print_freq == 0:
            print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                  'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                epoch + 1, i + 1, total_loss / (i + 1),
                loss.item(), 1000 * (time.time() - start) / (i + 1)),
                flush=True)

    return total_loss / (i + 1)


if __name__ == "__main__":
    for epoch in range(10):
        run_one_epoch(epoch=epoch)
    torch.save(model, '/second_4tb/kuchuganova/other/tasnet_model.ckpt')

