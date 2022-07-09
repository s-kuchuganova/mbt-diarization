from pathlib import Path

import pandas as pd
import torchaudio
from torch.utils.data import Dataset
import numpy as np

ROOT_DIR = '/second_4tb/kuchuganova/other'

mix_data = pd.read_csv('/second_4tb/kuchuganova/other/MiniLibriMix/metadata/mixture_train_mix_clean.csv')
mix_data_list = mix_data.to_dict('records')


# TODO: same size for every audio
# TODO: audio normalization?
class LabeledDataset(Dataset):
    def __init__(self, data_list):
        # TODO: del nan values in list
        self.data_list = data_list

    def __getitem__(self, idx):
        mix, minx_sr = torchaudio.load(Path(ROOT_DIR, self.data_list[idx]['mixture_path']), channels_first=False)
        s1, s1_sr = torchaudio.load(Path(ROOT_DIR, self.data_list[idx]['source_1_path']), channels_first=True)
        s2, s2_sr = torchaudio.load(Path(ROOT_DIR, self.data_list[idx]['source_2_path']), channels_first=True)
        len = self.data_list[idx]['length']
        sources = np.dstack((s1, s2))
        sources = np.transpose(sources, (2, 1, 0))  # C x L x K


        return {'mix': mix, 'sources': sources, 'length': len}

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    dataset = LabeledDataset(mix_data_list)
    print(dataset.__getitem__(1)['mix'].shape)
