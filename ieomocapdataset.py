from torch.utils.data import Dataset
from utils import get_emotion_class
import numpy as np

class IEMOCAPDataset(Dataset):
    def __init__(self, df) -> None:
        super().__init__()

        self._dataset = df
        self.clean_data()

        self._dataset.reset_index(drop=True, inplace=True)

        self.length = len(self._dataset.index)


    def __len__(self):
        return self.length

    def __getitem__(self, id):
        row = self._dataset.iloc[id]
        emotion = row.emotion
        emotion = get_emotion_class(emotion)
        return (row.mfcc, np.full((1), row.mfcc_max), np.full((1), row.mfcc_mean)), emotion


    def clean_data(self):
        self._dataset = self._dataset.replace('hap', 'pos')
        self._dataset = self._dataset.replace('exc', 'pos')

        self._dataset = self._dataset.drop(self._dataset[self._dataset.emotion == "sur"].index)
        self._dataset = self._dataset.drop(self._dataset[self._dataset.emotion == "fea"].index)
        self._dataset = self._dataset.drop(self._dataset[self._dataset.emotion == "dis"].index)

