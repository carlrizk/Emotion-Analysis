import pandas
import librosa
import os
import numpy as np
from tqdm import tqdm
from constants import AUDIO_TRIM_DB, CHECKPOINT_PATH, CHECKPOINT_PREFIX, N_MFCC

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class IEMOCAP:

    def __init__(self, path = "./datatset/", sequence_length = 32) -> None:
        self.root_path = path
        self.samplerate = 16_000
        self.frame_length = 512 # Corresponds to 32ms
        self.hop_length = self.frame_length // 4
        self.columns_to_split=['mfcc']
        self.sequence_length = sequence_length

        self._checkpoint = self.get_last_checkpoint()

        if(self._checkpoint == -1):
            print("Loading CSV:", path)
            self._dataframe = self.load_csv(self.root_path + "iemocap.csv")
            self.create_checkpoint()
        else:
            self.load_checkpoint(self._checkpoint)
            
        if(self._checkpoint == 0):
            print("Cleaning emotions")
            self.clean_emotions()
            self.create_checkpoint()
        
        if(self._checkpoint == 1):
            print("Generating MFCC with dim", N_MFCC)
            self.generate_mfccs(N_MFCC)
            self.create_checkpoint()

        if(self._checkpoint == 2):
            print("Splitting data into sequences of length", self.sequence_length)
            self.split()

        print("Loaded ", len(self._dataframe.index), "rows")
        

    def get_training(self):
        set = self._dataframe[self._dataframe.session != 5]
        train_set = set.sample(frac = 0.8)
        test_set = set.drop(train_set.index)
        return train_set, test_set

    def get_validation(self):
        return self._dataframe[self._dataframe.session == 5]

    def split(self):

        columns = self._dataframe.columns.values

        new_data = {}
        for col in columns:
            new_data[col] = []

        columns = np.setdiff1d(columns, self.columns_to_split)

        for _, row in tqdm(self._dataframe.iterrows()):
            for col_split in self.columns_to_split:
                data = row[col_split]
                last_cut = data.shape[0] % self.sequence_length
                if last_cut < self.sequence_length // 2:
                    data = data[:data.shape[0] - last_cut, :]
                    last_cut = data.shape[0] % self.sequence_length

                if(data.shape[0] == 0):
                    break
                
                data = np.pad(data, ((0, self.sequence_length - last_cut), (0, 0)), mode="wrap")

                number_cuts = data.shape[0] // self.sequence_length
                for i in range(number_cuts):
                    for col in columns:
                        new_data[col].append(row[col])
                    new_data[col_split].append(data[i * self.sequence_length: (i + 1) * self.sequence_length, :])
                
        
        self._dataframe = pandas.DataFrame(data=new_data)

    def generate_mfccs(self, dim):
        mfccs = []
        mfcc_max = []
        mfcc_mean = []

        for wav_path in tqdm(self._dataframe.wav_path):
            audio, sr = librosa.load(self.root_path + wav_path, sr=self.samplerate)
            audio, _ = librosa.effects.trim(
                audio,
                top_db=AUDIO_TRIM_DB,
                frame_length=self.frame_length,
                hop_length=self.hop_length
            )
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=dim,
                n_fft=self.frame_length,
                hop_length=self.hop_length
            )
            mfcc = np.transpose(mfcc)
            
            mfcc_mean.append(mfcc.mean())
            mfcc_max.append(mfcc.max())
            mfccs.append(mfcc)

        self._dataframe = self._dataframe.assign(mfcc = mfccs, mfcc_max = mfcc_max, mfcc_mean = mfcc_mean)

    def clean_emotions(self):
        emo_supp = ['xxx', 'oth', 'neu']
        self._dataframe = self._dataframe[~self._dataframe.emotion.isin(emo_supp)]
    
    def get_last_checkpoint(self):
        checkpoint = -1
        while os.path.exists(CHECKPOINT_PATH + CHECKPOINT_PREFIX + str(checkpoint + 1)):
            checkpoint += 1
        return checkpoint

    def create_checkpoint(self):
        self._checkpoint += 1
        print("Creating checkpoint", self._checkpoint)
        self.save(CHECKPOINT_PREFIX + str(self._checkpoint))

    def load_checkpoint(self, checkpoint):
        print("Loading checkpoint:", checkpoint)
        self.load(CHECKPOINT_PATH + CHECKPOINT_PREFIX + str(checkpoint))

    def save(self, name):
        self._dataframe.to_pickle(CHECKPOINT_PATH + name, protocol=4, compression='gzip')

    def load(self, name):
        self._dataframe = pandas.read_pickle(name, compression="gzip")
        

    def load_csv(self, path):
        dataframe = pandas.read_csv(path)
        dataframe = dataframe.rename(columns={"Unnamed: 0": "ID"})
        dataframe = dataframe[['ID', 'session', 'emotion', 'gender', 'wav_path']]
        return dataframe
