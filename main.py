from iemocap import IEMOCAP
from ieomocapdataset import IEMOCAPDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import Model
from constants import N_MFCC, MODEL_PATH
import torch
from utils import show_distribution

SEQUENCE_LENGTH = 2 * 32

BATCH_SIZE = 1024
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 1
EPOCHS = 5
LEARNING_RATE = 1e-3

iemocap = IEMOCAP(
    sequence_length=SEQUENCE_LENGTH
)

show_distribution(iemocap._dataframe.emotion, title="Dataset")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using", device)

training_set, testing_set = iemocap.get_training()
validation_set = iemocap.get_validation()

# training_set = training_set.head(100)
# testing_set = testing_set.head(100)

training_set = IEMOCAPDataset(training_set)
testing_set = IEMOCAPDataset(testing_set)
validation_set = IEMOCAPDataset(validation_set)

show_distribution(training_set._dataset.emotion, title="Training Set")
show_distribution(testing_set._dataset.emotion, title="Testing Set")
show_distribution(validation_set._dataset.emotion, title="Validation Set")

training_dataloader = DataLoader(
    training_set,
    batch_size = BATCH_SIZE
)
testing_dataloader = DataLoader(
    testing_set,
    batch_size = testing_set.__len__()
)
validation_dataloader = DataLoader(
    validation_set,
    batch_size = validation_set.__len__()
)

model = Model(input_size = N_MFCC, lstm_hidden_size = LSTM_HIDDEN_SIZE, lstm_num_layers = LSTM_LAYERS).to(device)
optimizer = Adam(params=model.parameters(), lr = LEARNING_RATE)

model.train_model(
    training_dataloader = training_dataloader,
    testing_dataloader = testing_dataloader,
    optimizer = optimizer,
    epochs = EPOCHS,
    device = device,
    checkpoint_path= MODEL_PATH + "checkpoints/"
)

model.validate(validation_dataloader, device)

print("Saving to", MODEL_PATH)
model.save_model(MODEL_PATH)

