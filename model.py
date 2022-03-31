import torch
from tqdm import tqdm
import numpy as np
from constants import CLASSES_EMOTIONS
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import csv
import json
import os



class Model(torch.nn.Module):
    def __init__(self, input_size, lstm_hidden_size, lstm_num_layers):
        super().__init__()

        self.lstm_hidden_size = lstm_hidden_size
        self.bidirectional = True

        self.lstm = torch.nn.LSTM(
            input_size = input_size,
            hidden_size = lstm_hidden_size,
            num_layers = lstm_num_layers,
            dropout = 0,
            bidirectional = self.bidirectional,
            batch_first = True
        )

        mlp_input_size = lstm_hidden_size
        if self.bidirectional:
            mlp_input_size *= 2
        mlp_input_size += 2 # MFCC_max & MFCC_mean

        self.mlp_emotion = torch.nn.Sequential(
            torch.nn.Linear(mlp_input_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, len(CLASSES_EMOTIONS)),
            torch.nn.Softmax(dim = 1)
        )

    def forward(self, x):
        (mfcc, mfcc_max, mfcc_mean) = x
        _, (h_n, _) = self.lstm(mfcc)
        h_n = torch.cat((h_n[self.lstm.num_layers - 1], h_n[-1]), dim=1)
        features = torch.cat((h_n, mfcc_max.float(), mfcc_mean.float()), dim=1)
        y_emotion = self.mlp_emotion(features)
        return y_emotion

    def train_model(self, optimizer, training_dataloader, testing_dataloader, epochs, device, checkpoint_path = None):
        
        loss_emotion = torch.nn.CrossEntropyLoss(reduction = 'sum')
        
        history_train_loss = []
        history_test_loss = []
        history_accuracy = []

        for epoch in range(epochs):
            
            loss_train = 0.0

            self.train()
            for _, (inputs, labels) in enumerate(tqdm(training_dataloader)):
                optimizer.zero_grad()

                inputs = [x.to(device) for x in inputs]
                labels = labels.to(device)

                output = self.forward(inputs)
                loss = loss_emotion(output, labels)
                
                loss.backward()
                optimizer.step()

                # calculate statistics
                loss_train += loss.item()

            loss_train /= len(training_dataloader.dataset)

            ###################################################################""
            self.eval()
            with torch.no_grad():
                (inputs, labels) = next(iter(testing_dataloader))

                inputs = [x.to(device) for x in inputs]
                labels = labels.to(device)

                output = self.forward(inputs)
                loss = loss_emotion(output, labels)

                # calculate statistics
                loss_test = loss.item() / len(testing_dataloader.dataset)

                predicted = np.argmax(output.cpu(), axis=1)
                accuracy = accuracy_score(labels.cpu(), predicted) * 100

            history_train_loss.append(loss_train)
            history_test_loss.append(loss_test)
            history_accuracy.append(accuracy)

            print("Epoch", epoch + 1, "Loss:", "{:.12f}".format(loss_train), "Test Loss:", "{:.12f}".format(loss_test), "Accuracy:", "{:.2f}".format(accuracy))

            if(checkpoint_path):
                self.save_state(checkpoint_path + str(epoch) + "/")

        self.train_loss = history_train_loss
        self.test_loss = history_test_loss
        self.accuracy = history_accuracy

        print('Finished Training')

    def validate(self, validation_dataloader, device):
        self.eval()
        (inputs, labels) = next(iter(validation_dataloader))

        inputs = [x.to(device) for x in inputs]
        labels = labels.to(device)

        Y_pred = self.forward(inputs)
        Y_pred = Y_pred.detach().cpu()

        Y_pred = np.argmax(Y_pred, axis=1)

        self.report = classification_report(labels.cpu(), Y_pred, target_names=CLASSES_EMOTIONS, output_dict=True)
        return self.report

    def save_state(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "model.pt")

    def save_model(self, path):
        self.save_state(path)
        self.save_stats(path)
        self.save_figure(path)
        if(self.report):
            self.save_validation_report(path)


    def save_validation_report(self, path):
        os.makedirs(path, exist_ok=True)
        with open(path + "validation.json", 'w') as f:
            json.dump(self.report, f,  indent=4)

    def save_figure(self, path):
        plt.figure(figsize=(8, 10), tight_layout=True)

        plt.subplot(3, 1, 1)
        plt.title("Train Loss")
        plt.plot(self.train_loss)

        plt.subplot(3, 1, 2)
        plt.title("Test Loss")
        plt.plot(self.test_loss)
        
        plt.subplot(3, 1, 3)
        plt.title("Accuracy")
        plt.plot(self.accuracy)
        
        os.makedirs(path, exist_ok=True)
        plt.savefig(path + "figure.png")

    def save_stats(self, path):           
        fields = ['Train Loss', 'Test Loss', 'Accuracy'] 
        
        os.makedirs(path, exist_ok=True)
        with open(path + "stats.csv", 'w') as f:
            write = csv.writer(f)
            
            write.writerow(fields)
            for i in range(len(self.train_loss)):
                write.writerow([self.train_loss[i], self.test_loss[i], self.accuracy[i]])
        

