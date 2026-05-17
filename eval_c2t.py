# %%
from typing import Union

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader, TensorDataset


class Discriminator(nn.Module):

    def __init__(self, dim: int, h: int, w: int):
        super(Discriminator, self).__init__()

        self.h = h
        self.w = w

        self.h_in = 6
        self.w_in = 5
        self.dim = dim

        self.output_size = 4 * self.h_in * self.w_in * dim

        self.main = nn.Sequential(
            nn.Conv2d(1, dim, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim, 2 * dim, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(2 * dim, 4 * dim, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
        )

        self.output = nn.Linear(self.output_size, 1)

    def forward(self, input: Tensor) -> Tensor:
        n_samples = input.shape[0]
        input = input.view(n_samples, 1, self.h, self.w)

        out = self.main(input)
        out = out.view(n_samples, self.output_size)
        out = self.output(out)

        return out.view(-1)

    def predict_proba(self, input: Tensor) -> Tensor:
        out = self.forward(input)
        return torch.sigmoid(out)


class TwoSampleTest:

    def __init__(
        self,
        dim_discriminator: int = 128,
        test_size: float = 0.2,
        batch_size: int = 256,
        lr: float = 1e-4,
        n_epochs: int = 10,
        seed: int = 0,
        device: str = "cpu",
    ):

        self.dim_discriminator = dim_discriminator
        self.test_size = test_size
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.n_epochs = n_epochs
        self.device = device

        self.discriminator_: Discriminator
        self.accuracy_: float

    def compute_accuracy(self, real_data: Tensor, generated_data: Tensor):
        dtype = real_data.dtype
        device = self.device

        # annotate real/generated data
        labels_real = np.ones(real_data.shape[0])
        labels_generated = np.zeros(generated_data.shape[0])

        X = np.concatenate([real_data, generated_data], axis=0)
        y = np.concatenate([labels_real, labels_generated], axis=0)

        # prepare data for train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed, shuffle=True
        )

        X_train = torch.tensor(X_train, dtype=dtype).unsqueeze(1)
        y_train = torch.tensor(y_train, dtype=dtype)
        train_data = TensorDataset(X_train, y_train)

        X_test = torch.tensor(X_test, dtype=dtype).unsqueeze(1)
        y_test = torch.tensor(y_test, dtype=dtype)
        test_data = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True)

        # init model and optimizer
        H, W = X_train.shape[-2], X_train.shape[-1]
        discriminator_ = Discriminator(self.dim_discriminator, H, W).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(discriminator_.parameters(), lr=self.lr)

        # train discriminator
        n_epochs = self.n_epochs
        for epoch in range(n_epochs):
            discriminator_.train()
            total_loss = 0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                outputs = discriminator_(X_batch).view(-1)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(
                f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(train_loader):.4f}"
            )

        self.discriminator_ = discriminator_

        # evaluation
        discriminator_.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                probas = discriminator_.predict_proba(X_batch)
                predicted = (probas > 0.5).float()

                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

        accuracy = correct / total
        self.accuracy_ = accuracy
        return accuracy
