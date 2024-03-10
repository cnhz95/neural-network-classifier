import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(f"Using {device} device")

random_seed = 42
batch_size = 16

data = pd.read_csv("data/TrainOnMe.csv")

# One-hot encode categorical data
data = pd.get_dummies(data, columns=["x7"], dtype=int)
data = pd.get_dummies(data, columns=["x12"], dtype=int)

X = data.values[:, 2:].astype(np.float32)
y = data.values[:, 1:2].reshape(data.shape[0])

# Encode labels
le = LabelEncoder().fit(y)
y = le.transform(y)

X = torch.from_numpy(X)
y = torch.from_numpy(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)


# Custom dataset
class ChallengeDataset(Dataset):
	def __init__(self, X, y):
		self.X = X
		self.y = y

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]


train_dataset = ChallengeDataset(X_train, y_train)
test_dataset = ChallengeDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
