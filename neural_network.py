import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(f"Using {device} device")

#random_seed = 42
batch_size = 32
epochs = 50

data = pd.read_csv("data/TrainOnMe.csv")

# One-hot encode categorical data
data = pd.get_dummies(data, columns=["x7"], dtype=int)
data = pd.get_dummies(data, columns=["x12"], dtype=int)

normalizer = StandardScaler()
X = normalizer.fit_transform(data.values[:, 2:].astype(np.float32))
y = data.values[:, 1:2].reshape(data.shape[0])

# Encode labels
le = LabelEncoder().fit(y)
y = le.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)


class CustomDataset(Dataset):
	def __init__(self, X, y):
		self.X = torch.from_numpy(X)
		self.y = torch.from_numpy(y)

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]


train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


class NeuralNetwork(nn.Module):
	def __init__(self, n_features):
		super().__init__()
		self.layer_1 = nn.Linear(in_features=n_features, out_features=n_features)
		self.layer_2 = nn.Linear(in_features=n_features, out_features=n_features)
		self.layer_3 = nn.Linear(in_features=n_features, out_features=n_features)
		self.layer_4 = nn.Linear(in_features=n_features, out_features=n_features)
		self.layer_5 = nn.Linear(in_features=n_features, out_features=3)

	def forward(self, x):
		x = F.relu(self.layer_1(x))
		x = F.relu(self.layer_2(x))
		x = F.relu(self.layer_3(x))
		x = F.relu(self.layer_4(x))
		return x


model = NeuralNetwork(X.shape[1])

# Define the loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# Train the network
for epoch in range(epochs):
	running_loss = 0.0
	for i, data in enumerate(train_dataloader, 0):
		inputs, labels = data
		optimizer.zero_grad()  # Zero the parameter gradients
		outputs = model(inputs)  # Passing the input data executes the `forward` method
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()

# Test the network on the test data
correct = 0
total = 0
with torch.no_grad():  # No need to calculate the gradients for the output since we are not training
	for data in test_dataloader:
		images, labels = data
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

print(f"Accuracy of the network: {100 * correct // total} %")
