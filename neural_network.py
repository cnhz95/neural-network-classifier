import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(f"Using {device} device")

batch_size = 8
epochs = 30
nodes = 50

def read_data(csv_file):
	data = pd.read_csv(csv_file)
	data.drop(data.columns[0], inplace=True, axis=1)
	data = pd.get_dummies(data, columns=["x7", "x12"], dtype=int)  # One-hot encode categorical data
	return data

training_set = read_data("data/TrainOnMe.csv")
evaluation_set = read_data("data/EvaluateOnMe.csv")

# Normalize input
normalizer = StandardScaler()
X = normalizer.fit_transform(training_set.values[:, 1:].astype(np.float32))
evaluation = normalizer.fit_transform(evaluation_set.values[:, :].astype(np.float32))

# Encode labels
y = training_set.values[:, 0:1].reshape(training_set.shape[0])
le = LabelEncoder().fit(y)
y = le.transform(y)

class CustomDataset(Dataset):
	def __init__(self, X=None, y=None, evaluation=None):
		self.X = torch.from_numpy(X) if X is not None else None
		self.y = torch.from_numpy(y) if y is not None else None
		self.evaluation = torch.from_numpy(evaluation) if evaluation is not None else None

	def __len__(self):
		return len(self.X) if self.X is not None else len(self.evaluation)

	def __getitem__(self, idx):
		return [self.X[idx], self.y[idx]] if self.X is not None else self.evaluation[idx]

class NeuralNetwork(nn.Module):
	def __init__(self, n_features):
		super().__init__()
		self.dropout = nn.Dropout(0.2)
		self.layer_1 = nn.Linear(in_features=n_features, out_features=nodes)
		self.layer_2 = nn.Linear(in_features=nodes, out_features=nodes)
		self.layer_3 = nn.Linear(in_features=nodes, out_features=3)

	def forward(self, x):
		self.dropout(x)
		x = F.relu(self.layer_1(x))
		self.dropout(x)
		x = F.relu(self.layer_2(x))
		return x

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

training_data = CustomDataset(X_train, y_train)
test_data = CustomDataset(X_test, y_test)
evaluation_data = CustomDataset(None, None, evaluation)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
evaluation_dataloader = DataLoader(evaluation_data, batch_size=batch_size, shuffle=True)

model = NeuralNetwork(X.shape[1])

# Define the loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

classes = ("Allan", "Barbie", "Ken")

# Train the network
for epoch in range(epochs):
	model.train()
	for i, data in enumerate(train_dataloader, 0):
		inputs, labels = data
		optimizer.zero_grad()  # Zero the parameter gradients
		outputs = model(inputs)  # Passing the input data executes the `forward` method
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

# Test the network on the test data
correct = 0
total = 0
model.eval()
with torch.no_grad():  # No need to calculate the gradients for the output since we are not training
	for data in test_dataloader:
		inputs, labels = data
		outputs = model(inputs)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

accuracy = 100 * correct // total
print(f"\nAccuracy of the network: {accuracy} %", file=sys.stderr)

# Predict the labels of the evaluation data
if accuracy >= 79:
	predictions = []
	model.eval()
	with torch.no_grad():
		for data in evaluation_dataloader:
			inputs = data
			outputs = model(inputs)
			_, predicted = torch.max(outputs.data, 1)
			predictions.append("".join(f"{classes[predicted[p]]}\n" for p in range(len(predicted))))

	print("Saving predictions")
	open("predictions.txt", "w").writelines(f"{predictions[p]}" for p in range(len(predictions)))
