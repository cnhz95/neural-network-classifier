import sys
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

batch_size = 8
epochs = 50
nodes = 30

def read_data(csv_file):
	data = pd.read_csv(csv_file)
	data.drop(columns=["Unnamed: 0", "x12"], inplace=True)
	data = pd.get_dummies(data, columns=["x7"], dtype=int)  # One-hot encode categorical data
	return data.values

training_set = read_data("data/TrainOnMe.csv")
evaluation_set = read_data("data/EvaluateOnMe.csv")

# Normalize input
normalizer = StandardScaler()
X = normalizer.fit_transform(training_set[:, 1:].astype(np.float32))
evaluation = normalizer.fit_transform(evaluation_set.astype(np.float32))

# Encode labels
y = training_set[:, 0:1].reshape(training_set.shape[0])
le = LabelEncoder().fit(y)
y = le.transform(y)

classes = ("Allan", "Barbie", "Ken")

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

iterations = 30
accuracy_goal = 80
accuracies = np.empty((iterations, epochs))
for k in range(iterations):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

	training_data = CustomDataset(X_train, y_train)
	test_data = CustomDataset(X_test, y_test)
	evaluation_data = CustomDataset(None, None, evaluation)

	# Pass samples in mini-batches, reshuffle the training data at every epoch to reduce model overfitting.
	train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
	test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
	evaluation_dataloader = DataLoader(evaluation_data, batch_size=batch_size, shuffle=False)

	model = NeuralNetwork(X.shape[1]).to(device)

	# Define the loss and optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

	# Train the network
	for epoch in range(epochs):
		model.train()
		for i, data in enumerate(train_dataloader, 0):
			inputs, labels = data[0].to(device), data[1].to(device)
			optimizer.zero_grad()  # Zero the parameter gradients
			outputs = model(inputs)  # Passing the input data executes the `forward` method
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

		# Test the network on the test data
		total = 0
		correct = 0
		model.eval()
		with torch.no_grad():  # No need to calculate the gradients for the output since we are not training
			for data in test_dataloader:
				inputs, labels = data[0].to(device), data[1].to(device)
				outputs = model(inputs)
				val_loss = criterion(outputs, labels)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()

		accuracy = 100 * correct / total
		accuracies[k][epoch] = accuracy
		print(f"{k + 1}. Accuracy of the network: {accuracy} %")

		# Classify the evaluation data
		if accuracy > accuracy_goal:
			predictions = []
			with torch.no_grad():
				for data in evaluation_dataloader:
					inputs = data.to(device)
					outputs = model(inputs)
					_, predicted = torch.max(outputs.data, 1)
					predictions.append("".join(f"{classes[predicted[p]]}\n" for p in range(len(predicted))))

			print(f"Saving classification with accuracy rate of {accuracy} %", file=sys.stderr)
			open("predictions.txt", "w").writelines(f"{predictions[p]}" for p in range(len(predictions)))
			accuracy_goal = accuracy

print(f"Max accuracy: {accuracy_goal if accuracy_goal > 80 else 'Below target'} %")
print(f"Mean accuracy: {np.mean(accuracies)} %")
print(f"Stddev: {np.std(accuracies):.2f} %")
