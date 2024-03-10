import torch
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(f"Using {device} device")

RANDOM_SEED = 42

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)
