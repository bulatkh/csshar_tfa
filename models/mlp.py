import torch
import torch.nn as nn

class MLP(nn.Module):
	def __init__(self, in_size, out_size, hidden=[256, 128]):
		super().__init__()
		self.name = 'MLP'
		self.relu = nn.ReLU()
		self.linear1 = nn.Sequential(
			nn.Linear(in_size, hidden[0]),
			nn.BatchNorm1d(hidden[0]),
			nn.ReLU(inplace=True)
		)
		self.linear2 = nn.Sequential(
			nn.Linear(hidden[0], hidden[1]),
			nn.BatchNorm1d(hidden[1]),
			nn.ReLU(inplace=True)
		)
		self.output = nn.Linear(hidden[1], out_size)

	def forward(self, x):
		x = self.linear1(x)
		x = self.linear2(x)
		x = self.output(x)
		return x


class LinearClassifier(nn.Module):
	def __init__(self, in_size, out_size):
		super().__init__()
		self.name = 'LinearClassifier'
		self.classifier = nn.Linear(in_size, out_size)

	def forward(self, x):
		x = self.classifier(x)
		return x


class ProjectionMLP(nn.Module):
	def __init__(self, in_size, fc_size, out_size):
		super().__init__()
		self.out_size = out_size
		
		self.layer1 = nn.Sequential(
			nn.Linear(in_size, fc_size),
			nn.ReLU(inplace=True)
		)
		self.layer2 = nn.Linear(fc_size, out_size)

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		return x


class MLPDropout(nn.Module):
	def __init__(self, in_size, out_size, hidden=[256, 128]):
		super(MLPDropout, self).__init__()
		self.name = 'MLP'
		self.relu = nn.ReLU()
		self.linear1 = nn.Sequential(
			nn.Linear(in_size, hidden[0]),
			nn.ReLU(inplace=True),
			nn.Dropout(0.2)
		)
		self.linear2 = nn.Sequential(
			nn.Linear(hidden[0], hidden[1]),
			nn.ReLU(inplace=True),
			nn.Dropout(0.2)
		)
		self.output = nn.Linear(hidden[1], out_size)

	def forward(self, x):
		x = self.linear1(x)
		x = self.linear2(x)
		x = self.output(x)
		return x 

