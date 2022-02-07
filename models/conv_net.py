import torch.nn as nn


class CNN1D(nn.Module):
	def __init__(self, 
				in_channels, 
				len_seq, 
				out_size,
				out_channels=[32, 64, 128], 
				fc_size=256, 
				kernel_size=3, 
				stride=1, 
				padding=1,
				pool_padding=0, 
				pool_size=2, 
				dropout_rate=0.1,
				supervised=True):
		"""
		1D-Convolutional Network
		"""
		super(CNN1D, self).__init__()
		self.name = 'cnn1d'
		self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=None)
		self.num_layers = len(out_channels)
		self.dropout = nn.Dropout(p=dropout_rate, inplace=True)
		self.relu = nn.ReLU()

		self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels[0], kernel_size=kernel_size, stride=stride, padding=padding)
		self.batchNorm1 = nn.BatchNorm1d(out_channels[0])
		self.conv2 = nn.Conv1d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=kernel_size, stride=stride, padding=padding)
		self.batchNorm2 = nn.BatchNorm1d(out_channels[1])
		self.conv3 = nn.Conv1d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=kernel_size, stride=stride, padding=padding)
		self.batchNorm3 = nn.BatchNorm1d(out_channels[2])
		
		self.flatten = nn.Flatten()
		conv_out_size = len_seq
		for _ in range(self.num_layers):
			conv_out_size = int((conv_out_size + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
			conv_out_size = int((conv_out_size + 2 * pool_padding - (pool_size - 1) - 1) / pool_size + 1)

		self.fc1 = nn.Linear(int(out_channels[self.num_layers - 1] * conv_out_size), fc_size)
		self.fc2 = nn.Linear(fc_size, int(fc_size / 2))
		self.supervised = supervised
		if supervised:
			self.fc3 = nn.Linear(int(fc_size / 2), out_size)

	def forward(self, x, skip_last_fc=False):
		x = self.relu(self.batchNorm1(self.conv1(x)))
		x = self.pool(x)
		x = self.relu(self.batchNorm2(self.conv2(x)))
		x = self.pool(x)
		x = self.relu(self.batchNorm3(self.conv3(x)))
		x = self.pool(x)

		x = self.flatten(x)
		x = self.fc1(x)
		x = self.dropout(x)
		x = self.fc2(x)
		if self.supervised:
			x = self.fc3(x)
		return x