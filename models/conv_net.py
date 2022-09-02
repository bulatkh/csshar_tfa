import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(self, 
                in_channels, 
                len_seq=30, 
                out_channels=[32, 64, 128], 
                fc_size=256, 
                kernel_size=3, 
                stride=1, 
                padding=1,
                pool_padding=0, 
                pool_size=2, 
                supervised=True,
				relu_type = 'relu',
                **kwargs):
        """
        1D-Convolutional Network
        """
        super(CNN1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size


        self.name = 'cnn1d'
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=None)
        self.num_layers = len(out_channels)

        if relu_type == 'leaky':
            self.relu = nn.LeakyReLU()
        else:
            self.relu = nn.ReLU()

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels[0], kernel_size=kernel_size, stride=stride, padding=padding)
        self.batchNorm1 = nn.BatchNorm1d(out_channels[0])
        self.conv2 = nn.Conv1d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=kernel_size, stride=stride, padding=padding)
        self.batchNorm2 = nn.BatchNorm1d(out_channels[1])
        self.conv3 = nn.Conv1d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=kernel_size, stride=stride, padding=padding)
        self.batchNorm3 = nn.BatchNorm1d(out_channels[2])

        self.out_size = self._compute_out_size(len_seq, padding, kernel_size, stride, 3, out_channels[-1], pool_size, pool_padding)

    @staticmethod
    def _compute_out_size(sample_length, padding, kernel_size, stride, num_layers, num_channels, pool_size, pool_padding):
        conv_out_size = sample_length
        for _ in range(num_layers):
            conv_out_size = int((conv_out_size + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
            # conv_out_size = int((conv_out_size + 2 * pool_padding - (pool_size - 1) - 1) / pool_size + 1)
        return int(num_channels * conv_out_size)

    def forward(self, x, skip_last_fc=False):
        x = self.relu(self.batchNorm1(self.conv1(x)))
        # x = self.pool(x)
        x = self.relu(self.batchNorm2(self.conv2(x)))
        # x = self.pool(x)
        x = self.relu(self.batchNorm3(self.conv3(x)))
        # x = self.pool(x)
        return x