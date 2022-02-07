import torch
import torch.nn as nn

from models.transformer import ConvLayers, PositionalEncoding, TransformerEncoderLayerWeights, TransformerEncoderWeights

class Encoder(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pooling_kernel, pooling_padding):
		super(Encoder, self).__init__()
		self.cnn_block1 = nn.Sequential(
			nn.Conv1d(in_channels=in_channels, out_channels=out_channels[0], kernel_size=kernel_size, stride=stride, padding=padding),
			# nn.BatchNorm1d(out_channels[0]),
			nn.ReLU(inplace=True),
			nn.MaxPool1d(pooling_kernel)
		)
		self.cnn_block2 = nn.Sequential(
			nn.Conv1d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=kernel_size, stride=stride, padding=padding),
			# nn.BatchNorm1d(out_channels[1]),
			nn.ReLU(inplace=True),
			nn.MaxPool1d(pooling_kernel)
		)
		self.cnn_block3 = nn.Sequential(
			nn.Conv1d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=kernel_size, stride=stride, padding=padding),
			# nn.BatchNorm1d(out_channels[2]),
			nn.ReLU(inplace=True),
			nn.MaxPool1d(pooling_kernel)
		)
		self.flatten = nn.Flatten()

	def forward(self, x):
		x = self.cnn_block1(x)
		x = self.cnn_block2(x)
		x = self.cnn_block3(x)
		return self.flatten(x)


class Decoder(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, upsample=True):
		super(Decoder, self).__init__()
		padding = int(kernel_size / 2)
		self.out_channels = out_channels
		if upsample:
			self.decnn_block1 = nn.Sequential(
				nn.ConvTranspose1d(in_channels=out_channels[2], out_channels=out_channels[1], kernel_size=kernel_size, stride=stride, padding=padding),
				nn.ReLU(inplace=True),
				nn.Upsample(7)
			)
			self.decnn_block2 = nn.Sequential(
				nn.ConvTranspose1d(in_channels=out_channels[1], out_channels=out_channels[0], kernel_size=kernel_size, stride=stride, padding=padding),
				nn.ReLU(inplace=True),
				nn.Upsample(15)
			)
			self.decnn_block3 = nn.Sequential(
				nn.ConvTranspose1d(in_channels=out_channels[0], out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding),
				nn.ReLU(inplace=True),
				nn.Upsample(30)
			)
		else:
			self.decnn_block1 = nn.Sequential(
				nn.ConvTranspose1d(in_channels=out_channels[2], out_channels=out_channels[1], kernel_size=kernel_size, stride=stride, padding=padding),
				nn.ReLU(inplace=True)
			)
			self.decnn_block2 = nn.Sequential(
				nn.ConvTranspose1d(in_channels=out_channels[1], out_channels=out_channels[0], kernel_size=kernel_size, stride=stride, padding=padding),
				nn.ReLU(inplace=True)
			)
			self.decnn_block3 = nn.Sequential(
				nn.ConvTranspose1d(in_channels=out_channels[0], out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding),
				nn.ReLU(inplace=True)
			)
	
	def forward(self, x):
		x = x.view(x.shape[0], self.out_channels[-1], -1)
		x = self.decnn_block1(x)
		x = self.decnn_block2(x)
		x = self.decnn_block3(x)
		return x


class Bottleneck(nn.Module):
	def __init__(self, conv_out_size, latent_size):
		super(Bottleneck, self).__init__()
		self.linear1 = nn.Linear(conv_out_size, latent_size)
		self.linear2 = nn.Linear(latent_size, conv_out_size)

	def forward(self, x, return_features=False):
		x = self.linear1(x)
		if not return_features:
			x = self.linear2(x)
		return x


class Autoencoder(nn.Module):
	def __init__(self, in_channels, out_channels, latent_size, kernel_size=3, stride=1, padding=1, pooling_kernel=2, pooling_padding=0, len_seq=30, supervised=False, return_attention=False):
		super(Autoencoder, self).__init__()
		self.name = 'cae'
		self.supervised = supervised
		self.num_layers = len(out_channels)
		self.encoder = Encoder(in_channels, out_channels, kernel_size, stride, padding, pooling_kernel, pooling_padding)
		conv_out_size = len_seq
		for _ in range(self.num_layers):
			conv_out_size = int((conv_out_size + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
			conv_out_size = int((conv_out_size + 2 * pooling_padding - (pooling_kernel - 1) - 1) / pooling_kernel + 1)
		conv_out_size = int(out_channels[-1] * conv_out_size)

		self.bottleneck = Bottleneck(conv_out_size, latent_size)

		self.decoder = Decoder(in_channels, out_channels, kernel_size, stride, padding, upsample=True)

	def forward(self, x, return_features=False):
		x = self.encoder(x)
		x = self.bottleneck(x, return_features)
		if not return_features:
			x = self.decoder(x)
		return x

class TransformerEncoder(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, num_head=8, num_layers=6, max_len=30, dropout=0.1, return_attention=False):
		super(TransformerEncoder, self).__init__()
		self.cnn = ConvLayers(in_channels, out_channels, kernel_size=kernel_size)
		self.positional_encoding = PositionalEncoding(d_model=out_channels[-1], dropout=dropout, max_len=max_len)
		self.return_attention = return_attention
		if return_attention:
			self.encoder_layer = TransformerEncoderLayerWeights(d_model=out_channels[-1], nhead=num_head)
			self.transformer_encoder = TransformerEncoderWeights(self.encoder_layer, num_layers=num_layers)
		else:
			self.encoder_layer = nn.TransformerEncoderLayer(d_model=out_channels[-1], nhead=num_head)
			self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
		self.flatten = nn.Flatten()

	def forward(self, x):
		x = self.cnn(x)
		x = self.positional_encoding(x)
		if self.return_attention:
			x, attention_maps = self.transformer_encoder(x)
		else:
			x = self.transformer_encoder(x)
		x = x.permute(1, 2, 0)
		if self.return_attention:
			return self.flatten(x), attention_maps
		else:
			return self.flatten(x)

		

class TransformerAutoencoder(nn.Module):
	def __init__(self, in_channels, len_seq, out_channels, num_head, num_layers, latent_size, dropout, kernel_size=3, supervised=False, return_attention=False):
		super(TransformerAutoencoder, self).__init__()
		self.name = 'transformer_cae'
		self.supervised = supervised
		self.num_layers = len(out_channels)
		self.return_attention = return_attention
		self.encoder = TransformerEncoder(in_channels, out_channels, kernel_size, num_head, num_layers, len_seq, dropout, return_attention=return_attention)
		conv_out_size = int(out_channels[-1] * len_seq)

		self.bottleneck = Bottleneck(conv_out_size, latent_size)

		self.decoder = Decoder(in_channels, out_channels, kernel_size, upsample=False)


	def forward(self, x, return_features=False):
		if self.return_attention:
			x, attention_maps = self.encoder(x)
		else:
			x = self.encoder(x)
		x = self.bottleneck(x, return_features)
		if not return_features:
			x = self.decoder(x)
		if self.return_attention:
			return x, attention_maps
		else:
			return x
