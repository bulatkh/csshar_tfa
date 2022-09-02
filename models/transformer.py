import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from pytorch_lightning.core.lightning import LightningModule


class PositionalEncoding(nn.Module):
    """
    Implementation of positional encoding from https://github.com/pytorch/examples/tree/master/word_language_model
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
        

class ConvLayers(nn.Module):
    def __init__(self, in_channels, out_channels=[32, 64, 128], kernel_size=3, stride=1, sample_len=30, relu_type='relu'):
        super(ConvLayers, self).__init__()

        padding = int(kernel_size / 2) 
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

        self.out_size = self._compute_out_size(sample_len, padding, kernel_size, stride, 3, out_channels[-1])

    @staticmethod
    def _compute_out_size(sample_length, padding, kernel_size, stride, num_layers, num_channels):
        conv_out_size = sample_length
        for _ in range(num_layers):
            conv_out_size = int((conv_out_size + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
        return int(num_channels * conv_out_size)
        
    
    def forward(self, x):
        x = self.relu(self.batchNorm1(self.conv1(x)))
        x = self.relu(self.batchNorm2(self.conv2(x)))
        x = self.relu(self.batchNorm3(self.conv3(x)))
        return x.permute(2, 0, 1)


class TransformerEncoderLayerWeights(nn.TransformerEncoderLayer):
    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src2, attention_maps = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attention_maps


class TransformerEncoderWeights(nn.TransformerEncoder):
    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None):
        output = src
        attention_maps_list = []

        for mod in self.layers:
            output, attention_maps = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attention_maps_list.append(attention_maps)

        if self.norm is not None:
            output = self.norm(output)

        return output, torch.stack(attention_maps_list)


class Transformer(nn.Module):
    def __init__(self, in_channels, max_len, out_channels=[32, 64, 128], num_head=8, num_layers=6, kernel_size=3, dropout=0.1, return_attention=False, use_cls=False, **kwargs):
        super().__init__()
        self.name = 'transformer'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_head = num_head
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.use_cls = use_cls
        self.cnn = ConvLayers(in_channels, out_channels, kernel_size=kernel_size, sample_len=max_len)
        self.positional_encoding = PositionalEncoding(d_model=out_channels[-1], dropout=dropout, max_len=max_len + int(self.use_cls))
        
        if return_attention:
            self.return_attention = True
            self.encoder_layer = TransformerEncoderLayerWeights(d_model=out_channels[-1], nhead=num_head)
            self.transformer_encoder = TransformerEncoderWeights(self.encoder_layer, num_layers=num_layers)
        else:
            self.return_attention = False
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=out_channels[-1], nhead=num_head)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(p=dropout)  

        self.use_cls = use_cls
        if not self.use_cls:
            self.out_size  = self.cnn.out_size 
        else:
            self.out_size = out_channels[-1]
    
    def forward(self, x):
        x = self.cnn(x)
        if self.use_cls:
            x = self._append_cls_token(x)
        x = self.positional_encoding(x)

        if self.return_attention:
            x, attention_maps = self.transformer_encoder(x)
        else:
            x = self.transformer_encoder(x)

        x = self.dropout(x)

        if self.use_cls:
            x = x[0]
        else:
            x = x.permute(1, 0, 2)

        if self.return_attention:
            return x, attention_maps
        else:
            return x

    def _append_cls_token(self, x):
        cls_batch = nn.Parameter((torch.randn(1, x.shape[1], self.out_channels[-1]))).cuda()
        x = torch.cat((cls_batch, x), 0)
        return x