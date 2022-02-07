import numpy as np
from torch import nn

import torch
import torch.nn.functional as F

from .mlp import ProjectionMLP_SimCLR, SimSiamMLP

class AttnLSTM(nn.Module):
	def __init__(self, 
				input_dim, 
				hidden_dim, 
				output_dim, 
				n_layers=1,
				sensor_attention=False, 
				temporal_attention=False, 
				return_weights=False,  
				norm_out=False, 
				get_lstm_features=False, 
				initialize_lstm=False,
				fc_size=256,
				supervised=True,
				framework = 'SimCLR'):
		super(AttnLSTM, self).__init__()
		self.name = 'attention_lstm'
		self.framework = framework
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.return_weights = return_weights
		self.sensor_attention = sensor_attention
		self.temporal_attention = temporal_attention
		self.n_layers = n_layers
		self.supervised = supervised
		
		if sensor_attention:
			self.sens_attn_layer = SensorAttention(self.input_dim)
		self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers)
		if temporal_attention:
			self.temp_attn_layer = TemporalAttention(self.hidden_dim)
		if self.supervised:
			self.linear = nn.Linear(self.hidden_dim, self.output_dim)
		else:
			if self.framework == 'SimCLR':
				self.projection = ProjectionMLP_SimCLR(self.hidden_dim, fc_size)
			elif self.framework == 'SimSiam':
				self.projection = ProjectionMLP_SimSiam(self.hidden_dim, fc_size)
				self.prediction = PredictionMLP(fc_size, int(fc_size / 2))

		

		self.norm_out = norm_out
		self.get_lstm_features = get_lstm_features
		self.initialize_lstm = initialize_lstm
		
	def forward(self, x, projection_head=True, prediction=False, hidden=None):
		if self.sensor_attention:
			x, sensor_weights = self.sens_attn_layer(x)
		else:
			sensor_weights = None

		if self.initialize_lstm:
			hidden_states, _ = self.lstm(x, hidden)
		else:
			hidden_states, _ = self.lstm(x)

		if self.temporal_attention:
			hidden_states, temporal_weights = self.temp_attn_layer(hidden_states)
		else:
			hidden_states = hidden_states[-1]
			temporal_weights = None
		if self.supervised:
			x = self.linear(hidden_states)
		elif projection_head:
			x = self.projection(hidden_states)
			if prediction:
				x = self.prediction(x)
		else:
			x = hidden_states

		if self.norm_out:
			x = F.normalize(x, p=2, dim=1)
		if self.return_weights:
			return x, sensor_weights, temporal_weights
		return x
		
class TemporalAttention(nn.Module):
	def __init__(self, input_dim):
		super(TemporalAttention, self).__init__()
		self.linear_s = nn.Linear(input_dim, input_dim)
	
	def forward(self, hidden_states):
		scores = []
		for i, hidden in enumerate(hidden_states):
			int_vector = self.linear_s(hidden)
			tmp_score = torch.bmm(hidden_states[-1].unsqueeze(1), int_vector.unsqueeze(2))
			scores.append(tmp_score.squeeze())
		scores = torch.stack(scores)
		weights = F.softmax(scores, dim=0).unsqueeze(-1)
		hidden_out = torch.sum(torch.mul(weights, hidden_states), dim=0)
		return hidden_out, weights.squeeze().permute(1, 0)


class SensorAttention(nn.Module):
	def __init__(self, input_dim):
		super(SensorAttention, self).__init__()
		self.input_dim = input_dim
		self.linear_x = nn.Linear(input_dim, input_dim)
		self.linear_b = nn.Linear(input_dim, input_dim)
		self.linear_e = nn.Linear(input_dim, input_dim)
	
	def forward(self, sensor_data):
		new_signals_arr = []
		beta_arr = []
		batch_size = sensor_data.shape[1]
		tmp_beta = torch.zeros(batch_size, self.input_dim).float().cuda()
		for tmp_signal_batch in sensor_data:
			signal_out = self.linear_x(tmp_signal_batch)
			beta_out = self.linear_b(tmp_beta)
			merged = torch.tanh(signal_out + beta_out)
			
			energy = self.linear_e(merged)
			tmp_beta = F.softmax(energy, dim=1)
			new_signal = torch.mul(tmp_beta, tmp_signal_batch)
			
			beta_arr.append(tmp_beta)
			new_signals_arr.append(new_signal)
		new_signals = torch.stack(new_signals_arr).cuda()
		sens_weights = torch.stack(beta_arr).cuda()
		return new_signals, sens_weights.permute(1, 0, 2)