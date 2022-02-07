from torch import nn

class VanillaLSTM(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1, norm_out=False, get_lstm_features=False, initialize_lstm=False):
		super(VanillaLSTM, self).__init__()
		self.name = 'vanilla_lstm'
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.n_layers = n_layers
		
		self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers)
		self.linear = nn.Linear(self.hidden_dim, output_dim)
		self.norm_out = norm_out
		self.get_lstm_features = get_lstm_features
		self.initialize_lstm = initialize_lstm
		
	def forward(self, x, hidden=None):
		if self.initialize_lstm:
			lstm_out, _ = self.lstm(x, hidden)
		else:
			lstm_out, _ = self.lstm(x)
		out = self.linear(lstm_out[-1])
		if self.norm_out:
			norm = out.norm(p=2, dim=1, keepdim=True)
			out = out.div(norm)
		if self.get_lstm_features:
			return out, lstm_out[-1]
		else:
			return out