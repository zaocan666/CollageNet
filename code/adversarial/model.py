import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear

class Actor(nn.Module):
	def __init__(self, d_z_melody, d_z_accompany, d_input=2048, d_mid=2048, layer_num=4, dropout_rate=0, bn_flag=False, leaky_relu=False, c_input_dims=[]):
		super(Actor, self).__init__()
		self.d_z_melody = d_z_melody
		self.d_z_accompany = d_z_accompany

		self.melody_input = nn.Linear(d_z_melody, d_input)
		self.accompany_input = nn.Linear(d_z_accompany, d_input)
		
		if len(c_input_dims)==4:
			self.c_input0_layer = nn.Linear(1, c_input_dims[0])
			self.c_input1_layer = nn.Linear(1, c_input_dims[1])
			self.c_input2_layer = nn.Linear(1, c_input_dims[2])
			self.c_input3_layer = nn.Linear(1, c_input_dims[3])
		elif len(c_input_dims)!=0:
			raise KeyError('c input dim wrong')
		self.c_input_dims = c_input_dims

		relu_layer = (lambda:nn.LeakyReLU(0.2)) if leaky_relu else nn.ReLU
		layer_list = [relu_layer()]
		for i in range(layer_num):
			if i == 0:
				layer_list.append(nn.Linear(d_input*2+sum(c_input_dims), d_mid))
			else:
				layer_list.append(nn.Linear(d_mid, d_mid))
			if bn_flag:
				layer_list.append(nn.BatchNorm1d(d_mid))
			layer_list += [relu_layer(), nn.Dropout(p=dropout_rate)]
		
		layer_list.append(nn.Linear(d_mid, (d_z_melody+d_z_accompany)*2))

		self.fw_layer = nn.Sequential(*layer_list)
		self.gate = nn.Sigmoid()

	def forward(self, z_melody_original, z_accompany_original, c0, c1, c2, c3):
		z_melody = self.melody_input(z_melody_original)
		z_accompany = self.accompany_input(z_accompany_original)

		if len(self.c_input_dims)>0:
			c_input0 = self.c_input0_layer(c0)
			c_input1 = self.c_input1_layer(c1)
			c_input2 = self.c_input2_layer(c2)
			c_input3 = self.c_input3_layer(c3)
			x = torch.cat([z_melody, z_accompany, c_input0, c_input1, c_input2, c_input3], dim=-1) # [B, x]
		else:
			x = torch.cat([z_melody, z_accompany], dim = -1)

		out = self.fw_layer(x)

		out_melody = out[:, :self.d_z_melody*2]
		out_accompany = out[:, self.d_z_melody*2:]
		gate_melody , dz_melody = out_melody.chunk(2, dim = -1)
		gate_accompany , dz_accompany = out_accompany.chunk(2, dim = -1)

		gate_melody = self.gate(gate_melody)
		gate_accompany = self.gate(gate_accompany)
		new_z_melody = (1-gate_melody)*z_melody_original + gate_melody*dz_melody
		new_z_accompany = (1-gate_accompany)*z_accompany_original + gate_accompany*dz_accompany
		return new_z_melody, new_z_accompany

class Critic(nn.Module):
	def __init__(self, d_z_melody, d_z_accompany, d_input=2048, d_mid=2048, layer_num=4, dropout_rate=0, bn_flag=False, leaky_relu=False):
		super(Critic,self).__init__()
		self.d_z_melody = d_z_melody
		self.d_z_accompany = d_z_accompany
		
		relu_layer = (lambda:nn.LeakyReLU(0.2)) if leaky_relu else nn.ReLU
		self.melody_input = nn.Linear(d_z_melody, d_input)
		self.accompany_input = nn.Linear(d_z_accompany, d_input)
		layer_list = [relu_layer()]
		for i in range(layer_num):
			if i == 0:
				layer_list.append(nn.Linear(d_input*2, d_mid))
			else:
				layer_list.append(nn.Linear(d_mid, d_mid))
			if bn_flag:
				layer_list.append(nn.BatchNorm1d(d_mid))
			layer_list += [relu_layer(), nn.Dropout(p=dropout_rate)]
		
		layer_list.append(nn.Linear(d_mid, 1))
		self.fw_layer = nn.Sequential(*layer_list)

	def forward(self, z_melody_original, z_accompany_original):
		z_melody = self.melody_input(z_melody_original)
		z_accompany = self.accompany_input(z_accompany_original)

		x = torch.cat([z_melody, z_accompany], dim = -1)
		out = self.fw_layer(x)
		
		return torch.sigmoid(out)