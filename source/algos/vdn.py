import torch
import os
from network.base_net import RNN
from network.vdn_net import VDNNet
from network.arcomm_net import Arcomm, Decoder_reg
import torch.nn as nn
from network.commnet import CommNet

import numpy as np
import sys
import torch.nn.functional as F


#torch.set_printoptions(threshold=10_000)
#np.set_printoptions(threshold=10_000)


class VDN:
	def __init__(self, args):
		self.n_actions = args.n_actions
		self.n_agents = args.n_agents
		self.state_shape = args.state_shape
		self.obs_shape = args.obs_shape
		input_shape = self.obs_shape

		# changed 
		if args.arcomm:
			input_comm_shape = self.obs_shape

		print("obs shape: ", self.obs_shape)

		# input dimension for rnn according to the params
		if args.last_action:
		    input_shape += self.n_actions
		if args.reuse_network:
		    input_shape += self.n_agents

		if args.arcomm:
			input_shape += (args.n_agents - 1) * args.final_msg_dim
			print("obs shape with comm: ", input_shape)


		if args.alg == 'vdn':
			self.eval_rnn = RNN(input_shape, args) 
			self.target_rnn = RNN(input_shape, args)
			print('VDN alg initialized')
		elif args.alg == 'vdn+commnet' and not args.arcomm:  # for commnet
			self.eval_rnn = CommNet(input_shape, args)
			self.target_rnn = CommNet(input_shape, args)
			print("VDN+COMMNET initialised")


		self.eval_vdn_net = VDNNet()  
		self.target_vdn_net = VDNNet()  
		self.args = args

		if args.arcomm:
			self.arcomm = Arcomm(input_comm_shape, args)
			self.target_arcomm = Arcomm(input_comm_shape, args)
			# will receive obs+msgs
			self.decoder_reg = Decoder_reg(input_comm_shape + (self.n_agents - 1) * args.final_msg_dim, input_comm_shape, args)

		# cuda
		if self.args.cuda:
			self.eval_rnn.cuda(device=self.args.cuda_device)
			self.target_rnn.cuda(device=self.args.cuda_device)
			self.eval_vdn_net.cuda(device=self.args.cuda_device)
			self.target_vdn_net.cuda(device=self.args.cuda_device)
			if args.arcomm:
				self.arcomm.cuda(device=self.args.cuda_device)
				self.target_arcomm.cuda(device=self.args.cuda_device)
				self.decoder_reg.cuda(device=self.args.cuda_device)


		self.model_dir = args.model_dir + '/' + args.alg

		if self.args.load_model:
		    if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
		        path_rnn = self.model_dir + '/rnn_net_params.pkl'
		        path_vdn = self.model_dir + '/vdn_net_params.pkl'
		        self.eval_rnn.load_state_dict(torch.load(path_rnn))
		        self.eval_vdn_net.load_state_dict(torch.load(path_vdn))
		        print('Successfully load the model: {} and {}'.format(path_rnn, path_vdn))
		    else:
		    	raise Exception("No such model!")


		# make parameters of target and eval the same
		self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
		self.target_vdn_net.load_state_dict(self.eval_vdn_net.state_dict())

		if args.arcomm:
			self.target_arcomm.load_state_dict(self.arcomm.state_dict())
			self.eval_parameters = list(self.eval_vdn_net.parameters()) + list(self.eval_rnn.parameters()) + list(self.arcomm.parameters()) + list(self.decoder_reg.parameters())  # changed  # changed comm 
		else:
			self.eval_parameters = list(self.eval_vdn_net.parameters()) + list(self.eval_rnn.parameters())

		if args.optimizer == "RMS":
		    self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)


		# during learning one should keep an eval_hidden and a target_hidden for each agent of each episode
		self.eval_hidden = None
		self.target_hidden = None

		print('VDN alg initialized')


	def learn(self, batch, max_episode_len, train_step, epsilon=None):  

		'''
			batch: batch with episode batches from before training the model
			max_episode_len: len of the longest episode batch in batch
			train_step: it is used to control and update the params of the target network

			------------------------------------------------------------------------------

			the extracted data is 4D, with meanings 1-> n_episodes, 2-> n_transitions in the episode, 
			3-> data of multiple agents, 4-> obs dimensions
			hidden_state is related to the previous experience (RNN ?) so one cant randomly extract
			experience to learn, so multiple episodes are extracted at a time and then given to the
			nn one at a time   
		'''

		episode_num = batch['obs'].shape[0]  # gets number of episode batches in batch
		self.init_hidden(episode_num)

		#convert data in batch to tensor
		for key in batch.keys():  
		    if key == 'actions':
		        batch[key] = torch.tensor(batch[key], dtype=torch.long)
		    else:
		        batch[key] = torch.tensor(batch[key], dtype=torch.float32)

		obs, actions, reward, avail_actions, avail_actions_next, terminated = batch['obs'], batch['actions'], batch['reward'],  batch['avail_actions'], \
		                                          batch['avail_actions_next'], batch['terminated']


		# used to set the td error of the filled experiments to 0, not to affect learning
		mask = 1 - batch["padded"].float()  

		# shift one step to predict the next one in the decoder NOTE for now only 1 step
		obs_aux = obs[:, 1:, :, :]

		# cuda
		if self.args.cuda:
			obs = obs.cuda(device=self.args.cuda_device)
			actions = actions.cuda(device=self.args.cuda_device)
			reward = reward.cuda(device=self.args.cuda_device)
			mask = mask.cuda(device=self.args.cuda_device)
			terminated = terminated.cuda(device=self.args.cuda_device)
			obs_aux = obs_aux.cuda(device=self.args.cuda_device)


		# gets q value corresponding to each agent, dimensions are (episode_number, max_episode_len, n_agents, n_actions)
		q_evals, q_targets, all_msgs_list, all_msgs_next_list = self.get_q_values(batch, max_episode_len)

		# decodes messages and obs to reconstruct next step
		x_dec, _ = self.decoder_reg(obs[:, :-1, :, :], all_msgs_list[:, :-1, :, :])

		loss_dec = F.mse_loss(x_dec, obs_aux)

		q_evals = torch.gather(q_evals, dim=3, index=actions).squeeze(3)

		# get real q_target
		# unavailable actions dont matter, low value
		q_targets[avail_actions_next == 0.0] = - 9999999
		q_targets = q_targets.max(dim=3)[0]

		q_total_eval = self.eval_vdn_net(q_evals)
		q_total_target = self.target_vdn_net(q_targets)

		targets = reward + self.args.gamma * q_total_target * (1 - terminated)

		td_error = targets.detach() - q_total_eval
		masked_td_error = mask * td_error  

		# there are still useless experiments, so the avg is according the number of real experiments
		loss = (masked_td_error ** 2).sum() / mask.sum()

		# loss for dec v1
		loss += loss_dec

		self.optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
		self.optimizer.step()

		# update target networks
		if train_step > 0 and train_step % self.args.target_update_cycle == 0:
			self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
			self.target_vdn_net.load_state_dict(self.eval_vdn_net.state_dict())
			if self.args.arcomm:
				self.target_arcomm.load_state_dict(self.arcomm.state_dict())


	def get_q_values(self, batch, max_episode_len):
		episode_num = batch['obs'].shape[0]  # gets number of episode batches in batch
		q_evals, q_targets = [], []
		all_msgs_list, all_msgs_next_list = [], []
		for transition_idx in range(max_episode_len):
			
			inputs, inputs_next, all_msgs, all_msgs_next = self._get_inputs(batch, transition_idx)  

			if self.args.cuda:
				inputs = inputs.cuda(device=self.args.cuda_device)
				inputs_next = inputs_next.cuda(device=self.args.cuda_device)
				self.eval_hidden = self.eval_hidden.cuda(device=self.args.cuda_device)
				self.target_hidden = self.target_hidden.cuda(device=self.args.cuda_device)
				if self.args.arcomm:
					all_msgs = all_msgs.cuda(device=self.args.cuda_device)
					all_msgs_next = all_msgs_next.cuda(device=self.args.cuda_device)
					
			if self.args.arcomm:
				q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden, msgs=all_msgs)  
				q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden, msgs=all_msgs_next)
				
			else:
				q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)  
				q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)

			# Change the q_eval dimension back to (8, 5(n_agents), n_actions)
			q_eval = q_eval.view(episode_num, self.n_agents, -1)
			q_target = q_target.view(episode_num, self.n_agents, -1)
			q_evals.append(q_eval)
			q_targets.append(q_target)

			all_msgs_list.append(all_msgs)
			all_msgs_next_list.append(all_msgs_next)

		'''
		q_eval and q_target are lists containing max_episode_len arrays with dimensions (episode_number, n_agents, n_actions)
		convert the lists into arrays of (episode_number, max_episode_len, n_agents, n_actions)
		'''

		q_evals = torch.stack(q_evals, dim=1)
		q_targets = torch.stack(q_targets, dim=1)

		all_msgs_list = torch.stack(all_msgs_list, dim=1)
		all_msgs_next_list = torch.stack(all_msgs_next_list, dim=1)

		return q_evals, q_targets, all_msgs_list, all_msgs_next_list



	def _get_inputs(self, batch, transition_idx):
		obs, obs_next, actions_onehot = batch['obs'][:, transition_idx], \
		                          batch['obs_next'][:, transition_idx], batch['actions_onehot'][:]
		episode_num = obs.shape[0]
		inputs, inputs_next = [], []
		inputs.append(obs)
		inputs_next.append(obs_next)

		all_msgs, all_msgs_next = None, None

		if self.args.arcomm:

			inputs_msg = torch.cat([x for x in inputs], dim=-1)
			inputs_msg_next = torch.cat([x for x in inputs_next], dim=-1)
			if self.args.cuda:
				inputs_msg = inputs_msg.cuda(device=self.args.cuda_device)
				inputs_msg_next = inputs_msg_next.cuda(device=self.args.cuda_device)
				
			all_msgs = self.arcomm(inputs_msg)
			all_msgs_next = self.target_arcomm(inputs_msg_next)
			

		# adds last action and agent number to obs
		if self.args.last_action:
		    if transition_idx == 0:  # if it is the first transition, let the previous action be a 0 vector
		        inputs.append(torch.zeros_like(actions_onehot[:, transition_idx]))
		    else:
		        inputs.append(actions_onehot[:, transition_idx - 1])
		    inputs_next.append(actions_onehot[:, transition_idx])

		if self.args.reuse_network: 

			inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
			inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))


		inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
		inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)


		return inputs, inputs_next, all_msgs, all_msgs_next


	def init_hidden(self, episode_num):

		self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
		self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))



	def save_model(self, train_step, end_training=False):
		if end_training:
			torch.save(self.eval_vdn_net.state_dict(), self.model_dir + '/' + 'final_vdn_net_params.pkl')
			torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + 'final_rnn_net_params.pkl')
			if self.args.arcomm:
				torch.save(self.arcomm.state_dict(),  self.model_dir + '/' + 'final_arcomm_net_params.pkl')	
		else:
			num = str(train_step // self.args.save_cycle)
			if not os.path.exists(self.model_dir):
				os.makedirs(self.model_dir)
			torch.save(self.eval_vdn_net.state_dict(), self.model_dir + '/' + num + '_vdn_net_params.pkl')
			torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_net_params.pkl')
			if self.args.arcomm:
				torch.save(self.arcomm.state_dict(),  self.model_dir + '/' + num + '_arcomm_net_params.pkl')

