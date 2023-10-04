import torch
import torch.nn as nn
import torch.nn.functional as F
#import sparse
#from numcompress import compress, decompress
import sys
import numpy as np
from torch_dct import dct, idct

def get_tensor_size(t: torch.tensor):
    return t.nelement() * t.element_size()



class SimpleAttentionModule(nn.Module):
    def __init__(self, input_dim, emb_dim, msg_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.input_dim = input_dim

        self.w_keys = nn.Linear(self.input_dim, self.emb_dim)
        self.w_queries = nn.Linear(msg_dim, self.emb_dim)
        self.w_values = nn.Linear(self.input_dim, self.emb_dim)


        self.fc_f = nn.Linear(self.emb_dim, msg_dim)


    def forward(self, x, m):
        # x: [n_ep, n_agents, input_size+msg_dim]
        # m: [n_ep, n_agents, msg_dim]

        keys = self.w_keys(x)
        queries = self.w_queries(m)
        values = self.w_values(x)

        # scale first and for **0.25 instead (save memory)
        queries = queries / (self.emb_dim ** (1/4))
        keys = keys / (self.emb_dim ** (1/4))

        # dot product q*k
        dot = torch.matmul(queries, keys.transpose(-1, -2))

        # softmax the weights
        dot = F.softmax(dot, dim=-1)

        # apply self attention to the residual values
        out = torch.matmul(dot, values)

        out = self.fc_f(out)

        return out


# input obs of all agents，output encoded message for each one of the agents
class Arcomm(nn.Module):
    def __init__(self, input_shape, args):
        super(Commtest_2, self).__init__()
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.final_msg_dim)

        self.args = args
        self.input_shape = input_shape

        self.att = SimpleAttentionModule(input_shape, args.rnn_hidden_dim, args.final_msg_dim)  # SEE CHANGE

        
    def forward(self, inputs):
        # inputs for now are the observations of all agents that will be here used to generate the messages and then 
        # the messages will be cut using lossy dct if msg_cut is true
        # [b, e, n_a, o_dim] -> [b, e, n_a, m_dim]

        ep_num = inputs.shape[0] // self.args.n_agents

        # simple fc net
        x1 = F.relu(self.fc1(inputs))
        x2 = F.relu(self.fc2(x1))
        x3 = self.fc3(x2)

        # pass through the att module
        att_v = self.att(inputs, x3)

        # apply dct to encode messages
        m = dct(att_v)

        # create mask to cut the message 
        if self.args.msg_cut:
            compressed_size = int(self.args.final_msg_dim * (1. - self.args.compression_rate))
            final_msg = m[..., :compressed_size].reshape(-1, self.args.n_agents, compressed_size) 
        else:
            final_msg = m.reshape(-1, self.args.n_agents, self.args.final_msg_dim) 

            
        return final_msg



class Decoder_reg(nn.Module):
    def __init__(self, input_shape_in, input_shape_out, args):
        super(Decoder_5, self).__init__()

        self.args = args

        self.rnn_1 = nn.LSTM(input_size=input_shape_in * self.args.n_agents,
                           hidden_size=self.args.rnn_hidden_dim,
                           batch_first=True)


        self.fc_1 = nn.Linear(self.args.rnn_hidden_dim, input_shape_out * self.args.n_agents)
        

    def forward(self, obs_in, msg, initial_state=None):

        ep_num = obs_in.shape[0]

        # concatenate only messages coming from the other agents, i.e., m-i
        # during training everything comes together (bs >= 1), so need another way to cat the respective messages to the right indices
        # i.e., all agents should only receive the messages from the others and not themselves (m_-i)
        a_mask = torch.eye(self.args.n_agents).reshape(self.args.n_agents, self.args.n_agents, 1)
        a_mask = torch.abs(a_mask - 1)
        if self.args.cuda:
            a_mask = a_mask.cuda(device=self.args.cuda_device)
        
        # msg_rec: [bs, n_agents, msg_dim]
        msg_rep = msg.repeat(1, 1, self.args.n_agents, 1).reshape(ep_num, obs_in.shape[1], self.args.n_agents, self.args.n_agents, -1)
        msgs_repective_idxs = msg_rep * a_mask
        # [bs, n_a, (n_a - 1)*msg_dim] - m_-i: each agent receives the messages from the others
        msgs_repective_idxs_no_0 = msgs_repective_idxs[msgs_repective_idxs.count_nonzero(dim=-1) != 0].reshape(ep_num, obs_in.shape[1], self.args.n_agents, -1)
        msg = msgs_repective_idxs_no_0

        # now concat with msg and change to previous shape: [bs, n_a, input_dim+msg_dim] -> [bs*n_a, input_dim+msg_dim]
        decoder_input_1 = torch.cat((obs_in, msg), dim=-1)        
        
        # [b, t, a, obs+m] -> [b, t, a*(obs+m)]
        decoder_input_1 = decoder_input_1.reshape(obs_in.shape[0], obs_in.shape[1], -1)

        out_1, final_state_1 = self.rnn_1(decoder_input_1, initial_state)

        result = self.fc_1(out_1)
        result = result.reshape(ep_num, obs_in.shape[1], self.args.n_agents, -1)

        return result, final_state_1


