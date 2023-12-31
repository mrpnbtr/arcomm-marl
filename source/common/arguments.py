from argparse import ArgumentParser

def common_args():
	parser = ArgumentParser()
	# smac args
	parser.add_argument('--difficulty', type=str, default='7', help='the difficulty of the game')
	parser.add_argument('--game_version', type=str, default='latest', help='the version of the game')
	parser.add_argument('--map', type=str, default='3m', help='the map of the game')
	parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
	parser.add_argument('--replay_dir', type=str, default='', help='absolute path to save the replay')

	# general args
	parser.add_argument("--env", "-e", default="8m", help="set env name")
	parser.add_argument("--n_steps", "-ns", type=int, default=2000000, help="set total time steps to run")
	parser.add_argument("--n_episodes", "-nep", type=int, default=1, help="set n_episodes")
	parser.add_argument("--epsilon", "-eps", default=0.5, help="set epsilon value")
	parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
	parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
	parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor')
	parser.add_argument('--evaluate_epoch', type=int, default=20, help='the number of the epoch to evaluate the agent')
	parser.add_argument('--alg', type=str, default='vdn', help='the algorithm to train the agent')
	parser.add_argument('--optimizer', type=str, default="RMS", help='the optimizer')
	parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
	parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
	parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
	parser.add_argument('--learn', type=bool, default=True, help='whether to train the model')
	parser.add_argument('--evaluate_cycle', type=int, default=10000, help='how often to eval the model')
	parser.add_argument('--target_update_cycle', type=int, default=200, help='how often to update the target network')
	parser.add_argument('--save_cycle', type=int, default=6650, help='how often to save the model')
	parser.add_argument('--cuda', type=bool, default=False, help='whether to use the GPU')
	# if using comm
	parser.add_argument('--arcomm', type=bool, default=False, help='whether to use arcomm')
	parser.add_argument('--msg_cut', type=bool, default=False, help='whether to cut msg')
	parser.add_argument('--cuda_device', type=int, default=0, help='which gpu number')



	args = parser.parse_args()

	return args



def coma_args(args):
	# network
	args.rnn_hidden_dim = 64
	args.critic_dim = 128
	args.lr_actor = 1e-4
	args.lr_critic = 1e-3

	# epsilon greedy
	args.epsilon = 0.5
	args.min_epsilon = 0.02
	args.anneal_epsilon = 0.00064
	args.epsilon_anneal_scale = 'epoch'

	# almbda for td-lambda return
	args.td_lambda = 0.8

	# prevent gradient explosion
	args.grad_norm_clip = 10

	return args


def vdn_qmix_args(args):
	# buffer/batch sizes
	args.batch_size = 32
	args.buffer_size = int(5e3)
	
	args.epsilon = 1
	args.min_epsilon = 0.05
	anneal_steps = 50000
	args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
	args.epsilon_anneal_scale = 'step'

	# netowrk args for vdn
	# network
	args.rnn_hidden_dim = 64
	args.qmix_hidden_dim = 32
	args.two_hyper_layers = False
	args.hyper_hidden_dim = 64
	args.qtran_hidden_dim = 64
	args.lr = 5e-4

	# train steps for vdn
	args.train_steps = 1

	# prevent gradient explosion
	args.grad_norm_clip = 10

	# QTRAN lambda
	args.lambda_opt = 1
	args.lambda_nopt = 1

	# CHANGED msg dim after net
	args.final_msg_dim = 10
	

	return args


def commnet_args(args):
	
	args.k = 3

	return args