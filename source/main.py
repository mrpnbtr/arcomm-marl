
import gym
import ma_gym
from common.arguments import common_args, vdn_qmix_args, coma_args, commnet_args
from runner import Runner
from smac.env import StarCraft2Env

#import warnings
#warnings.filterwarnings("ignore", category=UserWarning) 

N_EXPERIMENTS = 1

if __name__ == '__main__':
	args = common_args()

	'''
	n_actions: number of actions in the environment for each agent
	n_agents: number of agents in the environment
	state_shape: a state is an array with all the values that describe the current state, i.e., all the
	features of the state
	obs_shape: an observation in a partially observable env is what each agent can see; an array with the
	values that describe what each agent can see
	episode_limit: maximum number of steps until which the episode will run if a terminal state wasnt reached
	before

	'''

	if args.env == 'PredatorPrey-v0':
		# avoid registering a new environment in the ma_gym package
		env = gym.make('PredatorPrey7x7-v0', grid_shape=(14, 14), n_agents=4, n_preys=2)
		args.n_actions = env.action_space[0].n
		args.n_agents = env.n_agents
		args.state_shape = 28 * args.n_agents 
		args.obs_shape = 28
		args.episode_limit = env._max_steps
	elif args.env == 'Lumberjacks-v0':
		env = gym.make(args.env, grid_shape=(8,8), n_agents=4)
		args.n_actions = env.action_space[0].n
		args.n_agents = env.n_agents
		args.state_shape = 22 * args.n_agents
		args.obs_shape = 22
		args.episode_limit = env._max_steps
	elif args.env in ["3m", "5m_vs_6m", "3s_vs_5z", "MMM2", "2c_vs_64zg"]:
		env = StarCraft2Env(map_name=args.env)
		env_info = env.get_env_info()
		args.n_actions = env_info["n_actions"]
		args.n_agents = env_info["n_agents"]
		args.state_shape = env_info["state_shape"]
		args.obs_shape = env_info["obs_shape"]
		args.episode_limit = env_info["episode_limit"]
	elif args.env in ["PredatorPrey"]:
		env = gym.make('PredatorPrey7x7-v0', grid_shape=(7, 7), n_agents=4, n_preys=2, penalty=-1.0)
		args.n_actions = env.action_space[0].n
		args.n_agents = env.n_agents
		args.state_shape = 28 * args.n_agents 
		args.obs_shape = 28
		args.episode_limit = env._max_steps
		print("Env PP with penalty ", env._penalty)
	else:
		raise Exception('Invalid environment: environment not supported!')
		

	print("Environment {} initialized, for {} time steps and evaluating every {} time steps".format(args.env, \
																							args.n_steps, args.evaluate_cycle))


	# load args
	if args.alg == 'vdn' or args.alg == 'qmix' or args.alg == 'qtran_base':
		args = vdn_qmix_args(args)
	elif args.alg.find('coma') > -1:
		args = coma_args(args)
	elif args.alg.find('vdn') > -1 or args.alg.find('qmix') > -1:
		args = vdn_qmix_args(args)
	else:
		raise Exception('No such algorithm!')

	if args.alg.find('commnet') > -1:
		args = commnet_args(args)

	print("CUDA set to", args.cuda)
	print("ARCOMM set to", args.arcomm)
	print("MSG CUT set to", args.msg_cut)

	runner = Runner(env, args)

	# parameterize run according to the number of independent experiments to run, i.e., independent sets of n_epochs over the model; default is 1
	if args.learn:
		runner.run(N_EXPERIMENTS)
