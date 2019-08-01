import numpy as np
import torch
import gym
import argparse
from collections import deque
import os
import json
import utils
import TD3

def compute_reward(curr_pos, g, scale=0.5):
	return -1.0 * float(np.linalg.norm(curr_pos - g) > scale)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--env_name", default="AntMaze-v2")				# OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)					# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--max_timesteps", default=10e6, type=float)		# Max time steps to run environment for
	parser.add_argument("--start_timesteps", default=1e4, type=int)		# How many time steps purely random policy is run for
	parser.add_argument("--test_eps", default=10, type=int)				# Number of evaluation episodes to run
	parser.add_argument("--gpu", default=0, type=int)					# Which GPU to use; -1 for CPU

	# training parameters
	parser.add_argument("--expl_noise", default=0.1, type=float)		# Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=100, type=int)			# Batch size for actor and critic
	parser.add_argument("--discount", default=0.99, type=float)			# Discount factor for training
	parser.add_argument("--tau", default=0.005, type=float)				# Target newtork update rate
	parser.add_argument("--K", default=1, type=int)						# Number of ensemble members
	parser.add_argument("--rpf", action="store_true")					# Whether to use randomized prior functions

	# prioritized replay params
	parser.add_argument("--priority", action="store_true")				# Whether or not to use prioritization
	parser.add_argument("--alpha", default=0.6, type=float)
	parser.add_argument("--beta", default=0.4, type=float)
	parser.add_argument("--eps", default=1e-6, type=float)

	# TD3 specific parameters
	parser.add_argument("--policy_noise", default=0.2, type=float)		# Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5, type=float)		# Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)			# Frequency of delayed policy updates
	args = parser.parse_args()

	file_name = "HER_%s_%s" % (args.env_name, str(args.seed))
	print("---------------------------------------")
	print("Settings: " + file_name)
	print("---------------------------------------")

	if not os.path.exists("./her_models"):
		os.makedirs("./her_models")

	if torch.cuda.is_available():
		torch.cuda.set_device(args.gpu)
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	# Saving config parameters
	config_file_name = file_name + "_config.json"
	with open("./her_models/" + config_file_name, 'w') as f:
		config_dict = {k: v for (k, v) in vars(args).items()}
		json.dump(config_dict, f, indent=2)

	env = gym.make(args.env_name)
	eval_env = gym.make(args.env_name)

	# Set seeds
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	obs_space = env.observation_space

	state_dim = obs_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	# Initialize policy and buffer
	policy = TD3.TD3(state_dim, action_dim, max_action, device, K=args.K, rpf=args.rpf)
	if args.priority:
		replay_buffer = utils.PriorityReplayBuffer(timesteps=args.max_timesteps, alpha=args.alpha, beta=args.beta, eps=args.eps)
	else:
		replay_buffer = utils.ReplayBuffer()
	total_timesteps = 0
	total_episodes = 0
	episode_timesteps = 0
	done = True

	# Evaluation objects
	eval_rew_buffer = deque(maxlen=args.test_eps)
	eval_succ_buffer = deque(maxlen=args.test_eps)
	eval_obs = eval_env.reset()
	eval_done = False
	eval_return = 0.; eval_succ = 0.

	while total_timesteps < args.max_timesteps:
		if done: 
			if total_timesteps != 0:
				# Relabel experience with different goals
				g = episode_storage[-1][1][2:5]
				for item in episode_storage:
					x_h, x_tp1_h, a_h, d_h = item
					r_g = compute_reward(x_tp1_h[2:5], g)
					x_h[-3:] = g; x_tp1_h[-3:] = g
					replay_buffer.add((x_h, x_tp1_h, a_h, r_g, d_h))
				policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount,
					args.tau, args.policy_noise, args.noise_clip, args.policy_freq, args.priority, total_timesteps)

			# Evaluate and Save Policy
			if total_episodes % 5 == 0:
				eval_rew_mean = 0. if len(eval_rew_buffer) == 0 else np.mean(eval_rew_buffer)
				eval_succ_mean = 0. if len(eval_succ_buffer) == 0 else np.mean(eval_succ_buffer)
				print("Total T: %d Total Ep: %d Eval Avg. Rew: %f Eval Avg. Success: %f" % 
					(total_timesteps, total_episodes, eval_rew_mean, eval_succ_mean))
				policy.save(file_name, directory="./her_models")
			
			# Reset environment
			obs = env.reset()
			done = False
			total_episodes += 1
			episode_timesteps = 0
			episode_storage = []
		
		# Select action randomly or according to policy
		if total_timesteps < args.start_timesteps:
			action = env.action_space.sample()
			eval_action = policy.select_action(np.array(eval_obs)[None])[0]
		else:
			tot_action = policy.select_action(np.array([obs, eval_obs]))
			action, eval_action = tot_action[0], tot_action[1]
			if args.expl_noise != 0:
				action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0]))
				action = action.clip(env.action_space.low, env.action_space.high)

		# Perform action + Store Data in Buffer
		new_obs, reward, done, _ = env.step(action)
		eval_obs, eval_rew, eval_done, _ = eval_env.step(eval_action)
		eval_return += compute_reward(eval_env.unwrapped.get_current_pos(), eval_obs[-3:])
		eval_succ += compute_reward(eval_env.unwrapped.get_current_pos(), eval_obs[-3:]) + 1.0
		if eval_done:
			eval_rew_buffer.append(eval_return)
			eval_succ_buffer.append(np.sign(eval_succ))
			eval_obs = eval_env.reset()
			eval_return = 0.; eval_succ = 0.

		replay_buffer.add((obs, new_obs, action, compute_reward(env.unwrapped.get_current_pos(), obs[-3:]), float(done)))
		episode_storage.append((obs, new_obs, action, float(done)))
		obs = new_obs
		total_timesteps += 1
		episode_timesteps += 1

	# Save final policy
	policy.save("%s" % (file_name), directory="./her_models")
			