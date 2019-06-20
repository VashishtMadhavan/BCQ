import numpy as np
import torch
import gym
import argparse
import os

import utils
import HER

# Shortened version of code originally found at https://github.com/sfujim/TD3

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--env_name", default="FetchReach-v1")				# OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)					# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--max_timesteps", default=1e6, type=float)		# Max time steps to run environment for
	parser.add_argument("--start_timesteps", default=1e3, type=int)		# How many time steps purely random policy is run for
	parser.add_argument("--expl_noise", default=0.1, type=float)		# Std of Gaussian exploration noise
	args = parser.parse_args()

	file_name = "HER_%s_%s" % (args.env_name, str(args.seed))
	print("---------------------------------------")
	print("Settings: " + file_name)
	print("---------------------------------------")

	if not os.path.exists("./her_models"):
		os.makedirs("./her_models")

	env = gym.make(args.env_name)

	# Set seeds
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	obs_space = env.observation_space.spaces['observation']
	goal_space = env.observation_space.spaces['desired_goal']

	state_dim = obs_space.shape[0]
	goal_dim = goal_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	# Initialize policy and buffer
	policy = HER.HER(state_dim, action_dim, goal_dim, max_action)
	replay_buffer = utils.HERReplayBuffer()
	
	total_timesteps = 0
	episode_num = 0
	done = True

	while total_timesteps < args.max_timesteps:
		if done: 
			if total_timesteps != 0: 
				h_goal = episode_storage[-1][-2].copy()
				for item in episode_storage:
					o_g, o2_g, a_g, a_goal, d_g = item
					rew_g = env.compute_reward(a_goal, h_goal, {})
					replay_buffer.add((o_g, o2_g, a_g, rew_g, h_goal, d_g))
				success_rew = float(info['is_success'])
				print("Total T: %d Episode Num: %d Episode T: %d Reward: %f SuccessRate: %f" %
					(total_timesteps, episode_num, episode_timesteps, episode_reward, success_rew))
				policy.train(replay_buffer, episode_timesteps)
			
			# Save policy
			if total_timesteps % 1e5 == 0:
				policy.save(file_name, directory="./her_models")
			
			# Reset environment
			obs_dict = env.reset()
			obs = obs_dict['observation']
			goal = obs_dict['desired_goal']
			achieved_goal = obs_dict['achieved_goal']
			done = False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 
			episode_storage = []
		
		# Select action randomly or according to policy
		if total_timesteps < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = policy.select_action(np.array(obs), np.array(goal))
			if args.expl_noise != 0:
				action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)

		# Perform action
		new_obs, reward, done, info = env.step(action)
		done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
		episode_reward += reward

		# Store data in replay buffer
		replay_buffer.add((obs, new_obs['observation'], action, reward, goal, done_bool))
		episode_storage.append((obs, new_obs['observation'], action, new_obs['achieved_goal'], done_bool))
		obs = new_obs['observation']
		goal = new_obs['desired_goal']
		achieved_goal = new_obs['achieved_goal']
		episode_timesteps += 1
		total_timesteps += 1
		
	# Save final policy
	policy.save("%s" % (file_name), directory="./her_models")