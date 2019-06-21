import numpy as np
import torch
import gym
import argparse
import os

import utils
import DDPG
import TD3

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--env_name", default="Ant-v2")
	parser.add_argument("--file_name", default="TD3_Ant-v2_0")
	parser.add_argument("--test_eps", default=10)
	parser.add_argument("--render",  action="store_true")
	args = parser.parse_args()

	env = gym.make(args.env_name)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])
	algo = args.file_name.split("_")[0]

	# Initialize policy and buffer
	if algo == "DDPG":
		policy = DDPG.DDPG(state_dim, action_dim, max_action)
	else:
		policy = TD3.TD3(state_dim, action_dim, max_action)
	policy.load(args.file_name, directory='./pytorch_models')
	episode_num = 0; R = []; T = []

	# Run for args.test_eps episodes
	while episode_num < args.test_eps:
		obs = env.reset(); done = False; ep_rew = 0.; ep_step = 0.
		while not done:
			if args.render: env.render()
			action = policy.select_action(np.array(obs)[None])
			obs, reward, done, _ = env.step(action)
			ep_rew += reward
			ep_step += 1
		R.append(ep_rew)
		T.append(ep_step)

	# Logging Avg Reward + Steps
	print("Avg Rew: ", np.mean(R))
	print("Avg Steps: ", np.mean(T))