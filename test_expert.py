import numpy as np
import torch
import gym
import argparse
import os

import utils
import TD3

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", type=str, default="Ant-v2")
	parser.add_argument("--checkpoint", type=str)
	parser.add_argument("--test_eps", type=int, default=10)
	parser.add_argument("--render",  action="store_true")
	parser.add_argument("--rpf", action="store_true")
	parser.add_argument("--K", type=int, default=1)
	args = parser.parse_args()

	env = gym.make(args.env)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])
	if not args.checkpoint:
		raise Error("No checkpoint file found...")

	directory = args.checkpoint.split('/')[0]
	file_name = args.checkpoint.split('/')[1]
	algo = file_name.split("_")[0]

	# Set device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Initialize policy and buffer
	policy = TD3.TD3(state_dim, action_dim, max_action, device, K=args.K, rpf=args.rpf)
	policy.load(file_name, directory='./{}'.format(directory))
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