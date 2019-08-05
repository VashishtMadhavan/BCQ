import numpy as np
import gym
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
	sns.set()
	sns.set_color_codes('deep')
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", type=str, default="AntMaze-v2")
	parser.add_argument("--buffer", type=str)
	args = parser.parse_args()

	# Loading data and env
	env = gym.make(args.env)
	if not args.buffer:
		raise Error("No checkpoint file found...")
	buffer_data = np.load(args.buffer, allow_pickle=True)
	xy_points = np.array([x[1][2:5] for x in buffer_data])
	xy_points = xy_points

	plt.title("Scatterplot of Replay Buffer")
	# drawing in walls
	plt.xlim(-1, 5)
	plt.ylim(-1, 5)
	plt.scatter(xy_points[:,0], xy_points[:,1], s=20)
	plt.show()
