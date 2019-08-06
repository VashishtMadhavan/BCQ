import numpy as np
import gym
import argparse
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from colour import Color
import imageio


if __name__ == "__main__":
	sns.set()
	sns.set_color_codes('deep')
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", type=str, default="AntMaze-v2")
	parser.add_argument("--buffer", type=str)
	parser.add_argument("--chunks", type=int, default=500)
	args = parser.parse_args()

	# Loading data and env
	env = gym.make(args.env)
	if not args.buffer:
		raise Error("No checkpoint file found...")
	buffer_data = np.load(args.buffer, allow_pickle=True)
	xy_points = np.array([x[1][2:5] for x in buffer_data])
	chunk_size = xy_points.shape[0] // args.chunks
	frames = []

	for i in range(args.chunks):
		fig, ax = plt.subplots()
		points = xy_points[:(chunk_size*(i+1))]
		ax.set(title="Replay Buffer Iteration: {}".format(chunk_size * (i+1)))
		# drawing in walls
		#colors = list(Color("green").range_to(Color("black"), len(xy_points)))
		#colors = np.array([c.rgb for c in colors])
		ax.set_xlim(-1, 5)
		ax.set_ylim(-1, 5)
		ax.scatter(points[:,0], points[:,1], s=20, alpha=0.5)

		# drawing image and getting an RGB object
		fig.canvas.draw()
		image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
		image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
		frames.append(image)
	imageio.mimsave('./buffer_viz.gif', frames, fps=1)

