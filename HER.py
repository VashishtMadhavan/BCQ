import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, goal_dim, max_action):
		super(Actor, self).__init__()
		self.l1 = nn.Linear(state_dim + goal_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)
		self.max_action = max_action

	def forward(self, x, g):
		out = torch.cat([x, g], 1)
		x = F.relu(self.l1(out))
		x = F.relu(self.l2(x))
		x = self.max_action * torch.tanh(self.l3(x)) 
		return x

class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, goal_dim):
		super(Critic, self).__init__()
		# Q1 architecture
		self.l1 = nn.Linear(state_dim + goal_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + goal_dim + action_dim, 400)
		self.l5 = nn.Linear(400, 300)
		self.l6 = nn.Linear(300, 1)

	def forward(self, x, a, g):
		concat = torch.cat([x, g, a], 1)
		x1 = F.relu(self.l1(concat))
		x1 = F.relu(self.l2(x1))

		x2 = F.relu(self.l4(concat))
		x2 = F.relu(self.l5(x2))
		return self.l3(x1), self.l6(x2)

	def Q1(self, x, a, g):
		concat = torch.cat([x, g, a], 1)
		x1 = F.relu(self.l1(concat))
		x1 = F.relu(self.l2(x1))
		return self.l3(x1)

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3) w/
# Hindsight Experience Replay
# Paper: https://arxiv.org/abs/1802.09477

class HER(object):
	def __init__(self, state_dim, action_dim, goal_dim, max_action):
		self.actor = Actor(state_dim, action_dim, goal_dim, max_action).to(device)
		self.actor_target = Actor(state_dim, action_dim, goal_dim, max_action).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

		self.critic = Critic(state_dim, action_dim, goal_dim).to(device)
		self.critic_target = Critic(state_dim, action_dim, goal_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
		self.max_action = max_action

	def select_action(self, state, goal):
		with torch.no_grad():
			state = torch.FloatTensor(state).to(device)
			goal = torch.FloatTensor(goal).to(device)
			return self.actor(state, goal).cpu().numpy()

	def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
		for it in range(iterations):
			# Sample replay buffer 
			state, next_state, act, reward, goal, done = replay_buffer.sample(batch_size)
			state 		= torch.FloatTensor(state).to(device)
			action 		= torch.FloatTensor(act).to(device)
			next_state 	= torch.FloatTensor(next_state).to(device)
			reward 		= torch.FloatTensor(reward).to(device)
			goal        = torch.FloatTensor(goal).to(device)
			done 		= torch.FloatTensor(1 - done).to(device)

			# Select action according to policy and add clipped noise 
			noise = torch.FloatTensor(act).data.normal_(0, policy_noise).to(device)
			noise = noise.clamp(-noise_clip, noise_clip)
			next_action = (self.actor_target(next_state, goal) + noise).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action, goal)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + (done * discount * target_Q).detach()

			# Get current Q estimates
			current_Q1, current_Q2 = self.critic(state, action, goal)

			# Compute critic loss
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			# Delayed policy updates
			if it % policy_freq == 0:
				# Compute actor loss
				actor_loss = -self.critic.Q1(state, self.actor(state, goal), goal).mean()
				
				# Optimize the actor 
				self.actor_optimizer.zero_grad()
				actor_loss.backward()
				self.actor_optimizer.step()

				# Update the frozen target models
				for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
					target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

				for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
					target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

	def save(self, filename, directory):
		torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
		torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

	def load(self, filename, directory):
		map_location = None if torch.cuda.is_available() else 'cpu'
		self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename), map_location=map_location))
		self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename), map_location=map_location))
