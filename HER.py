import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Deep Deterministic Policy Gradients (DDPG) with Hindsight Experience Replay (HER)
# Paper: https://arxiv.org/abs/1509.02971
# HER Paper: https://arxiv.org/abs/1707.01495

# Returns an action for a given state
class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, goal_dim, max_action):
		super(Actor, self).__init__()
		self.l1 = nn.Linear(state_dim + goal_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)		
		self.max_action = max_action

	def forward(self, state, goal):
		a = torch.cat([state, goal], 1)
		a = F.relu(self.l1(a))
		a = F.relu(self.l2(a))
		a = self.max_action * torch.tanh(self.l3(a)) 
		return a

# Returns a Q-value for given state/action pair
class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, goal_dim):
		super(Critic, self).__init__()
		self.l1 = nn.Linear(state_dim + goal_dim, 256)
		self.l2 = nn.Linear(256 + action_dim, 256)
		self.l3 = nn.Linear(256, 1)

	def forward(self, state, action, goal):
		q = torch.cat([state, goal], 1)
		q = F.relu(self.l1(q))
		q = F.relu(self.l2(torch.cat([q, action], 1)))
		q = self.l3(q)
		return q

class HER(object):
	def __init__(self, state_dim, action_dim, goal_dim, max_action):
		self.actor = Actor(state_dim, action_dim, goal_dim, max_action).to(device)
		self.actor_target = Actor(state_dim, action_dim, goal_dim, max_action).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

		self.critic = Critic(state_dim, action_dim, goal_dim).to(device)
		self.critic_target = Critic(state_dim, action_dim, goal_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)
		self.state_dim = state_dim

	def select_action(self, state, goal):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		goal = torch.FloatTensor(goal.reshape(1, -1)).to(device)
		return self.actor(state, goal).cpu().data.numpy().flatten()

	def train(self, replay_buffer, iterations=500, batch_size=100, discount=0.99, tau=0.005):
		for it in range(iterations):
			# Each of these are batches 
			state, next_state, action, reward, goal, done = replay_buffer.sample(batch_size)
			state 		= torch.FloatTensor(state).to(device)
			action 		= torch.FloatTensor(action).to(device)
			next_state 	= torch.FloatTensor(next_state).to(device)
			reward 		= torch.FloatTensor(reward).to(device)
			goal        = torch.FloatTensor(goal).to(device)
			done 		= torch.FloatTensor(1 - done).to(device)

			# Compute the target Q value
			target_Q = self.critic_target(next_state, self.actor_target(next_state, goal), goal)
			target_Q = reward + (done * discount * target_Q).detach()

			# Get current Q estimate
			current_Q = self.critic(state, action, goal)

			# Compute critic loss
			critic_loss = F.mse_loss(current_Q, target_Q)

			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			# Compute actor loss
			actor_loss = -self.critic(state, self.actor(state, goal), goal).mean()
			
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