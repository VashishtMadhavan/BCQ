import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()
		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)
		self.max_action = max_action

	def forward(self, x):
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = self.max_action * torch.tanh(self.l3(x)) 
		return x

class EnsembleActor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, K=5):
		super(EnsembleActor, self).__init__()
		self.K = K
		self.models = nn.ModuleList([Actor(state_dim, action_dim, max_action) for _ in range(self.K)])

	def forward(self, x):
		x = torch.stack([self.models[i](x) for i in range(self.K)])
		return x

class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()
		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 400)
		self.l5 = nn.Linear(400, 300)
		self.l6 = nn.Linear(300, 1)

	def forward(self, x, a):
		concat = torch.cat([x, a], 1)
		x1 = F.relu(self.l1(concat))
		x1 = F.relu(self.l2(x1))

		x2 = F.relu(self.l4(concat))
		x2 = F.relu(self.l5(x2))
		return self.l3(x1), self.l6(x2)

	def Q1(self, x, a):
		concat = torch.cat([x, a], 1)
		x1 = F.relu(self.l1(concat))
		x1 = F.relu(self.l2(x1))
		return self.l3(x1)

class EnsembleCritic(nn.Module):
	def __init__(self, state_dim, action_dim, K=5):
		super(EnsembleCritic, self).__init__()
		self.K = K
		self.models = nn.ModuleList([Critic(state_dim, action_dim) for _ in range(self.K)])

	def forward(self, x, a):
		q1s = []; q2s = []
		for i in range(self.K):
			if len(a.shape) == 2:
				q1, q2 = self.models[i](x, a)
			else:
				q1, q2 = self.models[i](x, a[i])
			q1s.append(q1); q2s.append(q2)
		return torch.stack(q1s), torch.stack(q2s)

	def Q1(self, x, a):
		qs = []
		for i in range(self.K):
			if len(a.shape) == 2:
				qs.append(self.models[i].Q1(x, a))
			else:
				qs.append(self.models[i].Q1(x, a[i]))
		return torch.stack(qs)


class TD3(object):
	def __init__(self, state_dim, action_dim, max_action, K=1):
		self.K = K
		if self.K == 1:
			self.actor = Actor(state_dim, action_dim, max_action).to(device)
			self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
			self.critic = Critic(state_dim, action_dim).to(device)
			self.critic_target = Critic(state_dim, action_dim).to(device)
		else:
			self.actor = EnsembleActor(state_dim, action_dim, max_action, K=self.K).to(device)
			self.actor_target = EnsembleActor(state_dim, action_dim, max_action, K=self.K).to(device)
			self.critic = EnsembleCritic(state_dim, action_dim, K=self.K).to(device)
			self.critic_target = EnsembleCritic(state_dim, action_dim, K=self.K).to(device)

		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
		self.max_action = max_action

	def select_action(self, state):
		with torch.no_grad():
			state = torch.FloatTensor(state).to(device)
			if self.K == 1:
				return self.actor(state).cpu().numpy()
			else:
				ensemble_idx = np.random.choice(range(self.K))
				total_actions = self.actor(state).cpu().numpy()
				return total_actions[ensemble_idx]

	def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
		for it in range(iterations):
			# Sample replay buffer 
			state, next_state, act, reward, done = replay_buffer.sample(batch_size)
			state 		= torch.FloatTensor(state).to(device)
			action 		= torch.FloatTensor(act).to(device)
			next_state 	= torch.FloatTensor(next_state).to(device)
			reward 		= torch.FloatTensor(reward).to(device)
			done 		= torch.FloatTensor(1 - done).to(device)

			# Select action according to policy and add clipped noise
			if self.K == 1:
				noise = torch.FloatTensor(act).data.normal_(0, policy_noise).to(device)
			else:
				ens_act = np.repeat(np.expand_dims(act, 0), 2, axis=0)
				noise = torch.FloatTensor(ens_act).data.normal_(0, policy_noise).to(device)
			noise = noise.clamp(-noise_clip, noise_clip)
			next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + (done * discount * target_Q).detach()

			# Get current Q estimates
			current_Q1, current_Q2 = self.critic(state, action)

			# Compute critic loss
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			# Delayed policy updates
			if it % policy_freq == 0:
				# Compute actor loss
				actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
				
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
