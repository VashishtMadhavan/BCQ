import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import utils

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

LOG_STD_MIN = -20
LOG_STD_MAX = 2

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()
		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim * 2)
		self.max_action = max_action

	def mu_var(self, x):
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = self.l3(x)
		mu, log_std = torch.split(x, x.size(1) // 2, dim=1)

		log_std = torch.tanh(log_std)
		log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
		std = torch.exp(log_std)
		return mu, std

	def forward(self, x):
		mu, std = self.mu_var(x)
		normal = Normal(mu, std)
		act = torch.tanh(normal.rsample())
		return self.max_action * act

class EnsembleActor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, K=5):
		super(EnsembleActor, self).__init__()
		self.K = K
		self.max_action = max_action
		self.models = nn.ModuleList([Actor(state_dim, action_dim, max_action) for _ in range(self.K)])

	def mu_var(self, x):
		mu_K = []; var_K = []
		for i in range(self.K):
			mu, var = self.models[i].mu_var(x)
			mu_K.append(mu); var_K.append(var)
		return torch.stack(mu_K), torch.stack(var_K)

	def forward(self, x):
		x = torch.stack([self.models[i](x) for i in range(self.K)])
		return x

class RPFActor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, K=5):
		super(RPFActor, self).__init__()
		self.K = K
		self.max_action = max_action
		self.priors = nn.ModuleList([Actor(state_dim, action_dim, max_action) for _ in range(self.K)])
		self.models = nn.ModuleList([Actor(state_dim, action_dim, max_action) for _ in range(self.K)])
		for p in self.priors.parameters():
			p.requires_grad = False

	def mu_var(self, x):
		mu_K = []; var_K = []
		for i in range(self.K):
			mu, var = self.models[i].mu_var(x)
			mu_p, var_p = self.priors[i].mu_var(x)
			mu_K.append(mu + mu_p.detach()); var_K.append(var + var_p.detach())
		return torch.stack(mu_K), torch.stack(var_K)

	def forward(self, x):
		x = torch.stack([self.priors[i](x).detach() + self.models[i](x) for i in range(self.K)])
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

class RPFCritic(nn.Module):
	def __init__(self, state_dim, action_dim, K=5):
		super(RPFCritic, self).__init__()
		self.K = K
		self.priors = nn.ModuleList([Critic(state_dim, action_dim) for _ in range(self.K)])
		self.models = nn.ModuleList([Critic(state_dim, action_dim) for _ in range(self.K)])
		for p in self.priors.parameters():
			p.requires_grad = False

	def forward(self, x, a):
		q1s = []; q2s = []
		for i in range(self.K):
			if len(a.shape) == 2:
				q1, q2 = self.models[i](x, a)
				p_q1, p_q2 = self.priors[i](x, a)
			else:
				q1, q2 = self.models[i](x, a[i])
				p_q1, p_q2 = self.priors[i](x, a[i])
			q1s.append(q1 + p_q1.detach()); q2s.append(q2 + p_q2.detach())
		return torch.stack(q1s), torch.stack(q2s)

	def Q1(self, x, a):
		qs = []
		for i in range(self.K):
			if len(a.shape) == 2:
				qs.append(self.priors[i].Q1(x, a).detach() + self.models[i].Q1(x, a))
			else:
				qs.append(self.priors[i].Q1(x, a[i]).detach() + self.models[i].Q1(x, a[i]))
		return torch.stack(qs)

class TD3(object):
	def __init__(self, state_dim, action_dim, max_action, device, K=1, rpf=False):
		self.K = K
		self.device = device
		if self.K == 1:
			self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
			self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
			self.critic = Critic(state_dim, action_dim).to(self.device)
			self.critic_target = Critic(state_dim, action_dim).to(self.device)
		else:
			if rpf:
				self.actor = RPFActor(state_dim, action_dim, max_action, K=self.K).to(self.device)
				self.actor_target = RPFActor(state_dim, action_dim, max_action, K=self.K).to(self.device)
				self.critic = RPFCritic(state_dim, action_dim, K=self.K).to(self.device)
				self.critic_target = RPFCritic(state_dim, action_dim, K=self.K).to(self.device)
			else:
				self.actor = EnsembleActor(state_dim, action_dim, max_action, K=self.K).to(self.device)
				self.actor_target = EnsembleActor(state_dim, action_dim, max_action, K=self.K).to(self.device)
				self.critic = EnsembleCritic(state_dim, action_dim, K=self.K).to(self.device)
				self.critic_target = EnsembleCritic(state_dim, action_dim, K=self.K).to(self.device)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
		self.max_action = max_action

	def select_action(self, state):
		with torch.no_grad():
			state = torch.FloatTensor(state).to(self.device)
			if self.K == 1:
				return self.actor(state).cpu().numpy()
			else:
				ensemble_idx = np.random.choice(range(self.K))
				total_actions = self.actor(state).cpu().numpy()
				return total_actions[ensemble_idx]

	# Estimates Jensen-Renyi divergence for ensemble mixture of gaussians
	def jrd(self, state):
		with torch.no_grad():
			mu, var = self.actor.mu_var(state) # [K, B, A]
		mu = torch.tanh(mu) * self.actor.max_action # TODO: confirm that this is correct

		# reshaping tensor to [B, K, A]
		mu = mu.permute(1, 0, 2)
		var = var.permute(1, 0, 2)

		mu_diff = mu.unsqueeze(1) - mu.unsqueeze(2)
		var_sum = var.unsqueeze(1) + var.unsqueeze(2)
		n_act, es, a_s = mu.size()

		err = (mu_diff * 1 / var_sum * mu_diff)
		err = torch.sum(err, dim=-1)
		det = torch.sum(torch.log(var_sum), dim=-1)

		log_z = -0.5 * (err + det)
		log_z = log_z.reshape(n_act, es * es)
		mx, _ = log_z.max(dim=1, keepdim=True)
		log_z = log_z - mx
		exp = torch.exp(log_z).mean(dim=1, keepdim=True)
		ent_mean = -mx - torch.log(exp)
		ent_mean = ent_mean[:, 0]

		# mean of entropies
		total_ent = torch.sum(torch.log(var), dim=-1)
		mean_ent = total_ent.mean(dim=1) / 2 + a_s * np.log(2) / 2
		jrd = ent_mean - mean_ent
		jrd[jrd < 0] = 0
		return jrd

	def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, 
		policy_noise=0.2, noise_clip=0.5, policy_freq=2, priority=False, total_steps=1000):
		for it in range(iterations):
			# Sample replay buffer 
			if priority:
				state, next_state, act, reward, done, weights, idxes = replay_buffer.sample(batch_size, total_steps)
				weights = torch.FloatTensor(weights).to(self.device)
			else:
				state, next_state, act, reward, done = replay_buffer.sample(batch_size)
			state 		= torch.FloatTensor(state).to(self.device)
			action 		= torch.FloatTensor(act).to(self.device)
			next_state 	= torch.FloatTensor(next_state).to(self.device)
			reward 		= torch.FloatTensor(reward).to(self.device)
			done 		= torch.FloatTensor(1 - done).to(self.device)

			# Select action according to policy and add clipped noise
			if self.K == 1:
				noise = torch.FloatTensor(act).data.normal_(0, policy_noise).to(self.device)
			else:
				ens_act = np.repeat(np.expand_dims(act, 0), self.K, axis=0)
				noise = torch.FloatTensor(ens_act).data.normal_(0, policy_noise).to(self.device)
			noise = noise.clamp(-noise_clip, noise_clip)
			next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + (done * discount * target_Q).detach()

			# Get current Q estimates
			current_Q1, current_Q2 = self.critic(state, action)

			# Compute critic loss
			critic_loss = F.mse_loss(current_Q1, target_Q, reduction='none') + F.mse_loss(current_Q2, target_Q, reduction='none')

			# Updating priorities
			if priority:
				if self.K > 1:
					critic_loss = torch.mean(critic_loss, dim=0)
				priorities = critic_loss.detach().cpu().numpy().squeeze()
				replay_buffer.update_priorities(idxes, priorities)
				critic_loss = torch.mean(weights * critic_loss)
			else:
				critic_loss = torch.mean(critic_loss)

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
