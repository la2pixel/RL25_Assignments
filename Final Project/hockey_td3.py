import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.hstack([state, action])
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class TD3Agent:
    def __init__(
        self,
        state_dim,
        action_dim,
        buffer_size=100000,
        gamma=0.99,
        polyak=0.995,
        policy_lr=3e-4,
        critic_lr=3e-4,
        batch_size=256,
        act_noise_std = 0.1,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
    ):
        # check if gpu is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # initializing networks
        # main actor
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=policy_lr)

        # target actor
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # main critic1
        self.critic1 = Critic(state_dim, action_dim).to(self.device)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)

        # target critic1
        self.critic1_target = Critic(state_dim, action_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        # main critic2
        self.critic2 = Critic(state_dim, action_dim).to(self.device)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # target critic2
        self.critic2_target = Critic(state_dim, action_dim).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # hyperparameters
        self.gamma = gamma
        self.polyak = polyak
        self.act_noise_std = act_noise_std
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.buffer_size = buffer_size
        self.buffer = []
        self.batch_size = batch_size

        # tracking
        self.total_updates = 0

        print(f"using device: {self.device}")

    def store_in_buffer(self, observations, action, reward, next_observation, done):
        self.buffer.append((observations, action, reward, next_observation, done))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def sample_from_buffer(self, batch_size):
        # randomly sample from replay buffer
        batch = random.sample(self.buffer, batch_size)

        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []

        # convert batch into lists of observations, actions, rewards, next_observations, dones
        for observation, action, reward, next_observation, done in batch:
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            next_observations.append(next_observation)
            dones.append(done)

        # convert to numpy array for speed
        observations = np.array(observations)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_observations = np.array(next_observations)
        dones = np.array(dones)

        # convert the numpy arrays into tensors for training; shape: (batch_size, ELEMENT_DIMENSION)
        observations = torch.FloatTensor(observations).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).reshape(-1, 1).to(self.device)
        next_observations = torch.FloatTensor(next_observations).to(self.device)
        dones = torch.FloatTensor(dones).reshape(-1, 1).to(self.device)

        return observations, actions, rewards, next_observations, dones

    # AI Usage
    def polyak_update(self, main_network, target_network):
        for main_param, target_param in zip(main_network.parameters(), target_network.parameters()):
            target_param.data.copy_(
                self.polyak * target_param.data + (1 - self.polyak) * main_param.data
            )

    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().detach().numpy().flatten()

        if add_noise:
            noise = np.random.normal(0, self.act_noise_std, size=action.shape[0])
            action = action + noise
            action = np.clip(action, -1, 1)  # make sure not to go over the valid range of action values [-1, 1]

        return action

    def train(self):
        self.total_updates += 1

        # sampling transitions from the replay buffer
        observations, actions, rewards, next_observations, dones = self.sample_from_buffer(self.batch_size)

        # computing target Q-Values
        with torch.no_grad():
            # target policy smoothing to reduce exploitation in Q-Value errors (AI usage)
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_observations) + noise).clamp(-1, 1)

            target_q1 = self.critic1_target(next_observations, next_actions)
            target_q2 = self.critic2_target(next_observations, next_actions)
            target_q = torch.min(target_q1, target_q2)  # double Q-Learning to reduce overestimation of Q-Values
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # computing current Q-Values
        current_q1 = self.critic1(observations, actions)
        current_q2 = self.critic2(observations, actions)

        # update step for critic 1
        critic1_loss = F.mse_loss(current_q1, target_q)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        # update step for critic 2
        critic2_loss = F.mse_loss(current_q2, target_q)
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # delayed policy updates to reduce volatility
        if self.total_updates % self.policy_delay == 0:
            actor_actions = self.actor(observations)
            # update step for the actor
            actor_loss = -self.critic1(observations, actor_actions).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update target network for actor, critic1 and critic2
            self.polyak_update(self.actor, self.actor_target)
            self.polyak_update(self.critic1, self.critic1_target)
            self.polyak_update(self.critic2, self.critic2_target)

    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
        }, filename)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
