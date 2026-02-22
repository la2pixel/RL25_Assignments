import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Actor(nn.Module):
    """Same arch as before; layers.0/layers.1/out so state_dict matches saved checkpoints."""
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(state_dim, 512),
            nn.Linear(512, 512),
        ])
        self.out = nn.Linear(512, action_dim)

    def forward(self, state):
        x = state
        for layer in self.layers:
            x = F.relu(layer(x))
        return torch.tanh(self.out(x))


class Critic(nn.Module):
    """Same arch as before; layers.0/layers.1/out so state_dict matches saved checkpoints."""
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(state_dim + action_dim, 512),
            nn.Linear(512, 512),
        ])
        self.out = nn.Linear(512, 1)

    def forward(self, state, action):
        x = torch.hstack([state, action])
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.out(x)


class TD3Agent:
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        polyak=0.995,
        policy_lr=3e-4,
        critic_lr=3e-4,
        act_noise_std=0.1,
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

        # tracking
        self.total_updates = 0

        print(f"using device: {self.device}")

    # soft update of target network: blend target with main network using polyak
    def polyak_update(self, main_network, target_network):
        for main_param, target_param in zip(main_network.parameters(), target_network.parameters()):
            target_param.data.copy_(
                self.polyak * target_param.data + (1 - self.polyak) * main_param.data
            )

    def select_action(self, state, add_noise=True, deterministic=None):
        if deterministic is not None:
            add_noise = not deterministic

        # ensure we have a batch dimension: (state_dim,) -> (1, state_dim), (batch, state_dim) stays (batch, state_dim)
        state_np = np.asarray(state, dtype=np.float32)
        if state_np.ndim == 1:
            state_batch = state_np.reshape(1, -1)
            single_input = True
        else:
            state_batch = state_np
            single_input = False

        state_tensor = torch.FloatTensor(state_batch).to(self.device)
        action = self.actor(state_tensor).cpu().detach().numpy()

        # add exploration noise per action dimension (same for each row in the batch)
        if add_noise:
            noise = np.random.normal(0, self.act_noise_std, size=action.shape)
            action = action + noise
            action = np.clip(action, -1, 1)

        # return same shape as expected: 1D for single obs, 2D for batch
        if single_input:
            return action.flatten()
        return action

    def update(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return
        observations, actions, rewards, next_observations, dones = replay_buffer.sample(batch_size)
        self.total_updates += 1

        # computing target Q-Values
        with torch.no_grad():
            # target policy smoothing to reduce exploitation in Q-Value errors
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
