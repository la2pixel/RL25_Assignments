import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, dropout):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 1024)
        self.layer2 = nn.Linear(1024, 1024)
        self.layer3 = nn.Linear(1024, action_dim)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, state):
        x = self.dropout(F.relu(self.layer1(state)))
        x = self.dropout(F.relu(self.layer2(x)))
        x = torch.tanh(self.layer3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, dropout):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 1024)
        self.layer2 = nn.Linear(1024, 1024)
        self.layer3 = nn.Linear(1024, 1)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, state, action):
        # horizontally stack state and action for the critic as input
        x = torch.hstack([state, action])
        x = self.dropout(F.relu(self.layer1(x)))
        x = self.dropout(F.relu(self.layer2(x)))
        x = self.layer3(x)
        return x


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
        improvement=True,
    ):
        # decay: linear from start to _end over training
        self.improvement = improvement
        weight_decay = 1e-5 if improvement else 0
        weight_decay_end = 0
        dropout = 0.1 if improvement else 0
        dropout_end = 0
        policy_noise_end = 0.05 if improvement else policy_noise

        # current decay values for improvement params
        self._dropout, self._dropout_end = dropout, dropout_end
        self._weight_decay, self._weight_decay_end = weight_decay, weight_decay_end
        self._policy_noise, self._policy_noise_end = policy_noise, policy_noise_end

        # check if gpu is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # initializing networks
        # main actor
        self.actor = Actor(state_dim, action_dim, dropout).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=policy_lr, weight_decay=weight_decay)

        # target actor
        self.actor_target = Actor(state_dim, action_dim, dropout).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # main critic1
        self.critic1 = Critic(state_dim, action_dim, dropout).to(self.device)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr, weight_decay=weight_decay)

        # target critic1
        self.critic1_target = Critic(state_dim, action_dim, dropout).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        # main critic2
        self.critic2 = Critic(state_dim, action_dim, dropout).to(self.device)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr, weight_decay=weight_decay)

        # target critic2
        self.critic2_target = Critic(state_dim, action_dim, dropout).to(self.device)
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
        # AI Usage: asked AI how to update target network params with the main network params
        for main_param, target_param in zip(main_network.parameters(), target_network.parameters()):
            target_param.data.copy_(
                self.polyak * target_param.data + (1 - self.polyak) * main_param.data
            )

    def _apply_decay(self, progress):
        # set dropout, weight_decay, policy_noise to decay linearly
        dropout_p = self._dropout + (self._dropout_end - self._dropout) * progress
        wd = self._weight_decay + (self._weight_decay_end - self._weight_decay) * progress
        self.policy_noise = self._policy_noise + (self._policy_noise_end - self._policy_noise) * progress

        # AI Usage: asked AI how to update params
        for net in (self.actor, self.actor_target, self.critic1, self.critic1_target,
                    self.critic2, self.critic2_target):
            net.dropout.p = dropout_p
        for opt in (self.actor_optimizer, self.critic1_optimizer, self.critic2_optimizer):
            for group in opt.param_groups:
                group["weight_decay"] = wd

    def select_action(self, state, add_noise=True, deterministic=None):
        # added deterministic param for evaluate_hockey script compatibility
        if deterministic is not None:
            add_noise = not deterministic

        # ensure we have a batch dimension: (state_dim,) -> (1, state_dim)
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

    def update(self, replay_buffer, batch_size, step=None, max_steps=None):
        # wait til we have enough transition to sample a whole batch
        if len(replay_buffer) < batch_size:
            return
        # decay weight_decay, dropout and policy noise if improvement is set
        if self.improvement and step is not None and max_steps is not None and max_steps > 0:
            progress = step / max_steps
            self._apply_decay(progress)
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
            # Note: I update critic targets delayed as well, as this is how it is written in the pseudo-code
            # OpenAI-Spinning https://spinningup.openai.com/en/latest/algorithms/td3.html
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
