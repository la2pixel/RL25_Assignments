"""
Soft Actor-Critic (SAC) with optional Pink Noise Exploration.

When pink_noise=False (default): Standard SAC with white Gaussian exploration.
When pink_noise=True: Temporally correlated pink noise replaces iid Gaussian sampling.

Reference: Eberhard et al. (2023) "Pink Noise Is All You Need: Colored Noise
Exploration in Deep Reinforcement Learning" (ICLR 2023 Spotlight)

The key idea: Standard SAC samples actions from a Gaussian with iid noise at
each timestep (white noise). This produces jittery, uncorrelated exploration.
Pink noise (1/f noise, beta=1) is temporally correlated — consecutive samples
are similar, producing smooth, sustained exploration trajectories. This helps
discover multi-step action sequences needed for tasks like hitting a puck.

Implementation: We generate pink noise sequences in the frequency domain using
the Timmer & König (1995) algorithm, pre-buffer them, and use them as the
noise source for the reparameterization trick instead of iid Gaussian samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


# ============================================================
# Pink Noise Generator (Timmer & König 1995)
# ============================================================

def powerlaw_psd_gaussian(beta, size, fmin=0, rng=None):
    """
    Generate Gaussian-distributed noise with a power-law PSD: S(f) = 1/f^beta.
    
    beta=0: white noise (no correlation)
    beta=1: pink noise (1/f, the sweet spot per the paper)
    beta=2: red/Brownian noise (strong correlation)
    
    Based on the algorithm from:
    Timmer, J. and König, M. (1995) "On generating power law noise"
    Astronomy and Astrophysics, 300, 707-710
    
    Parameters
    ----------
    beta : float
        Exponent of the power law. 1.0 = pink noise.
    size : tuple or int
        Shape of output. Last dimension is the time axis.
    fmin : float
        Low-frequency cutoff (0 = no cutoff)
    rng : np.random.Generator
        Random number generator
        
    Returns
    -------
    np.ndarray
        Colored noise samples with unit variance, shape = size
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if isinstance(size, int):
        samples = size
        shape_prefix = ()
    else:
        samples = size[-1]
        shape_prefix = size[:-1]
    
    # Frequencies for rfft
    f = np.fft.rfftfreq(samples)
    
    # Minimum frequency
    if fmin <= 0:
        fmin = 1.0 / samples  # default: lowest non-zero frequency
    
    # Build the PSD: S(f) = 1/f^beta, but clip at fmin
    s_scale = np.where(f < fmin, fmin, f)
    s_scale = s_scale ** (-beta / 2.0)
    
    # Generate random complex coefficients
    dims = shape_prefix + (len(f),)
    sr = rng.normal(size=dims)
    si = rng.normal(size=dims)
    
    # DC and Nyquist components must be real
    if samples % 2 == 0:
        si[..., -1] = 0
        sr[..., 0] = sr[..., 0] * np.sqrt(2)
    si[..., 0] = 0
    
    # Combine and scale by PSD
    s = (sr + 1j * si) * s_scale
    
    # Inverse FFT to time domain
    y = np.fft.irfft(s, n=samples, axis=-1)
    
    # Normalize to unit variance
    y_std = y.std(axis=-1, keepdims=True)
    y_std = np.where(y_std == 0, 1.0, y_std)
    y = y / y_std
    
    return y.astype(np.float32)


class ColoredNoiseProcess:
    """
    Generates temporally correlated noise for exploration.
    
    Pre-generates a buffer of colored noise and steps through it.
    When the buffer is exhausted, a new one is generated.
    
    For SAC: This replaces the iid Gaussian sampling in the
    reparameterization trick with temporally correlated samples.
    
    Parameters
    ----------
    beta : float
        Noise color exponent (1.0 = pink)
    action_dim : int
        Dimensionality of noise
    seq_len : int
        Buffer length (typically episode length or longer)
    """
    
    def __init__(self, beta, action_dim, seq_len=1000, rng=None):
        self.beta = beta
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.rng = rng or np.random.default_rng()
        self.idx = 0
        self._generate_buffer()
    
    def _generate_buffer(self):
        """Generate a fresh buffer of colored noise."""
        # Shape: (action_dim, seq_len)
        self.buffer = powerlaw_psd_gaussian(
            self.beta, (self.action_dim, self.seq_len), rng=self.rng
        )
        self.idx = 0
    
    def sample(self):
        """
        Get one timestep of colored noise.
        
        Returns
        -------
        np.ndarray, shape (action_dim,)
        """
        if self.idx >= self.seq_len:
            self._generate_buffer()
        noise = self.buffer[:, self.idx]
        self.idx += 1
        return noise
    
    def sample_batch(self, batch_size):
        """
        Get a batch of consecutive colored noise samples.
        
        Returns
        -------
        np.ndarray, shape (batch_size, action_dim)
        """
        samples = []
        for _ in range(batch_size):
            samples.append(self.sample())
        return np.stack(samples, axis=0)
    
    def reset(self):
        """Reset the noise process (generate fresh buffer)."""
        self._generate_buffer()


# ============================================================
# Actor
# ============================================================

class Actor(nn.Module):
    """
    Gaussian policy network.
    
    When using pink noise, the sample() method can accept external noise
    instead of sampling from iid Gaussian.
    """
    
    def __init__(self, obs_dim, action_dim, hidden_sizes=[256, 256], 
                 log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        self.net = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(prev_size, action_dim)
        self.log_std_layer = nn.Linear(prev_size, action_dim)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.orthogonal_(self.log_std_layer.weight, gain=0.01)
        
    def forward(self, obs):
        x = self.net(obs)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, obs, deterministic=False, with_logprob=True, 
               external_noise=None):
        """
        Sample action from policy.
        
        Parameters
        ----------
        obs : torch.Tensor
        deterministic : bool
            If True, return tanh(mean)
        with_logprob : bool
            If True, compute log probability
        external_noise : torch.Tensor, optional
            Pre-generated noise (e.g. pink noise) to use instead of
            iid Gaussian. Shape must match (batch, action_dim).
            Reparameterization: x = mean + std * noise
        """
        mean, log_std = self.forward(obs)
        
        if deterministic:
            return torch.tanh(mean), None
        
        std = torch.exp(log_std)
        
        if external_noise is not None:
            # Pink noise path: use pre-generated correlated noise
            x = mean + std * external_noise
        else:
            # Standard SAC path: iid Gaussian via rsample
            normal = Normal(mean, std)
            x = normal.rsample()
        
        action = torch.tanh(x)
        
        if with_logprob:
            # Log prob (same formula regardless of noise source)
            var = std ** 2
            log_prob = -0.5 * ((x - mean) ** 2 / var + 2 * log_std + np.log(2 * np.pi))
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            return action, log_prob
        
        return action, None


# ============================================================
# Critics
# ============================================================

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=[256, 256]):
        super(Critic, self).__init__()
        layers = []
        prev_size = obs_dim + action_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


class TwinCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=[256, 256]):
        super(TwinCritic, self).__init__()
        self.critic1 = Critic(obs_dim, action_dim, hidden_sizes)
        self.critic2 = Critic(obs_dim, action_dim, hidden_sizes)
        
    def forward(self, obs, action):
        return self.critic1(obs, action), self.critic2(obs, action)


# ============================================================
# SAC Agent
# ============================================================

class SAC:
    """
    SAC agent with optional pink noise exploration.
    
    Usage:
        agent = SAC(obs_dim, act_dim, device)                      # vanilla
        agent = SAC(obs_dim, act_dim, device, pink_noise=True)     # pink noise
    """
    
    def __init__(self,
                 obs_dim,
                 action_dim,
                 device,
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 alpha_lr=3e-4,
                 hidden_sizes=[256, 256],
                 gamma=0.99,
                 tau=0.005,
                 alpha=0.2,
                 auto_entropy_tuning=True,
                 target_entropy=None,
                 grad_clip=None,
                 pink_noise=False,
                 noise_beta=1.0,
                 noise_seq_len=1000):
        """
        Parameters
        ----------
        pink_noise : bool
            If True, use colored noise for exploration instead of iid Gaussian.
        noise_beta : float
            Color exponent. 1.0 = pink (recommended). 0 = white. 2 = red.
        noise_seq_len : int
            Length of pre-generated noise buffer.
        """
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.grad_clip = grad_clip
        self.pink_noise = pink_noise
        self.noise_beta = noise_beta
        self.noise_seq_len = noise_seq_len
        
        self._alpha = alpha
        self.auto_entropy_tuning = auto_entropy_tuning
        
        # Actor
        self.actor = Actor(obs_dim, action_dim, hidden_sizes).to(device)
        
        # Critics + target
        self.critic = TwinCritic(obs_dim, action_dim, hidden_sizes).to(device)
        self.critic_target = TwinCritic(obs_dim, action_dim, hidden_sizes).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Entropy tuning
        if auto_entropy_tuning:
            self.target_entropy = target_entropy if target_entropy is not None else -action_dim
            self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32,
                                         requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        
        # Pink noise process for single-env select_action
        if pink_noise:
            self._noise_process = ColoredNoiseProcess(
                noise_beta, action_dim, noise_seq_len
            )
        else:
            self._noise_process = None
    
    @property
    def alpha(self):
        if self.auto_entropy_tuning:
            return self.log_alpha.exp().item()
        return self._alpha
    
    @alpha.setter
    def alpha(self, value):
        self._alpha = value
    
    def select_action(self, obs, deterministic=False):
        """Select action, optionally using pink noise for exploration."""
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                if obs_tensor.dim() == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
            else:
                obs_tensor = obs.to(self.device)
                if obs_tensor.dim() == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
            
            if deterministic or not self.pink_noise:
                action_tensor, _ = self.actor.sample(
                    obs_tensor, deterministic=deterministic, with_logprob=False
                )
            else:
                noise = self._noise_process.sample()
                noise_tensor = torch.FloatTensor(noise).unsqueeze(0).to(self.device)
                action_tensor, _ = self.actor.sample(
                    obs_tensor, deterministic=False, with_logprob=False,
                    external_noise=noise_tensor
                )
            
            return action_tensor.cpu().numpy().flatten()
    
    def reset_noise(self):
        """Reset pink noise process (call at episode boundaries)."""
        if self._noise_process is not None:
            self._noise_process.reset()
    
    def update(self, replay_buffer, batch_size):
        """
        Standard SAC update (critic, actor, alpha, soft target).
        
        NOTE: Updates always use iid Gaussian (standard reparameterization).
        Pink noise only affects action SELECTION during rollouts.
        The policy is still a Gaussian — we just sample from it differently
        when collecting data.
        """
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = \
            replay_buffer.sample(batch_size)
        
        if isinstance(obs_batch, torch.Tensor):
            obs, actions, next_obs, rewards, dones = \
                obs_batch, act_batch, next_obs_batch, rew_batch, done_batch
        else:
            obs = torch.FloatTensor(obs_batch).to(self.device)
            actions = torch.FloatTensor(act_batch).to(self.device)
            next_obs = torch.FloatTensor(next_obs_batch).to(self.device)
            rewards = torch.FloatTensor(rew_batch).unsqueeze(1).to(self.device)
            dones = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)
        
        # --- Critic update ---
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * q_next
        
        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()
        
        # --- Actor update ---
        new_actions, log_probs = self.actor.sample(obs)
        q1_new, q2_new = self.critic(obs, new_actions)
        actor_loss = (self.alpha * log_probs - torch.min(q1_new, q2_new)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()
        
        # --- Alpha update ---
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = torch.tensor(0.0)
        
        # --- Soft target update ---
        for tp, sp in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item() if isinstance(alpha_loss, torch.Tensor) else alpha_loss,
            'alpha': self.alpha,
            'mean_q1': q1.mean().item(),
            'mean_q2': q2.mean().item(),
            'mean_log_prob': log_probs.mean().item(),
        }
    
    def save(self, filepath):
        save_dict = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'auto_entropy_tuning': self.auto_entropy_tuning,
            'pink_noise': self.pink_noise,
        }
        if self.auto_entropy_tuning:
            save_dict['log_alpha'] = self.log_alpha
            save_dict['alpha_optimizer'] = self.alpha_optimizer.state_dict()
            save_dict['target_entropy'] = self.target_entropy
        else:
            save_dict['alpha'] = self._alpha
        torch.save(save_dict, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        if self.auto_entropy_tuning and 'log_alpha' in checkpoint:
            self.log_alpha.data = checkpoint['log_alpha'].data
            if 'alpha_optimizer' in checkpoint:
                self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        elif not self.auto_entropy_tuning and 'alpha' in checkpoint:
            self._alpha = checkpoint['alpha']