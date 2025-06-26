import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque, defaultdict
import pickle
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any

from environment import AirlineSchedulingMAEnvironment

class MAPPONetwork(nn.Module):
    """
    MAPPO Actor-Critic Network with shared feature extraction
    """

    def __init__(self, obs_dim: int, action_space, hidden_dim: int = 256, is_critic: bool = False):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_space = action_space
        self.hidden_dim = hidden_dim
        self.is_critic = is_critic

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        if is_critic:
            # Critic head - outputs single value
            self.value_head = nn.Linear(hidden_dim // 2, 1)
        else:
            # Actor head - outputs action distribution parameters
            self._init_actor_head()

    def _init_actor_head(self):
        """Initialize actor head based on action space type"""
        from gymnasium import spaces

        if isinstance(self.action_space, spaces.MultiBinary):
            # For binary actions (base agent, emissions agent)
            self.action_head = nn.Linear(self.hidden_dim // 2, self.action_space.n)
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            # For discrete actions (fleet agent, crew agent)
            self.action_heads = nn.ModuleList([
                nn.Linear(self.hidden_dim // 2, int(dim))
                for dim in self.action_space.nvec
            ])
        else:
            raise ValueError(f"Unsupported action space type: {type(self.action_space)}")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        features = self.feature_extractor(obs)

        if self.is_critic:
            return self.value_head(features)
        else:
            return self._get_action_logits(features)

    def _get_action_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Get action logits based on action space type"""
        from gymnasium import spaces

        if isinstance(self.action_space, spaces.MultiBinary):
            # Binary actions - sigmoid output
            return torch.sigmoid(self.action_head(features))
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            # Multi-discrete actions - softmax for each dimension
            logits = []
            for head in self.action_heads:
                logits.append(F.log_softmax(head(features), dim=-1))
            return torch.cat(logits, dim=-1)

    def get_action_and_log_prob(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and return log probability"""
        action_logits = self.forward(obs)

        from gymnasium import spaces
        if isinstance(self.action_space, spaces.MultiBinary):
            # Binary sampling
            dist = torch.distributions.Bernoulli(action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            # Multi-discrete sampling
            actions = []
            log_probs = []
            start_idx = 0

            for i, dim in enumerate(self.action_space.nvec):
                end_idx = start_idx + int(dim)
                logits_slice = action_logits[:, start_idx:end_idx]
                dist = Categorical(logits=logits_slice)
                action_slice = dist.sample()
                log_prob_slice = dist.log_prob(action_slice)

                actions.append(action_slice)
                log_probs.append(log_prob_slice)
                start_idx = end_idx

            action = torch.stack(actions, dim=-1)
            log_prob = torch.stack(log_probs, dim=-1).sum(dim=-1)

        return action, log_prob


class CentralizedCritic(nn.Module):
    """
    Centralized Critic for CTDE (Centralized Training, Decentralized Execution)
    Takes concatenated observations from all agents
    """

    def __init__(self, total_obs_dim: int, hidden_dim: int = 512):
        super().__init__()

        self.critic_network = nn.Sequential(
            nn.Linear(total_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_obs: Concatenated observations from all agents
        Returns:
            Value estimate
        """
        return self.critic_network(global_obs)


class ReplayBuffer:
    """Experience replay buffer for MAPPO"""

    def __init__(self, buffer_size: int, num_agents: int):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.clear()

    def clear(self):
        """Clear the buffer"""
        self.observations = {f'agent_{i}': [] for i in range(self.num_agents)}
        self.global_observations = []
        self.actions = {f'agent_{i}': [] for i in range(self.num_agents)}
        self.rewards = {f'agent_{i}': [] for i in range(self.num_agents)}
        self.log_probs = {f'agent_{i}': [] for i in range(self.num_agents)}
        self.values = []
        self.dones = []
        self.size = 0

    def store(self, obs_dict: Dict, global_obs: np.ndarray, action_dict: Dict,
              reward_dict: Dict, log_prob_dict: Dict, values: torch.Tensor, done: bool):
        """Store a transition"""
        agent_names = ['base_agent', 'fleet_agent', 'crew_agent', 'emissions_agent']

        for i, agent_name in enumerate(agent_names):
            self.observations[f'agent_{i}'].append(obs_dict[agent_name])
            self.actions[f'agent_{i}'].append(action_dict[agent_name])
            self.rewards[f'agent_{i}'].append(reward_dict[agent_name])
            self.log_probs[f'agent_{i}'].append(log_prob_dict[agent_name])

        self.global_observations.append(global_obs)
        self.values.append(values)
        self.dones.append(done)
        self.size += 1

    def get_batch(self) -> Dict:
        """Get a batch of experiences"""
        batch = {
            'observations': {},
            'global_observations': np.array(self.global_observations),
            'actions': {},
            'rewards': {},
            'log_probs': {},
            'values': torch.stack(self.values),
            'dones': np.array(self.dones)
        }

        for i in range(self.num_agents):
            batch['observations'][f'agent_{i}'] = np.array(self.observations[f'agent_{i}'])
            batch['actions'][f'agent_{i}'] = np.array(self.actions[f'agent_{i}'])
            batch['rewards'][f'agent_{i}'] = np.array(self.rewards[f'agent_{i}'])
            batch['log_probs'][f'agent_{i}'] = torch.stack(self.log_probs[f'agent_{i}'])

        return batch


class MAPPOTrainer:
    """
    Multi-Agent PPO Trainer with Centralized Training, Decentralized Execution (CTDE)
    """

    def __init__(self, env, config: Dict):
        self.env = env
        self.config = config
        self.agent_names = env.agents
        self.num_agents = len(self.agent_names)

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # IMPORTANT: Check observation dimensions first
        self._validate_observation_dimensions()

        # Initialize networks AFTER validation
        self._init_networks()

        # Initialize optimizers
        self._init_optimizers()

        # Experience buffer
        self.buffer = ReplayBuffer(config['buffer_size'], self.num_agents)

        # Training metrics
        self.training_metrics = defaultdict(list)
        self.episode_rewards = {agent: [] for agent in self.agent_names}
        self.episode_lengths = []
        self.episode_scheduled_flights = []  # Track scheduled flights per episode

        # Create save directory
        self.save_dir = config.get('save_dir', 'mappo_models')
        os.makedirs(self.save_dir, exist_ok=True)

    def _validate_observation_dimensions(self):
        """Validate observation dimensions and print debug info"""
        print("\n🔍 VALIDATING OBSERVATION DIMENSIONS")
        print("=" * 50)

        # Get environment info
        print(f"Environment data loaded:")
        if hasattr(self.env, 'flights_data'):
            print(f"  Flights: {len(self.env.flights_data)}")
        if hasattr(self.env, 'aircraft_data'):
            print(f"  Aircraft: {len(self.env.aircraft_data)}")
        if hasattr(self.env, 'crew_data'):
            print(f"  Crew: {len(self.env.crew_data)}")

        # Check observation spaces
        total_obs_dim = 0
        for agent_name in self.agent_names:
            obs_space = self.env.observation_spaces[agent_name]
            obs_dim = obs_space.shape[0] if hasattr(obs_space, 'shape') else len(obs_space)
            total_obs_dim += obs_dim
            print(f"  {agent_name}: {obs_dim} dimensions")

        print(f"  Total observation dimension: {total_obs_dim}")

        # Test actual observation
        try:
            test_obs, _ = self.env.reset()
            print(f"\nActual observation dimensions:")
            actual_total = 0
            for agent_name in self.agent_names:
                actual_dim = len(test_obs[agent_name])
                actual_total += actual_dim
                expected_dim = self.env.observation_spaces[agent_name].shape[0]
                match = "✅" if actual_dim == expected_dim else "❌"
                print(f"  {agent_name}: {actual_dim} (expected: {expected_dim}) {match}")

            print(f"  Actual total: {actual_total}")

            if actual_total != total_obs_dim:
                print(f"⚠️  WARNING: Dimension mismatch detected!")
                print(f"   Expected total: {total_obs_dim}")
                print(f"   Actual total: {actual_total}")

        except Exception as e:
            print(f"❌ Error testing observations: {e}")

        print("=" * 50)

    def _init_networks(self):
        """Initialize actor and critic networks with dynamic dimensions"""
        self.actors = {}

        # Test environment to get actual dimensions
        try:
            test_obs, _ = self.env.reset()
            print("🔧 Initializing networks with actual observation dimensions...")

            # Calculate actual total observation dimension
            actual_obs_dims = {}
            total_obs_dim = 0

            for agent_name in self.agent_names:
                actual_dim = len(test_obs[agent_name])
                actual_obs_dims[agent_name] = actual_dim
                total_obs_dim += actual_dim
                print(f"  {agent_name}: {actual_dim} dimensions")

            print(f"  Total (centralized critic): {total_obs_dim} dimensions")

        except Exception as e:
            print(f"❌ Error getting actual dimensions, using observation_spaces: {e}")
            # Fallback to observation_spaces
            actual_obs_dims = {}
            total_obs_dim = 0
            for agent_name in self.agent_names:
                dim = self.env.observation_spaces[agent_name].shape[0]
                actual_obs_dims[agent_name] = dim
                total_obs_dim += dim

        # Centralized critic with actual dimensions
        self.centralized_critic = CentralizedCritic(
            total_obs_dim=total_obs_dim,
            hidden_dim=self.config['critic_hidden_dim']
        ).to(self.device)

        # Decentralized actors with actual dimensions
        for agent_name in self.agent_names:
            obs_dim = actual_obs_dims[agent_name]
            action_space = self.env.action_spaces[agent_name]

            actor = MAPPONetwork(
                obs_dim=obs_dim,
                action_space=action_space,
                hidden_dim=self.config['actor_hidden_dim'],
                is_critic=False
            ).to(self.device)

            self.actors[agent_name] = actor
            print(f"✅ {agent_name} actor initialized: {obs_dim} → {action_space}")

        print("🎯 All networks initialized successfully!")

    def _init_optimizers(self):
        """Initialize optimizers"""
        # Actor optimizers
        self.actor_optimizers = {}
        for agent_name in self.agent_names:
            self.actor_optimizers[agent_name] = optim.Adam(
                self.actors[agent_name].parameters(),
                lr=self.config['actor_lr']
            )

        # Critic optimizer
        self.critic_optimizer = optim.Adam(
            self.centralized_critic.parameters(),
            lr=self.config['critic_lr']
        )

    def select_actions(self, observations: Dict) -> Tuple[Dict, Dict, torch.Tensor]:
        """Select actions for all agents with dimension validation"""

        # Validate observation dimensions
        for agent_name in self.agent_names:
            expected_dim = self.actors[agent_name].obs_dim
            actual_dim = len(observations[agent_name])
            if expected_dim != actual_dim:
                raise ValueError(f"Observation dimension mismatch for {agent_name}: "
                                 f"expected {expected_dim}, got {actual_dim}")

        actions = {}
        log_probs = {}

        # Get global observation for critic
        global_obs = np.concatenate([observations[agent] for agent in self.agent_names])
        global_obs_tensor = torch.FloatTensor(global_obs).unsqueeze(0).to(self.device)

        # Get value from centralized critic
        with torch.no_grad():
            value = self.centralized_critic(global_obs_tensor).squeeze()

        # Get actions from each actor
        for agent_name in self.agent_names:
            obs_tensor = torch.FloatTensor(observations[agent_name]).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, log_prob = self.actors[agent_name].get_action_and_log_prob(obs_tensor)
                actions[agent_name] = action.squeeze(0).cpu().numpy()
                log_probs[agent_name] = log_prob.squeeze(0)

        return actions, log_probs, value

    def compute_advantages(self, rewards: np.ndarray, values: torch.Tensor,
                           dones: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages using GAE (Generalized Advantage Estimation)"""
        # Ensure values is 1D
        values = values.view(-1)

        advantages = torch.zeros_like(values)
        returns = torch.zeros_like(values)

        gae = 0
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[step + 1]

            delta = rewards[step] + self.config['gamma'] * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.config['gamma'] * self.config['gae_lambda'] * (1 - dones[step]) * gae
            advantages[step] = gae
            returns[step] = advantages[step] + values[step]

        return advantages, returns

    def update_networks(self):
        """Update actor and critic networks using PPO"""
        batch = self.buffer.get_batch()

        # Compute advantages for each agent
        all_advantages = {}
        all_returns = {}

        for i, agent_name in enumerate(self.agent_names):
            agent_rewards = batch['rewards'][f'agent_{i}']
            advantages, returns = self.compute_advantages(
                agent_rewards, batch['values'], batch['dones']
            )
            all_advantages[agent_name] = advantages
            all_returns[agent_name] = returns

        # Update networks
        for epoch in range(self.config['ppo_epochs']):
            self._update_critic(batch, all_returns)
            self._update_actors(batch, all_advantages)

    def _update_critic(self, batch: Dict, all_returns: Dict):
        """Update centralized critic"""
        global_obs = torch.FloatTensor(batch['global_observations']).to(self.device)

        # Use average returns across agents as target
        target_returns = torch.stack([all_returns[agent] for agent in self.agent_names]).mean(dim=0)

        # Compute critic loss - ensure shapes match properly
        values = self.centralized_critic(global_obs)

        # Debug shapes to understand the issue
        # print(f"Raw values shape: {values.shape}")
        # print(f"Target returns shape: {target_returns.shape}")

        # Flatten both to ensure they match
        values = values.view(-1)  # Flatten to 1D
        target_returns = target_returns.view(-1)  # Flatten to 1D

        # print(f"After reshape - Values: {values.shape}, Target: {target_returns.shape}")

        critic_loss = F.mse_loss(values, target_returns.detach())

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.centralized_critic.parameters(), self.config['max_grad_norm'])
        self.critic_optimizer.step()

        # Log metrics
        self.training_metrics['critic_loss'].append(critic_loss.item())

    def _update_actors(self, batch: Dict, all_advantages: Dict):
        """Update decentralized actors with entropy bonus"""
        for i, agent_name in enumerate(self.agent_names):
            obs = torch.FloatTensor(batch['observations'][f'agent_{i}']).to(self.device)
            actions = torch.LongTensor(batch['actions'][f'agent_{i}']).to(self.device)
            old_log_probs = batch['log_probs'][f'agent_{i}'].to(self.device)
            advantages = all_advantages[agent_name].to(self.device)

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Get new action probabilities
            new_action_logits = self.actors[agent_name](obs)

            # Compute new log probabilities
            new_log_probs = self._compute_log_probs(new_action_logits, actions, agent_name)

            # Compute entropy for exploration bonus
            entropy = self._compute_entropy(new_action_logits, agent_name)

            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Compute PPO loss with entropy bonus
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config['clip_ratio'], 1 + self.config['clip_ratio']) * advantages

            # Add entropy bonus for exploration
            entropy_bonus = self.config.get('entropy_coef', 0.01) * entropy.mean()
            actor_loss = -torch.min(surr1, surr2).mean() - entropy_bonus

            # Update actor
            self.actor_optimizers[agent_name].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent_name].parameters(), self.config['max_grad_norm'])
            self.actor_optimizers[agent_name].step()

            # Log metrics including entropy
            self.training_metrics[f'{agent_name}_actor_loss'].append(actor_loss.item())
            self.training_metrics[f'{agent_name}_entropy'].append(entropy.mean().item())

    def _compute_entropy(self, action_logits: torch.Tensor, agent_name: str) -> torch.Tensor:
        """Compute entropy for exploration bonus"""
        action_space = self.env.action_spaces[agent_name]

        from gymnasium import spaces
        if isinstance(action_space, spaces.MultiBinary):
            # Bernoulli entropy
            probs = torch.sigmoid(action_logits)
            entropy = -(probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8))
            return entropy.sum(dim=-1)
        elif isinstance(action_space, spaces.MultiDiscrete):
            # Categorical entropy
            entropies = []
            start_idx = 0
            for dim in action_space.nvec:
                end_idx = start_idx + int(dim)
                logits_slice = action_logits[:, start_idx:end_idx]
                probs = torch.softmax(logits_slice, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
                entropies.append(entropy)
                start_idx = end_idx
            return torch.stack(entropies, dim=-1).sum(dim=-1)

        return torch.zeros(action_logits.shape[0]).to(action_logits.device)

    def _compute_log_probs(self, action_logits: torch.Tensor, actions: torch.Tensor, agent_name: str) -> torch.Tensor:
        """Compute log probabilities for given actions"""
        action_space = self.env.action_spaces[agent_name]

        from gymnasium import spaces
        if isinstance(action_space, spaces.MultiBinary):
            dist = torch.distributions.Bernoulli(action_logits)
            return dist.log_prob(actions.float()).sum(dim=-1)
        elif isinstance(action_space, spaces.MultiDiscrete):
            log_probs = []
            start_idx = 0

            for i, dim in enumerate(action_space.nvec):
                end_idx = start_idx + int(dim)
                logits_slice = action_logits[:, start_idx:end_idx]
                dist = Categorical(logits=logits_slice)
                log_prob_slice = dist.log_prob(actions[:, i])
                log_probs.append(log_prob_slice)
                start_idx = end_idx

            return torch.stack(log_probs, dim=-1).sum(dim=-1)

    def train(self, num_episodes: int):
        """Main training loop with enhanced logging"""
        print(f"Starting MAPPO training for {num_episodes} episodes...")

        for episode in range(num_episodes):
            episode_rewards = {agent: 0 for agent in self.agent_names}
            episode_length = 0
            scheduled_flights_count = 0  # Track scheduled flights this episode

            # Reset environment
            observations, _ = self.env.reset()

            while True:
                # Select actions
                actions, log_probs, value = self.select_actions(observations)

                # Take step in environment
                next_observations, rewards, terminated, truncated, _ = self.env.step(actions)

                # Store experience
                global_obs = np.concatenate([observations[agent] for agent in self.agent_names])
                done = any(terminated.values()) or any(truncated.values())

                self.buffer.store(
                    observations, global_obs, actions, rewards, log_probs, value, done
                )

                # Update metrics
                for agent in self.agent_names:
                    episode_rewards[agent] += rewards[agent]
                episode_length += 1

                # Track scheduled flights
                if hasattr(self.env, 'scheduled_flights'):
                    scheduled_flights_count = len(self.env.scheduled_flights)

                # Check if episode is done
                if done:
                    break

                observations = next_observations

            # Store episode metrics
            for agent in self.agent_names:
                self.episode_rewards[agent].append(episode_rewards[agent])
            self.episode_lengths.append(episode_length)
            self.episode_scheduled_flights.append(scheduled_flights_count)

            # Update networks
            if (episode + 1) % self.config['update_freq'] == 0:
                self.update_networks()
                self.buffer.clear()

            # Logging
            if (episode + 1) % self.config['log_freq'] == 0:
                self._log_progress(episode + 1)

            # Save models
            if (episode + 1) % self.config['save_freq'] == 0:
                self.save_models(episode + 1)

        print("Training completed!")
        self.save_models(num_episodes, final=True)

    def _log_progress(self, episode: int):
        """Log training progress with enhanced metrics"""
        recent_episodes = min(self.config['log_freq'], len(self.episode_rewards[self.agent_names[0]]))

        print(f"\nEpisode {episode}")
        print("-" * 50)

        # Reward statistics
        for agent in self.agent_names:
            recent_rewards = self.episode_rewards[agent][-recent_episodes:]
            avg_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards)
            print(f"{agent}: {avg_reward:.3f} ± {std_reward:.3f}")

        avg_length = np.mean(self.episode_lengths[-recent_episodes:])
        print(f"Average episode length: {avg_length:.1f}")

        # Scheduled flights statistics
        recent_scheduled = self.episode_scheduled_flights[-recent_episodes:]
        avg_scheduled = np.mean(recent_scheduled)
        max_scheduled = np.max(recent_scheduled)
        min_scheduled = np.min(recent_scheduled)
        total_available = len(self.env.current_episode_flights) if hasattr(self.env,
                                                                           'current_episode_flights') else 'N/A'

        print(
            f"Scheduled flights: {avg_scheduled:.1f} avg (min: {min_scheduled}, max: {max_scheduled}) out of {total_available} available")

        # Loss and entropy metrics
        if self.training_metrics['critic_loss']:
            recent_critic_loss = np.mean(self.training_metrics['critic_loss'][-10:])
            print(f"Critic loss: {recent_critic_loss:.6f}")

        # Log entropy for exploration monitoring
        for agent in self.agent_names:
            entropy_key = f'{agent}_entropy'
            if entropy_key in self.training_metrics and self.training_metrics[entropy_key]:
                recent_entropy = np.mean(self.training_metrics[entropy_key][-10:])
                print(f"{agent} entropy: {recent_entropy:.4f}")

        # Environment-specific metrics
        print(f"Environment metrics:")
        obs, _ = self.env.reset()
        test_actions, _, _ = self.select_actions(obs)
        obs, rewards, _, _, infos = self.env.step(test_actions)

        if 'base_agent' in infos:
            scheduled = infos['base_agent'].get('scheduled_flights_count', 0)
            total_flights = self.env.num_flights if hasattr(self.env, 'num_flights') else len(
                self.env.current_episode_flights)
            print(f"  Scheduling rate: {scheduled}/{total_flights} ({scheduled / total_flights * 100:.1f}%)")

            budget_util = infos['base_agent'].get('budget_utilization', 0)
            emission_util = infos['base_agent'].get('emission_utilization', 0)
            print(f"  Budget utilization: {budget_util:.1%}")
            print(f"  Emission utilization: {emission_util:.1%}")


    def save_models(self, episode: int, final: bool = False):
        """Save trained models"""
        suffix = 'final' if final else f'episode_{episode}'

        # Save actors
        for agent_name in self.agent_names:
            torch.save(
                self.actors[agent_name].state_dict(),
                os.path.join(self.save_dir, f'{agent_name}_actor_{suffix}.pth')
            )

        # Save critic
        torch.save(
            self.centralized_critic.state_dict(),
            os.path.join(self.save_dir, f'centralized_critic_{suffix}.pth')
        )

        # Save training metrics
        with open(os.path.join(self.save_dir, f'training_metrics_{suffix}.pkl'), 'wb') as f:
            pickle.dump({
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'episode_scheduled_flights': self.episode_scheduled_flights,
                'training_metrics': dict(self.training_metrics)
            }, f)

        print(f"Models saved with suffix: {suffix}")

    def plot_training_curves(self, save_path: str = None):
        """Plot training curves for all agents"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MAPPO Training Results - Airline Scheduling', fontsize=16, fontweight='bold')

        # Colors for each agent
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        agent_colors = dict(zip(self.agent_names, colors))

        # Plot 1: Episode Rewards over Time
        ax1 = axes[0, 0]
        for agent in self.agent_names:
            episodes = range(1, len(self.episode_rewards[agent]) + 1)
            rewards = self.episode_rewards[agent]

            # Plot raw rewards (light)
            ax1.plot(episodes, rewards, alpha=0.3, color=agent_colors[agent])

            # Plot moving average (bold)
            if len(rewards) >= 10:
                moving_avg = np.convolve(rewards, np.ones(10) / 10, mode='valid')
                moving_episodes = range(10, len(rewards) + 1)
                ax1.plot(moving_episodes, moving_avg, label=agent,
                         color=agent_colors[agent], linewidth=2)

        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Episode Reward')
        ax1.set_title('Episode Rewards (10-episode moving average)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Average Rewards (Last 50 episodes)
        ax2 = axes[0, 1]
        window = min(50, len(self.episode_rewards[self.agent_names[0]]))
        if window > 0:
            agent_avg_rewards = []
            agent_names_clean = []

            for agent in self.agent_names:
                recent_rewards = self.episode_rewards[agent][-window:]
                avg_reward = np.mean(recent_rewards)
                std_reward = np.std(recent_rewards)

                agent_avg_rewards.append(avg_reward)
                agent_names_clean.append(agent.replace('_', ' ').title())

            bars = ax2.bar(agent_names_clean, agent_avg_rewards,
                           color=[agent_colors[agent] for agent in self.agent_names])

            # Add value labels on bars
            for bar, avg_reward in zip(bars, agent_avg_rewards):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{avg_reward:.2f}', ha='center', va='bottom')

        ax2.set_ylabel('Average Reward')
        ax2.set_title(f'Average Rewards (Last {window} episodes)')
        ax2.tick_params(axis='x', rotation=45)

        # Plot 3: Episode Lengths
        ax3 = axes[1, 0]
        episodes = range(1, len(self.episode_lengths) + 1)
        ax3.plot(episodes, self.episode_lengths, alpha=0.6, color='purple')

        if len(self.episode_lengths) >= 10:
            moving_avg_lengths = np.convolve(self.episode_lengths, np.ones(10) / 10, mode='valid')
            moving_episodes = range(10, len(self.episode_lengths) + 1)
            ax3.plot(moving_episodes, moving_avg_lengths, color='darkblue', linewidth=2)

        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Episode Length')
        ax3.set_title('Episode Lengths (10-episode moving average)')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Training Losses
        ax4 = axes[1, 1]
        if self.training_metrics['critic_loss']:
            critic_episodes = np.linspace(1, len(self.episode_rewards[self.agent_names[0]]),
                                          len(self.training_metrics['critic_loss']))
            ax4.plot(critic_episodes, self.training_metrics['critic_loss'],
                     label='Critic Loss', color='red', linewidth=2)

            # Plot actor losses
            for i, agent in enumerate(self.agent_names):
                loss_key = f'{agent}_actor_loss'
                if loss_key in self.training_metrics and self.training_metrics[loss_key]:
                    ax4.plot(critic_episodes, self.training_metrics[loss_key],
                             label=f'{agent.replace("_", " ").title()} Actor',
                             color=agent_colors[agent], alpha=0.7)

        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Loss')
        ax4.set_title('Training Losses')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')  # Log scale for losses

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plots saved to: {save_path}")

        plt.show()


# Training configuration
def get_training_config():
    """Get default training configuration with flat reward fixes"""
    return {
        # Network architecture - smaller for faster learning
        'actor_hidden_dim': 64,  # Reduced from 256
        'critic_hidden_dim': 128,  # Reduced from 512

        # Training hyperparameters - more conservative
        'actor_lr': 1e-4,  # Reduced from 3e-4
        'critic_lr': 5e-4,  # Reduced from 1e-3
        'gamma': 0.95,  # Slightly reduced from 0.99
        'gae_lambda': 0.9,  # Reduced from 0.95
        'clip_ratio': 0.1,  # Reduced from 0.2
        'ppo_epochs': 5,  # Reduced from 10
        'max_grad_norm': 0.3,  # Reduced from 0.5

        # Experience collection - more frequent updates
        'buffer_size': 512,  # Reduced from 2048
        'update_freq': 5,  # Reduced from 10

        # Exploration enhancement
        'entropy_coef': 0.02,  # NEW: Exploration bonus

        # Logging and saving
        'log_freq': 10,  # More frequent logging
        'save_freq': 50,  # More frequent saving
        'save_dir': '01_mappo_airline_models'
    }


def main():
    """Main training function for 100 flights, 80 aircraft, 450 crew"""
    print("Initializing Airline Scheduling MAPPO Training...")
    print("Data: 100 flights, 80 aircraft, 450 crew members")

    # Initialize environment with your data
    try:
        env = AirlineSchedulingMAEnvironment(
            flights_csv_path='flights_data.csv',
            aircraft_csv_path='aircraft_data.csv',
            crew_csv_path='crew_data.csv',
            weekly_budget=50000,
            max_emissions=5000
        )
        print(f"Environment loaded successfully!")
        print(f"Agents: {env.agents}")

        # Detailed observation space checking
        print(f"\nObservation space details:")
        for agent, space in env.observation_spaces.items():
            print(f"  {agent}: {space.shape}")

        print(f"\nAction space details:")
        for agent, space in env.action_spaces.items():
            space_type = type(space).__name__
            if hasattr(space, 'n'):
                space_info = f"n={space.n}"
            elif hasattr(space, 'nvec'):
                space_info = f"nvec={space.nvec}"
            else:
                space_info = str(space)
            print(f"  {agent}: {space_type}({space_info})")

    except Exception as e:
        print(f"Error loading environment: {e}")
        import traceback
        traceback.print_exc()
        return

    # Get training configuration
    config = get_training_config()

    # Initialize trainer
    trainer = MAPPOTrainer(env, config)

    # Train the model
    num_episodes = 1000  # Adjust based on your needs
    trainer.train(num_episodes)

    # Plot results
    plot_save_path = os.path.join(config['save_dir'], 'training_curves.png')
    trainer.plot_training_curves(save_path=plot_save_path)

    print(f"\nTraining completed!")
    print(f"Models saved in: {config['save_dir']}")
    print(f"Training plots saved to: {plot_save_path}")


if __name__ == "__main__":
    main()