import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import pickle
import os
from typing import Dict, List, Tuple, Any

from environment import make_airline_env


class MAPPONetwork(nn.Module):
    """Enhanced MAPPO Actor-Critic Network with conditional normalization"""

    def __init__(self, obs_dim: int, action_space, hidden_dim: int = 256, is_critic: bool = False):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_space = action_space
        self.hidden_dim = hidden_dim
        self.is_critic = is_critic

        # Enhanced feature extractor with LayerNorm (works with single samples)
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Changed from BatchNorm1d
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Changed from BatchNorm1d
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),  # Changed from BatchNorm1d
            nn.ReLU()
        )

        if is_critic:
            self.value_head = nn.Linear(hidden_dim // 2, 1)
        else:
            self._init_actor_head()

        # Better weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Xavier/Glorot initialization for better gradient flow"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

    def _init_actor_head(self):
        from gymnasium import spaces

        if isinstance(self.action_space, spaces.MultiBinary):
            self.action_head = nn.Linear(self.hidden_dim // 2, self.action_space.n)
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            self.action_heads = nn.ModuleList([
                nn.Linear(self.hidden_dim // 2, int(dim))
                for dim in self.action_space.nvec
            ])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # LayerNorm works with any batch size including single samples
        features = self.feature_extractor(obs)

        if self.is_critic:
            output = self.value_head(features)
        else:
            output = self._get_action_logits(features)

        return output

    def _get_action_logits(self, features: torch.Tensor) -> torch.Tensor:
        from gymnasium import spaces

        if isinstance(self.action_space, spaces.MultiBinary):
            return torch.sigmoid(self.action_head(features))
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            logits = []
            for head in self.action_heads:
                logits.append(F.log_softmax(head(features), dim=-1))
            return torch.cat(logits, dim=-1)

    def get_action_and_log_prob(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action_logits = self.forward(obs)

        from gymnasium import spaces
        if isinstance(self.action_space, spaces.MultiBinary):
            # Add exploration bonus for better exploration
            exploration_noise = 0.1
            probs = torch.clamp(action_logits + exploration_noise * torch.randn_like(action_logits), 0.01, 0.99)
            dist = Bernoulli(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            actions = []
            log_probs = []
            start_idx = 0

            for dim in self.action_space.nvec:
                end_idx = start_idx + int(dim)
                logits_slice = action_logits[:, start_idx:end_idx]

                # Add temperature for better exploration
                temperature = 1.2
                logits_slice = logits_slice / temperature

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
    """Enhanced Centralized Critic with LayerNorm and attention"""

    def __init__(self, total_obs_dim: int, hidden_dim: int = 512):
        super().__init__()

        # Enhanced critic with attention for agent coordination
        self.input_layer = nn.Linear(total_obs_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)

        self.critic_network = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        if global_obs.dim() == 1:
            global_obs = global_obs.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        x = self.input_layer(global_obs)

        # Self-attention for better global state understanding
        x_reshaped = x.unsqueeze(1)
        attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        x = attn_out.squeeze(1) + x

        output = self.critic_network(x)

        if squeeze_output:
            output = output.squeeze(0)

        return output


class PrioritizedReplayBuffer:
    """Enhanced replay buffer with prioritized experience replay"""

    def __init__(self, buffer_size: int, num_agents: int, alpha: float = 0.6):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.alpha = alpha
        self.clear()

    def clear(self):
        self.observations = {f'agent_{i}': [] for i in range(self.num_agents)}
        self.global_observations = []
        self.actions = {f'agent_{i}': [] for i in range(self.num_agents)}
        self.rewards = {f'agent_{i}': [] for i in range(self.num_agents)}
        self.log_probs = {f'agent_{i}': [] for i in range(self.num_agents)}
        self.values = []
        self.dones = []
        self.priorities = []
        self.size = 0

    def store(self, obs_dict: Dict, global_obs: np.ndarray, action_dict: Dict,
              reward_dict: Dict, log_prob_dict: Dict, values: torch.Tensor, done: bool):
        agent_names = ['base_agent', 'fleet_agent', 'crew_agent', 'emissions_agent']

        for i, agent_name in enumerate(agent_names):
            self.observations[f'agent_{i}'].append(obs_dict[agent_name])
            self.actions[f'agent_{i}'].append(action_dict[agent_name])
            self.rewards[f'agent_{i}'].append(reward_dict[agent_name])
            self.log_probs[f'agent_{i}'].append(log_prob_dict[agent_name])

        self.global_observations.append(global_obs)
        self.values.append(values)
        self.dones.append(done)

        # Calculate priority based on reward magnitude
        reward_magnitude = sum(abs(reward_dict[agent]) for agent in agent_names)
        priority = (reward_magnitude + 1e-6) ** self.alpha
        self.priorities.append(priority)

        self.size += 1

    def get_batch(self) -> Dict:
        batch = {
            'observations': {},
            'global_observations': np.array(self.global_observations),
            'actions': {},
            'rewards': {},
            'log_probs': {},
            'values': torch.stack(self.values),
            'dones': np.array(self.dones),
            'priorities': np.array(self.priorities)
        }

        for i in range(self.num_agents):
            batch['observations'][f'agent_{i}'] = np.array(self.observations[f'agent_{i}'])
            batch['actions'][f'agent_{i}'] = np.array(self.actions[f'agent_{i}'])
            batch['rewards'][f'agent_{i}'] = np.array(self.rewards[f'agent_{i}'])
            batch['log_probs'][f'agent_{i}'] = torch.stack(self.log_probs[f'agent_{i}'])

        return batch


class MAPPOTrainer:
    """Enhanced Multi-Agent PPO Trainer with advanced optimization techniques"""

    def __init__(self, env, config: Dict):
        self.env = env
        self.config = config
        self.agent_names = env.agents
        self.num_agents = len(self.agent_names)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Learning rate scheduling
        self.lr_scheduler_enabled = config.get('lr_scheduling', True)
        self.current_lr_factor = 1.0

        # Initialize networks
        self._init_networks()
        self._init_optimizers()

        # Enhanced experience buffer
        self.buffer = PrioritizedReplayBuffer(config['buffer_size'], self.num_agents)

        # Training metrics
        self.training_metrics = defaultdict(list)
        self.episode_rewards = {agent: [] for agent in self.agent_names}
        self.episode_lengths = []

        # Performance tracking for adaptive training
        self.reward_window = deque(maxlen=50)
        self.best_avg_reward = float('-inf')
        self.stagnation_counter = 0

        # Create save directory
        self.save_dir = config.get('save_dir', 'mappo_models')
        os.makedirs(self.save_dir, exist_ok=True)

        print(f"Enhanced MAPPO Trainer initialized on {self.device}")

    def _init_networks(self):
        # Calculate total observation dimension for centralized critic
        test_obs, _ = self.env.reset()
        total_obs_dim = sum(len(obs) for obs in test_obs.values())

        print(f"🔍 Observation dimensions:")
        for agent, obs in test_obs.items():
            print(f"   {agent}: {len(obs)}")
        print(f"   Total: {total_obs_dim}")

        # Standard centralized critic (no adaptation needed)
        self.centralized_critic = CentralizedCritic(
            total_obs_dim=total_obs_dim,
            hidden_dim=self.config['critic_hidden_dim']
        ).to(self.device)

        # Decentralized actors
        self.actors = {}

        for agent_name in self.agent_names:
            obs_dim = len(test_obs[agent_name])
            action_space = self.env.action_spaces[agent_name]

            print(f"🤖 {agent_name}: obs_dim={obs_dim}, action_space={action_space}")

            actor = MAPPONetwork(
                obs_dim=obs_dim,
                action_space=action_space,
                hidden_dim=self.config['actor_hidden_dim'],
                is_critic=False
            ).to(self.device)

            self.actors[agent_name] = actor

    def _init_optimizers(self):
        # Use AdamW for better weight decay
        self.actor_optimizers = {}
        for agent_name in self.agent_names:
            self.actor_optimizers[agent_name] = optim.AdamW(
                self.actors[agent_name].parameters(),
                lr=self.config['actor_lr'],
                weight_decay=1e-5
            )

        self.critic_optimizer = optim.AdamW(
            self.centralized_critic.parameters(),
            lr=self.config['critic_lr'],
            weight_decay=1e-5
        )

        # Learning rate schedulers
        if self.lr_scheduler_enabled:
            self.actor_schedulers = {}
            for agent_name in self.agent_names:
                self.actor_schedulers[agent_name] = optim.lr_scheduler.CosineAnnealingLR(
                    self.actor_optimizers[agent_name], T_max=1000, eta_min=1e-6
                )

            self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.critic_optimizer, T_max=1000, eta_min=1e-6
            )

    def _adjust_difficulty(self, episode: int):
        """Curriculum learning: gradually increase environment difficulty"""
        if self.curriculum_enabled:
            # Increase max flights per episode gradually
            if episode % 100 == 0 and episode > 0:
                current_max = self.env.max_flights_per_episode
                new_max = min(current_max + 5, 50)  # Cap at 50

                if new_max != current_max:
                    print(f"🎓 Episode {episode}: Curriculum Learning - Increasing flights {current_max} -> {new_max}")
                    self.env.max_flights_per_episode = new_max

                    # IMPORTANT: Update environment spaces to reflect new dimensions
                    self.env._setup_spaces()

                    # Update our action spaces reference
                    self.action_spaces = self.env.action_spaces

                    # The adaptive critic will automatically handle the dimension change
                    print(f"📐 Environment and training spaces updated for curriculum learning")

                    # Re-initialize optimizers to include new parameters if any
                    if hasattr(self, 'critic_optimizer'):
                        self.critic_optimizer = optim.AdamW(
                            self.centralized_critic.parameters(),
                            lr=self.config['critic_lr'],
                            weight_decay=1e-5
                        )

                        if self.lr_scheduler_enabled:
                            self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                                self.critic_optimizer, T_max=1000, eta_min=1e-6
                            )

    def select_actions(self, observations: Dict) -> Tuple[Dict, Dict, torch.Tensor]:
        actions = {}
        log_probs = {}

        # Create global observation
        global_obs_list = []
        for agent in self.agent_names:
            if agent in observations:
                obs = observations[agent]
                if isinstance(obs, np.ndarray):
                    global_obs_list.append(obs.flatten())
                else:
                    global_obs_list.append(np.array(obs).flatten())

        global_obs = np.concatenate(global_obs_list)
        global_obs_tensor = torch.FloatTensor(global_obs).unsqueeze(0).to(self.device)

        # Get value from centralized critic
        with torch.no_grad():
            value = self.centralized_critic(global_obs_tensor).squeeze()

        # Get actions from each actor
        for agent_name in self.agent_names:
            if agent_name in observations:
                obs = observations[agent_name]
                if isinstance(obs, np.ndarray):
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                else:
                    obs_tensor = torch.FloatTensor(np.array(obs)).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    action, log_prob = self.actors[agent_name].get_action_and_log_prob(obs_tensor)
                    actions[agent_name] = action.squeeze(0).cpu().numpy()
                    log_probs[agent_name] = log_prob.squeeze(0)

        return actions, log_probs, value

    def compute_advantages(self, rewards: np.ndarray, values: torch.Tensor,
                           dones: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enhanced GAE with better normalization"""
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

        # Better advantage normalization
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update_networks(self):
        """Enhanced network updates with multiple techniques"""
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

        # Multiple update epochs with different techniques
        for epoch in range(self.config['ppo_epochs']):
            # Critic updates with gradient accumulation
            self._update_critic_enhanced(batch, all_returns, epoch)

            # Actor updates with adaptive clipping
            self._update_actors_enhanced(batch, all_advantages, epoch)

        # Update learning rates
        if self.lr_scheduler_enabled:
            for agent_name in self.agent_names:
                self.actor_schedulers[agent_name].step()
            self.critic_scheduler.step()

    def _update_critic_enhanced(self, batch: Dict, all_returns: Dict, epoch: int):
        """Enhanced critic update with better loss computation"""
        global_obs = torch.FloatTensor(batch['global_observations']).to(self.device)
        target_returns = torch.stack([all_returns[agent] for agent in self.agent_names]).mean(dim=0)

        # Handle potential dimension mismatches in batch
        if global_obs.dim() == 1:
            global_obs = global_obs.unsqueeze(0)

        # Multiple forward passes for better gradient estimation
        total_loss = 0
        for mini_epoch in range(2):
            try:
                values = self.centralized_critic(global_obs).view(-1)
                target_returns_flat = target_returns.view(-1)

                # Ensure dimensions match
                min_len = min(len(values), len(target_returns_flat))
                values = values[:min_len]
                target_returns_flat = target_returns_flat[:min_len]

                # Huber loss for better stability
                critic_loss = F.smooth_l1_loss(values, target_returns_flat.detach())

                # Add value clipping for stability
                if epoch > 0 and 'values' in batch:
                    old_values = batch['values'].view(-1).to(self.device)[:min_len]
                    values_clipped = old_values + torch.clamp(
                        values - old_values, -self.config['clip_ratio'], self.config['clip_ratio']
                    )
                    critic_loss_clipped = F.smooth_l1_loss(values_clipped, target_returns_flat.detach())
                    critic_loss = torch.max(critic_loss, critic_loss_clipped)

                total_loss += critic_loss

            except RuntimeError as e:
                print(f"⚠️ Critic update error: {e}")
                # Fallback to simple MSE loss
                critic_loss = F.mse_loss(values, target_returns_flat.detach())
                total_loss += critic_loss

        avg_loss = total_loss / 2

        self.critic_optimizer.zero_grad()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.centralized_critic.parameters(), self.config['max_grad_norm'])
        self.critic_optimizer.step()

        self.training_metrics['critic_loss'].append(avg_loss.item())

    def _update_actors_enhanced(self, batch: Dict, all_advantages: Dict, epoch: int):
        """Enhanced actor updates with adaptive techniques"""
        for i, agent_name in enumerate(self.agent_names):
            obs = torch.FloatTensor(batch['observations'][f'agent_{i}']).to(self.device)
            actions = torch.LongTensor(batch['actions'][f'agent_{i}']).to(self.device)
            old_log_probs = batch['log_probs'][f'agent_{i}'].to(self.device)
            advantages = all_advantages[agent_name].to(self.device)

            # Enhanced advantage normalization per agent
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Get new action probabilities
            new_action_logits = self.actors[agent_name](obs)
            new_log_probs = self._compute_log_probs(new_action_logits, actions, agent_name)

            # Enhanced entropy computation
            entropy = self._compute_entropy_enhanced(new_action_logits, agent_name)

            # Adaptive clipping based on training progress
            current_clip_ratio = self.config['clip_ratio']
            if epoch > self.config['ppo_epochs'] // 2:
                current_clip_ratio *= 0.8  # Reduce clipping in later epochs

            # Compute ratio and PPO loss
            ratio = torch.exp(new_log_probs - old_log_probs)

            # KL divergence penalty for stability
            kl_div = (old_log_probs - new_log_probs).mean()
            kl_penalty = 0.01 * torch.clamp(kl_div - 0.02, 0, float('inf')) ** 2

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - current_clip_ratio, 1 + current_clip_ratio) * advantages

            # Enhanced loss with multiple terms
            entropy_bonus = self.config.get('entropy_coef', 0.02) * entropy.mean()
            actor_loss = -torch.min(surr1, surr2).mean() - entropy_bonus + kl_penalty

            # Update actor
            self.actor_optimizers[agent_name].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent_name].parameters(), self.config['max_grad_norm'])
            self.actor_optimizers[agent_name].step()

            # Log metrics
            self.training_metrics[f'{agent_name}_actor_loss'].append(actor_loss.item())
            self.training_metrics[f'{agent_name}_entropy'].append(entropy.mean().item())
            self.training_metrics[f'{agent_name}_kl_div'].append(kl_div.item())

    def _compute_entropy_enhanced(self, action_logits: torch.Tensor, agent_name: str) -> torch.Tensor:
        """Enhanced entropy computation for better exploration"""
        action_space = self.env.action_spaces[agent_name]

        from gymnasium import spaces
        if isinstance(action_space, spaces.MultiBinary):
            probs = torch.sigmoid(action_logits)
            entropy = -(probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8))
            return entropy.sum(dim=-1)
        elif isinstance(action_space, spaces.MultiDiscrete):
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
        """Enhanced log probability computation"""
        action_space = self.env.action_spaces[agent_name]

        from gymnasium import spaces
        if isinstance(action_space, spaces.MultiBinary):
            dist = Bernoulli(action_logits)
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
        """Enhanced training loop without curriculum learning"""
        print(f"Starting Enhanced MAPPO training for {num_episodes} episodes...")

        for episode in range(num_episodes):
            episode_rewards = {agent: 0 for agent in self.agent_names}
            episode_length = 0

            observations, _ = self.env.reset()

            while True:
                actions, log_probs, value = self.select_actions(observations)
                next_observations, rewards, terminated, truncated, info = self.env.step(actions)

                global_obs_list = []
                for agent in self.agent_names:
                    if agent in observations:
                        obs = observations[agent]
                        if isinstance(obs, np.ndarray):
                            global_obs_list.append(obs.flatten())
                        else:
                            global_obs_list.append(np.array(obs).flatten())

                global_obs = np.concatenate(global_obs_list)
                done = any(terminated.values()) or any(truncated.values())

                self.buffer.store(
                    observations, global_obs, actions, rewards, log_probs, value, done
                )

                for agent in self.agent_names:
                    episode_rewards[agent] += rewards[agent]
                episode_length += 1

                if done:
                    break

                observations = next_observations

            # Store episode metrics
            for agent in self.agent_names:
                self.episode_rewards[agent].append(episode_rewards[agent])
            self.episode_lengths.append(episode_length)

            # Performance tracking
            avg_reward = np.mean([episode_rewards[agent] for agent in self.agent_names])
            self.reward_window.append(avg_reward)

            # Adaptive training frequency
            update_freq = self.config['update_freq']
            if len(self.reward_window) > 20:
                recent_improvement = np.mean(list(self.reward_window)[-10:]) - np.mean(
                    list(self.reward_window)[-20:-10])
                if recent_improvement < 0.01:  # Slow improvement
                    update_freq = max(3, update_freq // 2)  # Update more frequently
                else:
                    update_freq = self.config['update_freq']

            # Update networks
            if (episode + 1) % update_freq == 0:
                self.update_networks()
                self.buffer.clear()

            # Early stopping and model checkpointing
            if len(self.reward_window) >= 50:
                current_avg = np.mean(self.reward_window)
                if current_avg > self.best_avg_reward:
                    self.best_avg_reward = current_avg
                    self.stagnation_counter = 0
                    # Save best model
                    self.save_models(episode + 1, best=True)
                else:
                    self.stagnation_counter += 1

            # Logging
            if (episode + 1) % self.config['log_freq'] == 0:
                self._log_progress_enhanced(episode + 1)

            # Regular saves
            if (episode + 1) % self.config['save_freq'] == 0:
                self.save_models(episode + 1)

        print("Enhanced training completed!")
        self.save_models(num_episodes, final=True)

    def _log_progress_enhanced(self, episode: int):
        """Enhanced logging with more detailed metrics"""
        recent_episodes = min(self.config['log_freq'], len(self.episode_rewards[self.agent_names[0]]))

        print(f"\n🚀 Episode {episode}")
        print("-" * 50)

        total_rewards = []
        for agent in self.agent_names:
            recent_rewards = self.episode_rewards[agent][-recent_episodes:]
            avg_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards)
            total_rewards.extend(recent_rewards)
            print(f"{agent}: {avg_reward:.3f} ± {std_reward:.3f}")

        avg_length = np.mean(self.episode_lengths[-recent_episodes:])
        total_avg_reward = np.mean(total_rewards)

        print(f"📊 Avg episode length: {avg_length:.1f}")
        print(f"🎯 Total avg reward: {total_avg_reward:.3f}")
        print(f"🏆 Best avg reward: {self.best_avg_reward:.3f}")

        # Learning metrics
        if self.training_metrics['critic_loss']:
            recent_critic_loss = np.mean(self.training_metrics['critic_loss'][-10:])
            print(f"📉 Critic loss: {recent_critic_loss:.6f}")

        # Exploration metrics
        for agent in self.agent_names:
            entropy_key = f'{agent}_entropy'
            if entropy_key in self.training_metrics and self.training_metrics[entropy_key]:
                recent_entropy = np.mean(self.training_metrics[entropy_key][-10:])
                print(f"🔄 {agent} entropy: {recent_entropy:.4f}")

        print(f"⚡ Stagnation counter: {self.stagnation_counter}")

    def save_models(self, episode: int, final: bool = False, best: bool = False):
        """Enhanced model saving with best model tracking"""
        if best:
            suffix = 'best'
        elif final:
            suffix = 'final'
        else:
            suffix = f'episode_{episode}'

        for agent_name in self.agent_names:
            torch.save(
                self.actors[agent_name].state_dict(),
                os.path.join(self.save_dir, f'{agent_name}_actor_{suffix}.pth')
            )

        torch.save(
            self.centralized_critic.state_dict(),
            os.path.join(self.save_dir, f'centralized_critic_{suffix}.pth')
        )

        with open(os.path.join(self.save_dir, f'training_metrics_{suffix}.pkl'), 'wb') as f:
            pickle.dump({
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'training_metrics': dict(self.training_metrics),
                'best_avg_reward': self.best_avg_reward
            }, f)

        if best:
            print(f"💾 Best model saved! Reward: {self.best_avg_reward:.3f}")
        else:
            print(f"💾 Models saved: {suffix}")

    def plot_training_curves(self, save_path: str = None):
        """Enhanced plotting with more detailed metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Enhanced MAPPO Training Results - Airline Scheduling', fontsize=16, fontweight='bold')

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        agent_colors = dict(zip(self.agent_names, colors))

        # Plot 1: Episode Rewards with trend lines
        ax1 = axes[0, 0]
        for agent in self.agent_names:
            episodes = range(1, len(self.episode_rewards[agent]) + 1)
            rewards = self.episode_rewards[agent]

            ax1.plot(episodes, rewards, alpha=0.3, color=agent_colors[agent], linewidth=0.5)

            if len(rewards) >= 20:
                moving_avg = np.convolve(rewards, np.ones(20) / 20, mode='valid')
                moving_episodes = range(20, len(rewards) + 1)
                ax1.plot(moving_episodes, moving_avg, label=agent,
                         color=agent_colors[agent], linewidth=2)

        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Episode Reward')
        ax1.set_title('Episode Rewards (20-episode moving average)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Learning Curves Comparison
        ax2 = axes[0, 1]
        if len(self.reward_window) > 0:
            window_episodes = range(len(self.episode_rewards[self.agent_names[0]]) - len(self.reward_window) + 1,
                                    len(self.episode_rewards[self.agent_names[0]]) + 1)
            ax2.plot(window_episodes, self.reward_window, 'b-', linewidth=2, label='Average Reward')
            ax2.axhline(y=self.best_avg_reward, color='r', linestyle='--',
                        label=f'Best Avg: {self.best_avg_reward:.3f}')

        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Reward')
        ax2.set_title('Learning Progress (50-episode window)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Training Losses
        ax3 = axes[0, 2]
        if self.training_metrics['critic_loss']:
            episodes_loss = np.linspace(1, len(self.episode_rewards[self.agent_names[0]]),
                                        len(self.training_metrics['critic_loss']))
            ax3.plot(episodes_loss, self.training_metrics['critic_loss'], 'r-', linewidth=1, label='Critic Loss')

            # Add smoothed loss
            if len(self.training_metrics['critic_loss']) > 50:
                smoothed_loss = np.convolve(self.training_metrics['critic_loss'], np.ones(50) / 50, mode='valid')
                smoothed_episodes = np.linspace(50, len(self.episode_rewards[self.agent_names[0]]),
                                                len(smoothed_loss))
                ax3.plot(smoothed_episodes, smoothed_loss, 'darkred', linewidth=2, label='Smoothed')

        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Loss')
        ax3.set_title('Critic Loss Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')

        # Plot 4: Exploration Metrics (Entropy)
        ax4 = axes[1, 0]
        for agent in self.agent_names:
            entropy_key = f'{agent}_entropy'
            if entropy_key in self.training_metrics and self.training_metrics[entropy_key]:
                episodes_entropy = np.linspace(1, len(self.episode_rewards[self.agent_names[0]]),
                                               len(self.training_metrics[entropy_key]))
                ax4.plot(episodes_entropy, self.training_metrics[entropy_key],
                         color=agent_colors[agent], alpha=0.7, label=agent)

        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Entropy')
        ax4.set_title('Exploration (Entropy) Over Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Plot 5: Episode Lengths
        ax5 = axes[1, 1]
        episodes = range(1, len(self.episode_lengths) + 1)
        ax5.plot(episodes, self.episode_lengths, alpha=0.6, color='purple', linewidth=1)

        if len(self.episode_lengths) >= 20:
            moving_avg_lengths = np.convolve(self.episode_lengths, np.ones(20) / 20, mode='valid')
            moving_episodes = range(20, len(self.episode_lengths) + 1)
            ax5.plot(moving_episodes, moving_avg_lengths, color='darkblue', linewidth=2, label='20-ep avg')
            ax5.legend()

        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Episode Length')
        ax5.set_title('Episode Lengths Over Time')
        ax5.grid(True, alpha=0.3)

        # Plot 6: Performance Summary
        ax6 = axes[1, 2]
        if len(self.episode_rewards[self.agent_names[0]]) >= 100:
            # Last 100 episodes performance
            recent_rewards = []
            agent_names_clean = []

            for agent in self.agent_names:
                last_100 = self.episode_rewards[agent][-100:]
                recent_rewards.append(np.mean(last_100))
                agent_names_clean.append(agent.replace('_', ' ').title())

            bars = ax6.bar(agent_names_clean, recent_rewards,
                           color=[agent_colors[agent] for agent in self.agent_names])

            for bar, reward in zip(bars, recent_rewards):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{reward:.2f}', ha='center', va='bottom')

        ax6.set_ylabel('Average Reward')
        ax6.set_title('Performance Summary (Last 100 episodes)')
        ax6.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Enhanced training plots saved to: {save_path}")

        plt.show()


def get_enhanced_training_config():
    """Enhanced training configuration without curriculum learning"""
    return {
        # Network architecture - Keep original sizes but with enhancements
        'actor_hidden_dim': 256,
        'critic_hidden_dim': 512,

        # Enhanced training hyperparameters
        'actor_lr': 3e-4,
        'critic_lr': 1e-3,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_ratio': 0.2,
        'ppo_epochs': 10,
        'max_grad_norm': 0.5,

        # Experience collection
        'buffer_size': 2048,
        'update_freq': 10,

        # Enhancement features
        'lr_scheduling': True,
        'entropy_coef': 0.02,

        # Logging and saving
        'log_freq': 10,
        'save_freq': 50,
        'save_dir': '02_mappo_airline_models'
    }


def main():
    """Enhanced main function without curriculum learning"""
    print("🚀 Initializing Enhanced Airline Scheduling MAPPO Training...")

    # Initialize environment with fixed parameters (no curriculum)
    env = make_airline_env(
        flights_csv='flights_data.csv',
        aircraft_csv='aircraft_data.csv',
        crew_csv='crew_data.csv',
        action_space_type="discrete",
        max_flights_per_episode=20,  # Fixed size
        episode_length=50,
        weekly_budget=50000,
        max_emissions=5000
    )

    print(f"✅ Environment loaded with {env.num_flights} flights")
    print(f"🤖 Agents: {env.agents}")
    print(f"📊 Fixed max flights per episode: {env.max_flights_per_episode}")

    # Get enhanced training configuration
    config = get_enhanced_training_config()
    print(f"⚙️ Training configuration loaded with enhancements:")
    print(f"   - Learning rate scheduling: {config['lr_scheduling']}")
    print(f"   - Enhanced exploration: {config['entropy_coef']}")
    print(f"   - No curriculum learning (fixed difficulty)")

    # Initialize enhanced trainer
    trainer = MAPPOTrainer(env, config)

    # Train the model
    num_episodes = 1000
    print(f"🎯 Starting training for {num_episodes} episodes...")
    trainer.train(num_episodes)

    # Plot enhanced results
    plot_save_path = os.path.join(config['save_dir'], 'enhanced_training_curves.png')
    trainer.plot_training_curves(save_path=plot_save_path)

    print(f"\n🎉 Enhanced training completed!")
    print(f"💾 Models saved in: {config['save_dir']}")
    print(f"📊 Training plots saved to: {plot_save_path}")
    print(f"🏆 Best average reward achieved: {trainer.best_avg_reward:.3f}")

if __name__ == "__main__":
    main()