import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli
import matplotlib.pyplot as plt
from collections import defaultdict
import os
from typing import Dict, List, Tuple

from environment import make_airline_env


class SimpleMAPPONetwork(nn.Module):
    """Simple MAPPO Network with stable training"""

    def __init__(self, obs_dim: int, action_space, hidden_dim: int = 256, is_critic: bool = False):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_space = action_space
        self.is_critic = is_critic

        # Simple network with proper initialization
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        if is_critic:
            self.value_head = nn.Linear(hidden_dim // 2, 1)
        else:
            self._init_actor_head()

        # Proper weight initialization to prevent NaN
        self._init_weights()

    def _init_weights(self):
        """Stable weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization for better stability
                nn.init.xavier_uniform_(module.weight, gain=0.5)  # Smaller gain for stability
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _init_actor_head(self):
        from gymnasium import spaces

        if isinstance(self.action_space, spaces.MultiBinary):
            self.action_head = nn.Linear(self.network[-2].out_features, self.action_space.n)
            # Initialize action head with smaller values
            nn.init.xavier_uniform_(self.action_head.weight, gain=0.1)
            nn.init.constant_(self.action_head.bias, 0.0)
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            self.action_heads = nn.ModuleList([
                nn.Linear(self.network[-2].out_features, int(dim))
                for dim in self.action_space.nvec
            ])
            for head in self.action_heads:
                nn.init.xavier_uniform_(head.weight, gain=0.1)
                nn.init.constant_(head.bias, 0.0)

    def forward(self, obs):
        # Add input normalization to prevent exploding values
        obs = torch.clamp(obs, -10, 10)  # Clamp input

        features = self.network(obs)

        if self.is_critic:
            return self.value_head(features)
        else:
            return self._get_action_logits(features)

    def _get_action_logits(self, features):
        from gymnasium import spaces

        if isinstance(self.action_space, spaces.MultiBinary):
            logits = self.action_head(features)
            # Clamp logits to prevent extreme values
            logits = torch.clamp(logits, -5, 5)
            return torch.sigmoid(logits)
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            logits = []
            for head in self.action_heads:
                head_logits = head(features)
                # Clamp logits to prevent extreme values
                head_logits = torch.clamp(head_logits, -5, 5)
                logits.append(F.log_softmax(head_logits, dim=-1))
            return torch.cat(logits, dim=-1)

    def get_action_and_log_prob(self, obs):
        action_logits = self.forward(obs)

        # Check for NaN values and handle them
        if torch.isnan(action_logits).any():
            print("⚠️ NaN detected in action logits, using random actions")
            # Fallback to random actions if NaN detected
            from gymnasium import spaces
            if isinstance(self.action_space, spaces.MultiBinary):
                action = torch.randint(0, 2, (obs.shape[0], self.action_space.n), dtype=torch.float)
                log_prob = torch.log(torch.tensor(0.5)) * self.action_space.n
                return action, log_prob
            elif isinstance(self.action_space, spaces.MultiDiscrete):
                actions = []
                log_probs = []
                for dim in self.action_space.nvec:
                    action_slice = torch.randint(0, int(dim), (obs.shape[0],))
                    log_prob_slice = torch.log(torch.tensor(1.0 / dim))
                    actions.append(action_slice)
                    log_probs.append(log_prob_slice)
                action = torch.stack(actions, dim=-1)
                log_prob = torch.stack(log_probs).sum()
                return action, log_prob

        from gymnasium import spaces
        if isinstance(self.action_space, spaces.MultiBinary):
            # More conservative exploration to prevent NaN
            exploration_noise = 0.1  # Reduced from 0.3
            noise = exploration_noise * torch.randn_like(action_logits)
            probs = torch.clamp(action_logits + noise, 0.01, 0.99)  # Stricter bounds

            # Additional safety check
            probs = torch.where(torch.isnan(probs), torch.ones_like(probs) * 0.5, probs)

            dist = Bernoulli(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        elif isinstance(self.action_space, spaces.MultiDiscrete):
            actions = []
            log_probs = []
            start_idx = 0

            for dim in self.action_space.nvec:
                end_idx = start_idx + int(dim)
                logits_slice = action_logits[:, start_idx:end_idx].clone()

                # More conservative exploration
                temperature = 1.5  # Reduced from 3.0
                logits_slice = logits_slice / temperature

                # Smaller uniform noise
                uniform_noise = torch.ones_like(logits_slice) * 0.1  # Reduced from 0.5
                logits_slice = logits_slice + uniform_noise

                # Additional safety check
                logits_slice = torch.where(torch.isnan(logits_slice),
                                           torch.zeros_like(logits_slice), logits_slice)

                dist = Categorical(logits=logits_slice)
                action_slice = dist.sample()
                log_prob_slice = dist.log_prob(action_slice)

                actions.append(action_slice)
                log_probs.append(log_prob_slice)
                start_idx = end_idx

            action = torch.stack(actions, dim=-1)
            log_prob = torch.stack(log_probs, dim=-1).sum(dim=-1)

        return action, log_prob


class SimpleMAPPOTrainer:
    """Simple MAPPO Trainer with WORKING plots"""

    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.agent_names = env.agents
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize networks
        self._init_networks()
        self._init_optimizers()

        # Training metrics - SIMPLE LISTS
        self.episode_rewards = {agent: [] for agent in self.agent_names}
        self.critic_losses = []
        self.entropies = {agent: [] for agent in self.agent_names}
        self.episode_lengths = []

        # Create save directory
        self.save_dir = config.get('save_dir', 'simple_mappo_models')
        os.makedirs(self.save_dir, exist_ok=True)

        print(f"🚀 Simple MAPPO Trainer initialized!")

    def _init_networks(self):
        # Get dimensions
        test_obs, _ = self.env.reset()
        total_obs_dim = sum(len(obs) for obs in test_obs.values())

        # Centralized critic
        self.critic = SimpleMAPPONetwork(
            obs_dim=total_obs_dim,
            action_space=None,
            hidden_dim=256,
            is_critic=True
        ).to(self.device)

        # Actors
        self.actors = {}
        for agent_name in self.agent_names:
            obs_dim = len(test_obs[agent_name])
            action_space = self.env.action_spaces[agent_name]

            actor = SimpleMAPPONetwork(
                obs_dim=obs_dim,
                action_space=action_space,
                hidden_dim=256,
                is_critic=False
            ).to(self.device)

            self.actors[agent_name] = actor

    def _init_optimizers(self):
        # More conservative learning rates to prevent NaN
        self.actor_optimizers = {}
        for agent_name in self.agent_names:
            self.actor_optimizers[agent_name] = optim.Adam(
                self.actors[agent_name].parameters(),
                lr=5e-4,  # Reduced from 1e-3
                eps=1e-8,
                weight_decay=1e-6
            )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=1e-3,  # Reduced from 3e-3
            eps=1e-8,
            weight_decay=1e-6
        )

    def select_actions(self, observations):
        actions = {}
        log_probs = {}

        # Robust global observation creation
        try:
            global_obs_list = []
            for agent in self.agent_names:
                obs = observations[agent]
                # Ensure observation is a numpy array and handle edge cases
                if isinstance(obs, np.ndarray):
                    obs_clean = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
                else:
                    obs_clean = np.array(obs, dtype=np.float32)
                    obs_clean = np.nan_to_num(obs_clean, nan=0.0, posinf=1.0, neginf=-1.0)

                global_obs_list.append(obs_clean)

            global_obs = np.concatenate(global_obs_list)
            global_obs = np.clip(global_obs, -10, 10)  # Clip to reasonable range
            global_obs_tensor = torch.FloatTensor(global_obs).unsqueeze(0).to(self.device)

            # Get value from critic with error handling
            with torch.no_grad():
                try:
                    value = self.critic(global_obs_tensor).squeeze()
                    if torch.isnan(value).any() or torch.isinf(value).any():
                        print("⚠️ NaN/Inf in critic value, using zero")
                        value = torch.tensor(0.0).to(self.device)
                except Exception as e:
                    print(f"⚠️ Error in critic forward pass: {e}")
                    value = torch.tensor(0.0).to(self.device)

        except Exception as e:
            print(f"⚠️ Error creating global observation: {e}")
            value = torch.tensor(0.0).to(self.device)

        # Get actions from each actor with robust error handling
        for agent_name in self.agent_names:
            try:
                obs = observations[agent_name]

                # Clean and validate observation
                if isinstance(obs, np.ndarray):
                    obs_clean = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
                else:
                    obs_clean = np.array(obs, dtype=np.float32)
                    obs_clean = np.nan_to_num(obs_clean, nan=0.0, posinf=1.0, neginf=-1.0)

                obs_clean = np.clip(obs_clean, -10, 10)
                obs_tensor = torch.FloatTensor(obs_clean).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    action, log_prob = self.actors[agent_name].get_action_and_log_prob(obs_tensor)

                    # Validate action and log_prob
                    if torch.isnan(action).any() or torch.isnan(log_prob).any():
                        print(f"⚠️ NaN in {agent_name} action/log_prob, using random fallback")
                        # Fallback to random action
                        from gymnasium import spaces
                        if isinstance(self.env.action_spaces[agent_name], spaces.MultiBinary):
                            action = torch.randint(0, 2, action.shape, dtype=torch.float)
                            log_prob = torch.log(torch.tensor(0.5)) * action.numel()
                        elif isinstance(self.env.action_spaces[agent_name], spaces.MultiDiscrete):
                            random_actions = []
                            for dim in self.env.action_spaces[agent_name].nvec:
                                random_actions.append(torch.randint(0, int(dim), (1,)))
                            action = torch.stack(random_actions, dim=-1)
                            log_prob = torch.tensor(0.0)

                    actions[agent_name] = action.squeeze(0).cpu().numpy()
                    log_probs[agent_name] = log_prob.squeeze(0) if log_prob.dim() > 0 else log_prob

            except Exception as e:
                print(f"⚠️ Error in {agent_name} action selection: {e}")
                # Create fallback random action
                from gymnasium import spaces
                if isinstance(self.env.action_spaces[agent_name], spaces.MultiBinary):
                    actions[agent_name] = np.random.randint(0, 2, self.env.action_spaces[agent_name].n)
                    log_probs[agent_name] = torch.tensor(0.0)
                elif isinstance(self.env.action_spaces[agent_name], spaces.MultiDiscrete):
                    random_actions = []
                    for dim in self.env.action_spaces[agent_name].nvec:
                        random_actions.append(np.random.randint(0, int(dim)))
                    actions[agent_name] = np.array(random_actions)
                    log_probs[agent_name] = torch.tensor(0.0)

        return actions, log_probs, value

    def update_networks(self, batch):
        """Stable network updates with NaN checking"""

        # Check for NaN in batch data
        for key, value in batch.items():
            if torch.isnan(value).any():
                print(f"⚠️ NaN detected in batch[{key}], skipping update")
                return

        # Update critic with gradient clipping
        for _ in range(3):  # Reduced from 5
            values = self.critic(batch['global_obs']).squeeze()
            target_values = batch['returns'].detach()

            # Check for NaN in values
            if torch.isnan(values).any():
                print("⚠️ NaN in critic values, skipping critic update")
                break

            critic_loss = F.mse_loss(values, target_values)

            # Check for NaN in loss
            if torch.isnan(critic_loss).any():
                print("⚠️ NaN in critic loss, skipping critic update")
                break

            self.critic_optimizer.zero_grad()
            critic_loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)

            self.critic_optimizer.step()

        # Store critic loss (with NaN check)
        if not torch.isnan(critic_loss).any():
            self.critic_losses.append(critic_loss.detach().item())

        # Update actors with gradient clipping
        for agent_name in self.agent_names:
            obs = batch[f'{agent_name}_obs']
            actions = batch[f'{agent_name}_actions']
            old_log_probs = batch[f'{agent_name}_log_probs'].detach()
            advantages = batch[f'{agent_name}_advantages'].detach()

            # Check for NaN in advantages
            if torch.isnan(advantages).any():
                print(f"⚠️ NaN in {agent_name} advantages, skipping actor update")
                continue

            # Reduced actor update epochs to prevent instability
            for epoch in range(2):  # Reduced from 3
                # Fresh forward pass each time
                action_logits = self.actors[agent_name](obs)

                # Check for NaN in action logits
                if torch.isnan(action_logits).any():
                    print(f"⚠️ NaN in {agent_name} action logits, skipping actor update")
                    break

                new_log_probs = self._compute_log_probs(action_logits, actions, agent_name)

                # Check for NaN in log probs
                if torch.isnan(new_log_probs).any():
                    print(f"⚠️ NaN in {agent_name} log probs, skipping actor update")
                    break

                # Compute entropy
                entropy = self._compute_entropy(action_logits, agent_name)

                # Check for NaN in entropy
                if torch.isnan(entropy).any():
                    print(f"⚠️ NaN in {agent_name} entropy, using zero entropy")
                    entropy = torch.zeros_like(entropy)

                # PPO loss with moderate entropy bonus
                ratio = torch.exp(new_log_probs - old_log_probs)

                # Clamp ratio to prevent extreme values
                ratio = torch.clamp(ratio, 0.1, 10.0)

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages

                # Moderate entropy bonus (reduced from 0.2)
                entropy_bonus = 0.05 * entropy.mean()
                actor_loss = -torch.min(surr1, surr2).mean() - entropy_bonus

                # Check for NaN in actor loss
                if torch.isnan(actor_loss).any():
                    print(f"⚠️ NaN in {agent_name} actor loss, skipping actor update")
                    break

                # Update actor with gradient clipping
                self.actor_optimizers[agent_name].zero_grad()
                actor_loss.backward()

                # Strong gradient clipping to prevent NaN
                torch.nn.utils.clip_grad_norm_(self.actors[agent_name].parameters(), max_norm=0.5)

                self.actor_optimizers[agent_name].step()

            # Store entropy (from last epoch, with NaN check)
            try:
                with torch.no_grad():
                    final_logits = self.actors[agent_name](obs)
                    if not torch.isnan(final_logits).any():
                        final_entropy = self._compute_entropy(final_logits, agent_name)
                        if not torch.isnan(final_entropy).any():
                            self.entropies[agent_name].append(final_entropy.mean().item())
                        else:
                            self.entropies[agent_name].append(0.0)  # Fallback
                    else:
                        self.entropies[agent_name].append(0.0)  # Fallback
            except:
                self.entropies[agent_name].append(0.0)  # Fallback

    def _compute_log_probs(self, action_logits, actions, agent_name):
        """Compute log probabilities"""
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

    def _compute_entropy(self, action_logits, agent_name):
        """Compute entropy"""
        action_space = self.env.action_spaces[agent_name]

        from gymnasium import spaces
        if isinstance(action_space, spaces.MultiBinary):
            probs = action_logits
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

    def train(self, num_episodes):
        """Simple training loop with GUARANTEED PLOTS"""
        print(f"🚀 Starting Simple MAPPO training for {num_episodes} episodes...")

        for episode in range(num_episodes):
            episode_rewards = {agent: 0 for agent in self.agent_names}
            episode_data = {
                'observations': {agent: [] for agent in self.agent_names},
                'global_obs': [],
                'actions': {agent: [] for agent in self.agent_names},
                'rewards': {agent: [] for agent in self.agent_names},
                'log_probs': {agent: [] for agent in self.agent_names},
                'values': [],
                'dones': []
            }

            # Run episode
            observations, _ = self.env.reset()
            episode_length = 0

            while True:
                # Select actions
                actions, log_probs, value = self.select_actions(observations)

                # Environment step
                next_observations, rewards, terminated, truncated, _ = self.env.step(actions)

                # Store data
                global_obs = np.concatenate([observations[agent] for agent in self.agent_names])
                done = any(terminated.values()) or any(truncated.values())

                episode_data['global_obs'].append(global_obs)
                episode_data['values'].append(value)
                episode_data['dones'].append(done)

                for agent in self.agent_names:
                    episode_data['observations'][agent].append(observations[agent])
                    episode_data['actions'][agent].append(actions[agent])
                    episode_data['rewards'][agent].append(rewards[agent])
                    episode_data['log_probs'][agent].append(log_probs[agent])

                    episode_rewards[agent] += rewards[agent]

                episode_length += 1

                if done:
                    break

                observations = next_observations

            # Store episode metrics
            for agent in self.agent_names:
                self.episode_rewards[agent].append(episode_rewards[agent])
            self.episode_lengths.append(episode_length)

            # Update networks every few episodes
            if (episode + 1) % 5 == 0 and len(episode_data['dones']) > 0:
                batch = self._prepare_batch(episode_data)
                self.update_networks(batch)

            # FORCE PLOT GENERATION every 50 episodes
            if (episode + 1) % 50 == 0:
                print(f"\n📊 Episode {episode + 1} - GENERATING PLOTS...")
                self.plot_results(episode + 1)
                self._log_progress(episode + 1)

        # Final plot
        print("\n🎨 GENERATING FINAL PLOTS...")
        self.plot_results(num_episodes, final=True)

    def _prepare_batch(self, episode_data):
        """Prepare batch for training - ROBUST advantage calculation"""
        batch = {}

        # Convert to tensors
        batch['global_obs'] = torch.FloatTensor(np.array(episode_data['global_obs'])).to(self.device)
        values = torch.stack(episode_data['values']).to(self.device)
        dones = torch.FloatTensor(episode_data['dones']).to(self.device)

        # Check for NaN in values from the start
        if torch.isnan(values).any():
            print("⚠️ NaN detected in values, using zero values")
            values = torch.zeros_like(values)

        # Compute returns and advantages for each agent with robust calculation
        all_returns = []

        for agent in self.agent_names:
            rewards = torch.FloatTensor(episode_data['rewards'][agent]).to(self.device)

            # Check for NaN/inf in rewards
            if torch.isnan(rewards).any() or torch.isinf(rewards).any():
                print(f"⚠️ NaN/Inf in {agent} rewards, using zero rewards")
                rewards = torch.zeros_like(rewards)

            # Clamp rewards to reasonable range
            rewards = torch.clamp(rewards, -100, 100)

            # ROBUST returns calculation with stability checks
            returns = torch.zeros_like(rewards)
            running_return = 0.0

            for t in reversed(range(len(rewards))):
                if dones[t]:
                    running_return = 0.0
                running_return = rewards[t] + 0.99 * running_return

                # Check for overflow/underflow
                if abs(running_return) > 1000:
                    running_return = torch.clamp(torch.tensor(running_return), -1000, 1000).item()

                returns[t] = running_return

            # Additional safety check for returns
            if torch.isnan(returns).any() or torch.isinf(returns).any():
                print(f"⚠️ NaN/Inf in {agent} returns, using rewards as returns")
                returns = rewards.clone()

            all_returns.append(returns)

            # ROBUST advantages calculation
            try:
                # Ensure values and returns have compatible shapes
                if len(values) != len(returns):
                    min_len = min(len(values), len(returns))
                    values_trimmed = values[:min_len]
                    returns_trimmed = returns[:min_len]
                else:
                    values_trimmed = values
                    returns_trimmed = returns

                advantages = returns_trimmed - values_trimmed

                # Check for NaN/inf in raw advantages
                if torch.isnan(advantages).any() or torch.isinf(advantages).any():
                    print(f"⚠️ NaN/Inf in raw {agent} advantages, using zero advantages")
                    advantages = torch.zeros_like(advantages)
                else:
                    # Robust normalization
                    adv_mean = advantages.mean()
                    adv_std = advantages.std()

                    # Check if std is too small or NaN
                    if torch.isnan(adv_mean) or torch.isnan(adv_std) or adv_std < 1e-8:
                        print(f"⚠️ Invalid {agent} advantage statistics, using unnormalized advantages")
                        advantages = torch.clamp(advantages, -10, 10)
                    else:
                        # Safe normalization
                        advantages = (advantages - adv_mean) / (adv_std + 1e-8)
                        # Clamp normalized advantages to prevent extreme values
                        advantages = torch.clamp(advantages, -5, 5)

            except Exception as e:
                print(f"⚠️ Error computing {agent} advantages: {e}, using zero advantages")
                advantages = torch.zeros_like(returns)

            # Final safety check
            if torch.isnan(advantages).any() or torch.isinf(advantages).any():
                print(f"⚠️ Final check: NaN/Inf in {agent} advantages, using zero advantages")
                advantages = torch.zeros_like(advantages)

            # Store in batch - convert lists to tensors properly
            try:
                obs_array = np.array(episode_data['observations'][agent])
                actions_array = np.array(episode_data['actions'][agent])

                # Ensure arrays have expected dimensions
                if len(obs_array.shape) == 1:
                    obs_array = obs_array.reshape(1, -1)
                if len(actions_array.shape) == 1:
                    actions_array = actions_array.reshape(-1, 1) if len(actions_array) > 0 else actions_array.reshape(0,
                                                                                                                      1)

                batch[f'{agent}_obs'] = torch.FloatTensor(obs_array).to(self.device)
                batch[f'{agent}_actions'] = torch.LongTensor(actions_array).to(self.device)
                batch[f'{agent}_log_probs'] = torch.stack(episode_data['log_probs'][agent]).to(self.device)
                batch[f'{agent}_advantages'] = advantages.clone().detach()  # Clone to avoid sharing

            except Exception as e:
                print(f"⚠️ Error creating {agent} batch tensors: {e}")
                # Create dummy tensors if there's an error
                batch[f'{agent}_obs'] = torch.zeros(1, self.actors[agent].obs_dim).to(self.device)
                batch[f'{agent}_actions'] = torch.zeros(1, 1, dtype=torch.long).to(self.device)
                batch[f'{agent}_log_probs'] = torch.zeros(1).to(self.device)
                batch[f'{agent}_advantages'] = torch.zeros(1).to(self.device)

        # Use average returns for critic (with safety checks)
        try:
            if all_returns and len(all_returns) > 0:
                # Ensure all returns have the same length
                min_len = min(len(ret) for ret in all_returns)
                trimmed_returns = [ret[:min_len] for ret in all_returns]
                batch['returns'] = torch.stack(trimmed_returns).mean(dim=0).clone().detach()
            else:
                batch['returns'] = torch.zeros(1).to(self.device)

            # Final check for returns
            if torch.isnan(batch['returns']).any() or torch.isinf(batch['returns']).any():
                print("⚠️ NaN/Inf in final returns, using zero returns")
                batch['returns'] = torch.zeros_like(batch['returns'])

        except Exception as e:
            print(f"⚠️ Error computing final returns: {e}")
            batch['returns'] = torch.zeros(1).to(self.device)

        return batch

    def plot_results(self, episode, final=False):
        """GUARANTEED WORKING PLOTS"""
        print(f"🎨 Creating plots for episode {episode}...")

        try:
            # Set up the plot
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'🔥 AGGRESSIVE MAPPO Training - Episode {episode}', fontsize=14, fontweight='bold')

            colors = ['red', 'blue', 'green', 'orange']

            # Plot 1: Episode Rewards
            ax1 = axes[0, 0]
            for i, agent in enumerate(self.agent_names):
                if len(self.episode_rewards[agent]) > 0:
                    episodes = list(range(1, len(self.episode_rewards[agent]) + 1))
                    rewards = self.episode_rewards[agent]
                    ax1.plot(episodes, rewards, color=colors[i], linewidth=2, label=agent)

                    # Moving average if enough data
                    if len(rewards) > 10:
                        window = min(10, len(rewards) // 2)
                        moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
                        mov_episodes = list(range(window, len(rewards) + 1))
                        ax1.plot(mov_episodes, moving_avg, color=colors[i], linewidth=3, alpha=0.7, linestyle='--')

            ax1.set_title('Episode Rewards')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.legend()
            ax1.grid(True)

            # Plot 2: Critic Loss
            ax2 = axes[0, 1]
            if len(self.critic_losses) > 0:
                ax2.plot(range(1, len(self.critic_losses) + 1), self.critic_losses, 'r-', linewidth=2)
                ax2.set_title('Critic Loss')
                ax2.set_xlabel('Update Step')
                ax2.set_ylabel('Loss')
                ax2.set_yscale('log')
                ax2.grid(True)

            # Plot 3: Entropy (MOST IMPORTANT)
            ax3 = axes[1, 0]
            for i, agent in enumerate(self.agent_names):
                if len(self.entropies[agent]) > 0:
                    ax3.plot(range(1, len(self.entropies[agent]) + 1), self.entropies[agent],
                             color=colors[i], linewidth=2, label=agent)

            ax3.set_title('🔥 ENTROPY (Exploration Level)')
            ax3.set_xlabel('Update Step')
            ax3.set_ylabel('Entropy')
            ax3.legend()
            ax3.grid(True)

            # Plot 4: Episode Lengths
            ax4 = axes[1, 1]
            if len(self.episode_lengths) > 0:
                ax4.plot(range(1, len(self.episode_lengths) + 1), self.episode_lengths, 'purple', linewidth=2)
                ax4.set_title('Episode Lengths')
                ax4.set_xlabel('Episode')
                ax4.set_ylabel('Length')
                ax4.grid(True)

            plt.tight_layout()

            # Save plot
            suffix = 'final' if final else f'episode_{episode}'
            plot_path = os.path.join(self.save_dir, f'training_plot_{suffix}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')

            # FORCE DISPLAY
            plt.show()
            plt.close()

            print(f"✅ Plot saved: {plot_path}")

        except Exception as e:
            print(f"❌ Plot error: {e}")
            import traceback
            traceback.print_exc()

    def _log_progress(self, episode):
        """Simple progress logging"""
        print(f"\n🔥 Episode {episode} Progress:")
        print("-" * 40)

        # Recent rewards
        recent = min(10, len(self.episode_rewards[self.agent_names[0]]))
        for agent in self.agent_names:
            if len(self.episode_rewards[agent]) >= recent:
                recent_rewards = self.episode_rewards[agent][-recent:]
                avg_reward = np.mean(recent_rewards)
                print(f"{agent}: {avg_reward:.3f}")

        # Recent entropy
        if len(self.entropies[self.agent_names[0]]) > 0:
            print("\n🔄 Recent Entropy:")
            for agent in self.agent_names:
                if len(self.entropies[agent]) > 0:
                    recent_entropy = self.entropies[agent][-1]
                    print(f"{agent}: {recent_entropy:.4f}")

        # Critic loss
        if len(self.critic_losses) > 0:
            recent_loss = self.critic_losses[-1]
            print(f"\n📉 Critic Loss: {recent_loss:.6f}")


def main():
    """Simple main with guaranteed plots"""
    print("🔥 Simple AGGRESSIVE MAPPO with GUARANTEED PLOTS!")

    # Initialize environment
    env = make_airline_env(
        flights_csv='flights_data.csv',
        aircraft_csv='aircraft_data.csv',
        crew_csv='crew_data.csv',
        action_space_type="discrete",
        max_flights_per_episode=20,
        episode_length=50,
        weekly_budget=50000,
        max_emissions=5000
    )

    config = {
        'save_dir': 'simple_aggressive_mappo'
    }

    # Initialize trainer
    trainer = SimpleMAPPOTrainer(env, config)

    # Train with GUARANTEED plots every 50 episodes
    print("🎯 Training with plots every 50 episodes...")
    trainer.train(500)

    print("🎉 Training completed with plots!")


if __name__ == "__main__":
    main()