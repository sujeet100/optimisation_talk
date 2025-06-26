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
    """MAPPO Actor-Critic Network with shared feature extraction"""

    def __init__(self, obs_dim: int, action_space, hidden_dim: int = 256, is_critic: bool = False):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_space = action_space
        self.hidden_dim = hidden_dim
        self.is_critic = is_critic

        # Shared feature extractor - RESTORED original architecture
        self.feature_extractor = nn.Sequential(
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
        features = self.feature_extractor(obs)

        if self.is_critic:
            return self.value_head(features)
        else:
            return self._get_action_logits(features)

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
            dist = Bernoulli(action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            actions = []
            log_probs = []
            start_idx = 0

            for dim in self.action_space.nvec:
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
    """Centralized Critic for CTDE (Centralized Training, Decentralized Execution)"""

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
        return self.critic_network(global_obs)


class ReplayBuffer:
    """Experience replay buffer for MAPPO"""

    def __init__(self, buffer_size: int, num_agents: int):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.clear()

    def clear(self):
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
    """Multi-Agent PPO Trainer with CTDE"""

    def __init__(self, env, config: Dict):
        self.env = env
        self.config = config
        self.agent_names = env.agents
        self.num_agents = len(self.agent_names)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize networks
        self._init_networks()
        self._init_optimizers()

        # Experience buffer
        self.buffer = ReplayBuffer(config['buffer_size'], self.num_agents)

        # Training metrics
        self.training_metrics = defaultdict(list)
        self.episode_rewards = {agent: [] for agent in self.agent_names}
        self.episode_lengths = []

        # Create save directory
        self.save_dir = config.get('save_dir', 'mappo_models')
        os.makedirs(self.save_dir, exist_ok=True)

    def _init_networks(self):
        # Calculate total observation dimension for centralized critic
        total_obs_dim = sum(space.shape[0] for space in self.env.observation_spaces.values())

        # Centralized critic
        self.centralized_critic = CentralizedCritic(
            total_obs_dim=total_obs_dim,
            hidden_dim=self.config['critic_hidden_dim']
        ).to(self.device)

        # Decentralized actors
        self.actors = {}
        for agent_name in self.agent_names:
            obs_dim = self.env.observation_spaces[agent_name].shape[0]
            action_space = self.env.action_spaces[agent_name]

            actor = MAPPONetwork(
                obs_dim=obs_dim,
                action_space=action_space,
                hidden_dim=self.config['actor_hidden_dim'],
                is_critic=False
            ).to(self.device)

            self.actors[agent_name] = actor

    def _init_optimizers(self):
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
        global_obs = torch.FloatTensor(batch['global_observations']).to(self.device)
        target_returns = torch.stack([all_returns[agent] for agent in self.agent_names]).mean(dim=0)

        values = self.centralized_critic(global_obs).view(-1)
        target_returns = target_returns.view(-1)

        critic_loss = F.mse_loss(values, target_returns.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.centralized_critic.parameters(), self.config['max_grad_norm'])
        self.critic_optimizer.step()

        self.training_metrics['critic_loss'].append(critic_loss.item())

    def _update_actors(self, batch: Dict, all_advantages: Dict):
        for i, agent_name in enumerate(self.agent_names):
            obs = torch.FloatTensor(batch['observations'][f'agent_{i}']).to(self.device)
            actions = torch.LongTensor(batch['actions'][f'agent_{i}']).to(self.device)
            old_log_probs = batch['log_probs'][f'agent_{i}'].to(self.device)
            advantages = all_advantages[agent_name].to(self.device)

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Get new action probabilities
            new_action_logits = self.actors[agent_name](obs)
            new_log_probs = self._compute_log_probs(new_action_logits, actions, agent_name)

            # Compute ratio and PPO loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config['clip_ratio'], 1 + self.config['clip_ratio']) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()

            # Update actor
            self.actor_optimizers[agent_name].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent_name].parameters(), self.config['max_grad_norm'])
            self.actor_optimizers[agent_name].step()

            self.training_metrics[f'{agent_name}_actor_loss'].append(actor_loss.item())

    def _compute_log_probs(self, action_logits: torch.Tensor, actions: torch.Tensor, agent_name: str) -> torch.Tensor:
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
        print(f"Starting MAPPO training for {num_episodes} episodes...")

        for episode in range(num_episodes):
            episode_rewards = {agent: 0 for agent in self.agent_names}
            episode_length = 0

            observations, _ = self.env.reset()

            while True:
                actions, log_probs, value = self.select_actions(observations)
                next_observations, rewards, terminated, truncated, info = self.env.step(actions)

                global_obs = np.concatenate([observations[agent] for agent in self.agent_names])
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
        recent_episodes = min(self.config['log_freq'], len(self.episode_rewards[self.agent_names[0]]))

        print(f"\nEpisode {episode}")
        print("-" * 40)

        for agent in self.agent_names:
            recent_rewards = self.episode_rewards[agent][-recent_episodes:]
            avg_reward = np.mean(recent_rewards)
            print(f"{agent}: {avg_reward:.3f}")

        avg_length = np.mean(self.episode_lengths[-recent_episodes:])
        print(f"Avg episode length: {avg_length:.1f}")

        if self.training_metrics['critic_loss']:
            recent_critic_loss = np.mean(self.training_metrics['critic_loss'][-5:])
            print(f"Critic loss: {recent_critic_loss:.6f}")

    def save_models(self, episode: int, final: bool = False):
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
                'training_metrics': dict(self.training_metrics)
            }, f)

        print(f"Models saved: {suffix}")

    def plot_training_curves(self, save_path: str = None):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('MAPPO Training Results', fontsize=14)

        colors = ['blue', 'orange', 'green', 'red']
        agent_colors = dict(zip(self.agent_names, colors))

        # Episode Rewards
        ax1 = axes[0, 0]
        for agent in self.agent_names:
            episodes = range(1, len(self.episode_rewards[agent]) + 1)
            rewards = self.episode_rewards[agent]
            ax1.plot(episodes, rewards, alpha=0.3, color=agent_colors[agent])

            if len(rewards) >= 10:
                moving_avg = np.convolve(rewards, np.ones(10) / 10, mode='valid')
                moving_episodes = range(10, len(rewards) + 1)
                ax1.plot(moving_episodes, moving_avg, label=agent, color=agent_colors[agent], linewidth=2)

        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Episode Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Average Rewards
        ax2 = axes[0, 1]
        if len(self.episode_rewards[self.agent_names[0]]) >= 50:
            agent_avg_rewards = []
            for agent in self.agent_names:
                recent_rewards = self.episode_rewards[agent][-50:]
                agent_avg_rewards.append(np.mean(recent_rewards))

            ax2.bar(range(len(self.agent_names)), agent_avg_rewards, color=colors)
            ax2.set_xticks(range(len(self.agent_names)))
            ax2.set_xticklabels([a.replace('_', ' ').title() for a in self.agent_names], rotation=45)

        ax2.set_ylabel('Average Reward')
        ax2.set_title('Average Rewards (Last 50 episodes)')

        # Episode Lengths
        ax3 = axes[1, 0]
        episodes = range(1, len(self.episode_lengths) + 1)
        ax3.plot(episodes, self.episode_lengths, alpha=0.6, color='purple')

        if len(self.episode_lengths) >= 10:
            moving_avg_lengths = np.convolve(self.episode_lengths, np.ones(10) / 10, mode='valid')
            moving_episodes = range(10, len(self.episode_lengths) + 1)
            ax3.plot(moving_episodes, moving_avg_lengths, color='darkblue', linewidth=2)

        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Episode Length')
        ax3.set_title('Episode Lengths')
        ax3.grid(True, alpha=0.3)

        # Training Losses
        ax4 = axes[1, 1]
        if self.training_metrics['critic_loss']:
            critic_episodes = np.linspace(1, len(self.episode_rewards[self.agent_names[0]]),
                                          len(self.training_metrics['critic_loss']))
            ax4.plot(critic_episodes, self.training_metrics['critic_loss'], label='Critic Loss', color='red')

        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Loss')
        ax4.set_title('Training Losses')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plots saved to: {save_path}")

        plt.show()


def get_training_config():
    return {
        # Network architecture - RESTORED original sizes
        'actor_hidden_dim': 256,
        'critic_hidden_dim': 512,

        # Training hyperparameters
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

        # Logging and saving
        'log_freq': 10,
        'save_freq': 50,
        'save_dir': '01_mappo_airline_models'
    }


def main():
    print("Initializing Airline Scheduling MAPPO Training...")

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

    print(f"Environment loaded with {env.num_flights} flights")
    print(f"Agents: {env.agents}")

    # Get training configuration
    config = get_training_config()

    # Initialize trainer
    trainer = MAPPOTrainer(env, config)

    # Train the model
    num_episodes = 500
    trainer.train(num_episodes)

    # Plot results
    plot_save_path = os.path.join(config['save_dir'], 'training_curves.png')
    trainer.plot_training_curves(save_path=plot_save_path)

    print(f"Training completed! Models saved in: {config['save_dir']}")


if __name__ == "__main__":
    main()