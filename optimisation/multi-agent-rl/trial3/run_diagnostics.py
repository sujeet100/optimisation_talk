"""
Standalone Training Diagnostics Tool for MAPPO

Usage:
1. Run this after your training script starts showing flat rewards
2. It will analyze your trainer and environment to identify issues
3. Provides specific recommendations for fixes

Example:
    from training_debug_tool import diagnose_training_issues
    diagnose_training_issues(trainer, num_test_episodes=10)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict


def diagnose_training_issues(trainer, num_test_episodes=10):
    """
    Main diagnostic function to identify why training rewards are flat

    Args:
        trainer: Your MAPPOTrainer instance
        num_test_episodes: Number of episodes to run for diagnosis
    """
    print("🔍 MAPPO Training Diagnostics")
    print("=" * 60)
    print(f"Analyzing {num_test_episodes} test episodes...")
    print()

    # Run all diagnostic checks
    reward_issues = check_reward_scale(trainer, num_test_episodes)
    exploration_issues = check_exploration(trainer, num_test_episodes)
    learning_issues = check_learning_progress(trainer)
    environment_issues = check_environment_signal(trainer, num_test_episodes)

    # Provide consolidated recommendations
    provide_recommendations(reward_issues, exploration_issues, learning_issues, environment_issues)

    return {
        'reward_issues': reward_issues,
        'exploration_issues': exploration_issues,
        'learning_issues': learning_issues,
        'environment_issues': environment_issues
    }


def check_reward_scale(trainer, num_episodes):
    """Check if rewards are in appropriate scale and range"""
    print("1. 📊 REWARD SCALE ANALYSIS")
    print("-" * 40)

    env = trainer.env
    issues = []

    # Collect reward statistics
    all_rewards = {agent: [] for agent in env.agents}
    step_rewards = {agent: [] for agent in env.agents}

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_rewards = {agent: 0 for agent in env.agents}

        while True:
            # Use current policy
            actions, _, _ = trainer.select_actions(obs)
            obs, rewards, terminated, truncated, _ = env.step(actions)

            for agent in env.agents:
                episode_rewards[agent] += rewards[agent]
                step_rewards[agent].append(rewards[agent])

            if any(terminated.values()) or any(truncated.values()):
                break

        for agent in env.agents:
            all_rewards[agent].append(episode_rewards[agent])

    # Analyze statistics
    for agent in env.agents:
        episode_array = np.array(all_rewards[agent])
        step_array = np.array(step_rewards[agent])

        print(f"\n{agent}:")
        print(f"  Episode rewards - Mean: {episode_array.mean():.4f}, Std: {episode_array.std():.4f}")
        print(f"  Episode range: [{episode_array.min():.4f}, {episode_array.max():.4f}]")
        print(f"  Step rewards - Mean: {step_array.mean():.4f}, Std: {step_array.std():.4f}")

        # Identify issues
        if abs(episode_array.mean()) < 0.001:
            print(f"  ⚠️  Issue: Episode rewards too small (near zero)")
            issues.append(f"{agent}_small_rewards")

        if episode_array.std() < 0.001:
            print(f"  ⚠️  Issue: No reward variance (all episodes same)")
            issues.append(f"{agent}_no_variance")

        if episode_array.max() - episode_array.min() < 0.01:
            print(f"  ⚠️  Issue: Very narrow reward range")
            issues.append(f"{agent}_narrow_range")

        if len(set(step_array)) < 3:
            print(f"  ⚠️  Issue: Very few unique step rewards")
            issues.append(f"{agent}_few_unique_rewards")

    return issues


def check_exploration(trainer, num_episodes):
    """Check if agents are exploring the action space"""
    print("\n\n2. 🎯 EXPLORATION ANALYSIS")
    print("-" * 40)

    env = trainer.env
    issues = []

    # Track action diversity
    action_counts = {agent: defaultdict(int) for agent in env.agents}
    total_actions = {agent: 0 for agent in env.agents}

    for episode in range(num_episodes):
        obs, _ = env.reset()

        while True:
            actions, _, _ = trainer.select_actions(obs)

            # Record actions
            for agent in env.agents:
                action = actions[agent]
                # Convert to hashable string
                if isinstance(action, np.ndarray):
                    action_key = str(tuple(action.flatten()))
                else:
                    action_key = str(action)

                action_counts[agent][action_key] += 1
                total_actions[agent] += 1

            obs, _, terminated, truncated, _ = env.step(actions)
            if any(terminated.values()) or any(truncated.values()):
                break

    # Analyze diversity
    for agent in env.agents:
        unique_actions = len(action_counts[agent])
        total = total_actions[agent]

        # Calculate entropy
        if total > 0:
            probs = [count / total for count in action_counts[agent].values()]
            entropy = -sum(p * np.log(p + 1e-8) for p in probs if p > 0)
        else:
            entropy = 0

        print(f"\n{agent}:")
        print(f"  Unique actions: {unique_actions}")
        print(f"  Total actions: {total}")
        print(f"  Action entropy: {entropy:.4f}")

        # Show top actions
        if action_counts[agent]:
            sorted_actions = sorted(action_counts[agent].items(),
                                    key=lambda x: x[1], reverse=True)
            top_3 = sorted_actions[:3]
            print(f"  Top actions: {[(action[:20], count) for action, count in top_3]}")

            # Check for issues
            if unique_actions < 3:
                print(f"  ⚠️  Issue: Very low action diversity")
                issues.append(f"{agent}_low_diversity")

            if entropy < 0.5:
                print(f"  ⚠️  Issue: Low exploration (entropy < 0.5)")
                issues.append(f"{agent}_low_exploration")

            # Check if stuck on one action
            top_action_freq = sorted_actions[0][1] / total
            if top_action_freq > 0.9:
                print(f"  ⚠️  Issue: Stuck on one action ({top_action_freq:.1%} frequency)")
                issues.append(f"{agent}_stuck_action")

    return issues


def check_learning_progress(trainer):
    """Check if networks are actually learning"""
    print("\n\n3. 🧠 LEARNING PROGRESS ANALYSIS")
    print("-" * 40)

    issues = []

    # Check gradient magnitudes
    print("\nGradient Analysis:")
    for agent_name in trainer.env.agents:
        actor = trainer.actors[agent_name]

        # Calculate gradient norms
        total_grad_norm = 0
        param_count = 0
        params_with_grad = 0

        for param in actor.parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                total_grad_norm += grad_norm ** 2
                params_with_grad += 1
            param_count += param.numel()

        if params_with_grad > 0:
            avg_grad_norm = (total_grad_norm ** 0.5) / params_with_grad
            print(f"  {agent_name}: Avg gradient norm = {avg_grad_norm:.6f}")

            if avg_grad_norm < 1e-6:
                print(f"    ⚠️  Issue: Vanishing gradients")
                issues.append(f"{agent_name}_vanishing_gradients")
            elif avg_grad_norm > 1.0:
                print(f"    ⚠️  Issue: Exploding gradients")
                issues.append(f"{agent_name}_exploding_gradients")
        else:
            print(f"  {agent_name}: No gradients computed")
            issues.append(f"{agent_name}_no_gradients")

    # Check loss trends
    print("\nLoss Trend Analysis:")
    if trainer.training_metrics['critic_loss']:
        losses = trainer.training_metrics['critic_loss']
        if len(losses) >= 10:
            recent_loss = np.mean(losses[-10:])
            initial_loss = np.mean(losses[:10])

            print(f"  Critic loss: {initial_loss:.6f} → {recent_loss:.6f}")

            if abs(recent_loss - initial_loss) / (initial_loss + 1e-8) < 0.01:
                print(f"    ⚠️  Issue: Critic loss not improving")
                issues.append("critic_loss_stagnant")
        else:
            print("  Not enough training data to analyze trends")

    # Check if rewards are improving
    print("\nReward Improvement Analysis:")
    for agent in trainer.env.agents:
        if len(trainer.episode_rewards[agent]) >= 20:
            rewards = trainer.episode_rewards[agent]
            early_rewards = np.mean(rewards[:10])
            recent_rewards = np.mean(rewards[-10:])

            improvement = recent_rewards - early_rewards
            print(f"  {agent}: {early_rewards:.3f} → {recent_rewards:.3f} (Δ{improvement:+.3f})")

            if abs(improvement) < 0.01:
                print(f"    ⚠️  Issue: No reward improvement")
                issues.append(f"{agent}_no_improvement")

    return issues


def check_environment_signal(trainer, num_episodes):
    """Check if environment provides meaningful learning signal"""
    print("\n\n4. 🌍 ENVIRONMENT SIGNAL ANALYSIS")
    print("-" * 40)

    env = trainer.env
    issues = []

    # Test random vs trained policy
    random_performance = test_random_policy(env, num_episodes // 2)
    trained_performance = test_trained_policy(trainer, num_episodes // 2)

    print("\nPolicy Comparison:")
    print("Random Policy:")
    for agent in env.agents:
        random_mean = np.mean(random_performance[agent])
        random_std = np.std(random_performance[agent])
        print(f"  {agent}: {random_mean:.4f} ± {random_std:.4f}")

    print("\nTrained Policy:")
    for agent in env.agents:
        trained_mean = np.mean(trained_performance[agent])
        trained_std = np.std(trained_performance[agent])
        print(f"  {agent}: {trained_mean:.4f} ± {trained_std:.4f}")

    print("\nImprovement over Random:")
    for agent in env.agents:
        improvement = np.mean(trained_performance[agent]) - np.mean(random_performance[agent])
        print(f"  {agent}: {improvement:+.4f}")

        if abs(improvement) < 0.01:
            print(f"    ⚠️  Issue: No improvement over random")
            issues.append(f"{agent}_no_improvement_over_random")

    # Check environment complexity
    print("\nEnvironment Complexity:")
    obs, _ = env.reset()
    total_obs_dim = sum(space.shape[0] for space in env.observation_spaces.values())
    total_action_dim = sum(_get_action_dim(space) for space in env.action_spaces.values())

    print(f"  Total observation dimensions: {total_obs_dim}")
    print(f"  Total action dimensions: {total_action_dim}")
    print(f"  Flights: {env.num_flights}, Aircraft: {env.num_aircraft}")
    print(f"  Pilots: {env.num_pilots}, Cabin crew: {env.num_cabin_crew}")

    if total_obs_dim > 1000:
        print(f"    ⚠️  Issue: Very high-dimensional observation space")
        issues.append("high_dimensional_obs")

    if total_action_dim > 500:
        print(f"    ⚠️  Issue: Very high-dimensional action space")
        issues.append("high_dimensional_action")

    return issues


def test_random_policy(env, num_episodes):
    """Test random policy performance"""
    all_rewards = {agent: [] for agent in env.agents}

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_rewards = {agent: 0 for agent in env.agents}

        while True:
            actions = {}
            for agent in env.agents:
                actions[agent] = env.action_spaces[agent].sample()

            obs, rewards, terminated, truncated, _ = env.step(actions)

            for agent in env.agents:
                episode_rewards[agent] += rewards[agent]

            if any(terminated.values()) or any(truncated.values()):
                break

        for agent in env.agents:
            all_rewards[agent].append(episode_rewards[agent])

    return all_rewards


def test_trained_policy(trainer, num_episodes):
    """Test current trained policy performance"""
    env = trainer.env
    all_rewards = {agent: [] for agent in env.agents}

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_rewards = {agent: 0 for agent in env.agents}

        while True:
            actions, _, _ = trainer.select_actions(obs)
            obs, rewards, terminated, truncated, _ = env.step(actions)

            for agent in env.agents:
                episode_rewards[agent] += rewards[agent]

            if any(terminated.values()) or any(truncated.values()):
                break

        for agent in env.agents:
            all_rewards[agent].append(episode_rewards[agent])

    return all_rewards


def _get_action_dim(action_space):
    """Get total action dimensions"""
    from gymnasium import spaces

    if isinstance(action_space, spaces.MultiBinary):
        return action_space.n
    elif isinstance(action_space, spaces.MultiDiscrete):
        return len(action_space.nvec)
    elif isinstance(action_space, spaces.Discrete):
        return 1
    else:
        return 1


def provide_recommendations(reward_issues, exploration_issues, learning_issues, environment_issues):
    """Provide specific recommendations based on identified issues"""
    print("\n\n5. 💡 RECOMMENDATIONS")
    print("=" * 60)

    all_issues = reward_issues + exploration_issues + learning_issues + environment_issues

    if not all_issues:
        print("✅ No major issues detected! Training should be working.")
        return

    print("Based on the diagnosis, here are specific fixes to try:")
    print()

    # Reward-related fixes
    reward_issue_types = [issue for issue in all_issues if
                          any(x in issue for x in ['small_rewards', 'no_variance', 'narrow_range'])]
    if reward_issue_types:
        print("🔧 REWARD ENGINEERING FIXES:")
        print("  1. Scale rewards to meaningful range:")
        print("     reward = np.tanh(raw_reward / 1000)  # Adjust divisor")
        print("  2. Add dense intermediate rewards:")
        print("     reward += 0.1 * progress_made")
        print("  3. Reduce large penalties that discourage action")
        print()

    # Exploration fixes
    exploration_issue_types = [issue for issue in all_issues if
                               any(x in issue for x in ['low_diversity', 'low_exploration', 'stuck_action'])]
    if exploration_issue_types:
        print("🎯 EXPLORATION FIXES:")
        print("  1. Increase entropy coefficient:")
        print("     config['entropy_coef'] = 0.05  # Higher than 0.02")
        print("  2. Add noise to actions during training:")
        print("     action += np.random.normal(0, 0.1, action.shape)")
        print("  3. Use epsilon-greedy exploration:")
        print("     if random.random() < epsilon: action = random_action()")
        print()

    # Learning fixes
    learning_issue_types = [issue for issue in all_issues if
                            any(x in issue for x in ['gradients', 'stagnant', 'no_improvement'])]
    if learning_issue_types:
        print("📚 LEARNING FIXES:")
        print("  1. Adjust learning rates:")
        print("     config['actor_lr'] = 1e-5    # Much smaller")
        print("     config['critic_lr'] = 5e-5   # Much smaller")
        print("  2. Change network architecture:")
        print("     config['actor_hidden_dim'] = 32   # Smaller")
        print("  3. More frequent updates:")
        print("     config['update_freq'] = 2    # Update every 2 episodes")
        print()

    # Environment fixes
    environment_issue_types = [issue for issue in all_issues if
                               any(x in issue for x in ['high_dimensional', 'no_improvement_over_random'])]
    if environment_issue_types:
        print("🌍 ENVIRONMENT FIXES:")
        print("  1. Reduce problem size:")
        print("     Use 10-15 flights instead of 30")
        print("  2. Simplify constraints:")
        print("     Allow partial crew assignments initially")
        print("  3. Add curriculum learning:")
        print("     Start easy, gradually increase difficulty")
        print()

    print("⚡ QUICK WINS TO TRY FIRST:")
    print("  1. Reduce learning rates by 10x")
    print("  2. Increase entropy_coef to 0.05")