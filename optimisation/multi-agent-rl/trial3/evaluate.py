import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

# Import your training modules
from mappo_training import MAPPONetwork, get_training_config
from environment import AirlineSchedulingMAEnvironment


class SimpleEvaluator:
    """Simple evaluator for MAPPO agents - focused on key metrics only"""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = get_training_config()

    def load_models(self, env):
        """Load trained actors with dynamic dimension detection"""
        self.env = env
        self.actors = {}

        print("🔍 Loading models with dynamic dimension detection...")

        # Get actual observation dimensions from environment (like training script does)
        try:
            test_obs, _ = env.reset()
            actual_obs_dims = {}
            for agent_name in env.agents:
                actual_dim = len(test_obs[agent_name])
                actual_obs_dims[agent_name] = actual_dim
                print(f"📊 {agent_name}: detected {actual_dim} observation dimensions")
        except Exception as e:
            print(f"⚠️ Could not reset environment to detect dimensions: {e}")
            # Fallback to observation_spaces
            actual_obs_dims = {}
            for agent_name in env.agents:
                actual_dim = env.observation_spaces[agent_name].shape[0]
                actual_obs_dims[agent_name] = actual_dim
                print(f"📊 {agent_name}: using observation_space {actual_dim} dimensions")

        # Load models with detected dimensions
        for agent_name in env.agents:
            obs_dim = actual_obs_dims[agent_name]
            action_space = env.action_spaces[agent_name]

            # Create network with detected dimensions
            actor = MAPPONetwork(obs_dim, action_space,
                                 self.config['actor_hidden_dim'], is_critic=False).to(self.device)

            # Load weights
            model_path = os.path.join(self.model_dir, f'{agent_name}_actor_final.pth')

            try:
                checkpoint = torch.load(model_path, map_location=self.device)

                # Verify dimensions match saved model
                first_layer_weight = checkpoint['feature_extractor.0.weight']
                saved_obs_dim = first_layer_weight.shape[1]

                if saved_obs_dim != obs_dim:
                    print(f"❌ {agent_name}: dimension mismatch!")
                    print(f"   Saved model: {saved_obs_dim} dims")
                    print(f"   Current env: {obs_dim} dims")
                    raise ValueError(f"Dimension mismatch for {agent_name}")

                actor.load_state_dict(checkpoint)
                actor.eval()
                self.actors[agent_name] = actor
                print(f"✅ {agent_name}: loaded successfully ({obs_dim} dims)")

            except Exception as e:
                print(f"❌ Failed to load {agent_name}: {e}")
                raise

        print("🎯 All models loaded and ready for evaluation!")

    def select_actions(self, observations):
        """Get deterministic actions from trained models"""
        actions = {}

        for agent_name in self.env.agents:
            obs_tensor = torch.FloatTensor(observations[agent_name]).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action_logits = self.actors[agent_name](obs_tensor)
                action_space = self.env.action_spaces[agent_name]

                from gymnasium import spaces
                if isinstance(action_space, spaces.MultiBinary):
                    action = (action_logits > 0.5).float().squeeze(0).cpu().numpy()
                elif isinstance(action_space, spaces.MultiDiscrete):
                    actions_list = []
                    start_idx = 0
                    for dim in action_space.nvec:
                        end_idx = start_idx + int(dim)
                        logits_slice = action_logits[:, start_idx:end_idx]
                        action_slice = torch.argmax(logits_slice, dim=-1).item()
                        actions_list.append(action_slice)
                        start_idx = end_idx
                    action = np.array(actions_list)

                actions[agent_name] = action

        return actions

    def evaluate(self, num_episodes=10):
        """Run evaluation and collect results"""
        print(f"Running evaluation for {num_episodes} episodes...")

        results = {
            'scheduled_flights': [],
            'total_flights': [],
            'aircraft_assignments': [],
            'crew_assignments': [],
            'emissions': []
        }

        for episode in range(num_episodes):
            obs, _ = self.env.reset()

            while True:
                actions = self.select_actions(obs)
                obs, rewards, terminated, truncated, _ = self.env.step(actions)

                if any(terminated.values()) or any(truncated.values()):
                    break

            # Collect episode results
            scheduled = len(self.env.scheduled_flights)
            total = len(self.env.current_episode_flights)

            results['scheduled_flights'].append(scheduled)
            results['total_flights'].append(total)
            results['aircraft_assignments'].append(dict(self.env.aircraft_assignments))
            results['crew_assignments'].append(dict(self.env.crew_assignments))

            # Calculate emissions
            emissions = self._calculate_emissions()
            results['emissions'].append(emissions)

        return results

    def _calculate_emissions(self):
        """Simple emissions calculation"""
        total_emissions = 0
        for flight_id in self.env.scheduled_flights:
            if flight_id in self.env.aircraft_assignments and flight_id in self.env.flights_lookup:
                flight = self.env.flights_lookup[flight_id]
                aircraft_id = self.env.aircraft_assignments[flight_id]
                if aircraft_id in self.env.aircraft_lookup:
                    aircraft = self.env.aircraft_lookup[aircraft_id]
                    emissions = aircraft.get('fuel_consumption_per_nm', 0.5) * flight['distance_nm']
                    total_emissions += emissions
        return total_emissions

    def print_metrics(self, results):
        """Print key metrics"""
        scheduled = np.array(results['scheduled_flights'])
        total = np.array(results['total_flights'])
        emissions = np.array(results['emissions'])

        # Scheduling percentage
        scheduling_rate = np.mean(scheduled / total) * 100

        # Resource utilization
        all_aircraft = []
        all_crew = []

        for assignments in results['aircraft_assignments']:
            all_aircraft.extend(assignments.values())

        for assignments in results['crew_assignments']:
            for crew_list in assignments.values():
                all_crew.extend(crew_list)

        aircraft_utilization = len(set(all_aircraft)) / len(self.env.available_aircraft) * 100
        crew_utilization = len(set(all_crew)) / len(self.env.available_crew) * 100

        print("\n" + "=" * 50)
        print("🎯 EVALUATION RESULTS")
        print("=" * 50)
        print(f"📊 Scheduling Rate: {scheduling_rate:.1f}%")
        print(f"✈️  Aircraft Utilization: {aircraft_utilization:.1f}%")
        print(f"👥 Crew Utilization: {crew_utilization:.1f}%")
        print(f"🌍 Average Emissions: {np.mean(emissions):.1f} kg CO2")
        print(f"📈 Flights Scheduled: {np.mean(scheduled):.1f}/{np.mean(total):.1f}")
        print("=" * 50)

    def plot_assignments(self, results, save_dir="evaluation_plots"):
        """Create assignment visualizations"""
        os.makedirs(save_dir, exist_ok=True)

        # Collect all assignment data
        all_aircraft_assignments = []
        all_pilot_assignments = []
        all_cabin_assignments = []

        for aircraft_assign, crew_assign in zip(results['aircraft_assignments'], results['crew_assignments']):
            # Aircraft assignments
            all_aircraft_assignments.extend(aircraft_assign.values())

            # Separate pilots and cabin crew
            for crew_list in crew_assign.values():
                for crew_id in crew_list:
                    if crew_id in self.env.crew_lookup:
                        crew_type = self.env.crew_lookup[crew_id]['crew_type'].lower()
                        if 'pilot' in crew_type:
                            all_pilot_assignments.append(crew_id)
                        elif 'cabin' in crew_type:
                            all_cabin_assignments.append(crew_id)

        # Create plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Assignment Analysis for Scheduled Flights', fontsize=14, fontweight='bold')

        # Plot 1: Aircraft Assignment
        ax1 = axes[0]
        if all_aircraft_assignments:
            aircraft_counts = Counter(all_aircraft_assignments)
            aircraft_ids = list(aircraft_counts.keys())
            counts = list(aircraft_counts.values())

            bars = ax1.bar(range(len(aircraft_ids)), counts, color='steelblue', alpha=0.7)
            ax1.set_xlabel('Aircraft ID')
            ax1.set_ylabel('Times Assigned')
            ax1.set_title('Aircraft Assignment to Flights')
            ax1.set_xticks(range(len(aircraft_ids)))
            ax1.set_xticklabels([f'A{i + 1}' for i in range(len(aircraft_ids))], rotation=45)

            # Add value labels
            for bar, count in zip(bars, counts):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                         str(count), ha='center', va='bottom')

        # Plot 2: Pilot Assignment
        ax2 = axes[1]
        if all_pilot_assignments:
            pilot_counts = Counter(all_pilot_assignments)
            pilot_ids = list(pilot_counts.keys())
            counts = list(pilot_counts.values())

            bars = ax2.bar(range(len(pilot_ids)), counts, color='darkgreen', alpha=0.7)
            ax2.set_xlabel('Pilot ID')
            ax2.set_ylabel('Times Assigned')
            ax2.set_title('Pilot Assignment to Flights')
            ax2.set_xticks(range(len(pilot_ids)))
            ax2.set_xticklabels([f'P{i + 1}' for i in range(len(pilot_ids))], rotation=45)

            # Add value labels
            for bar, count in zip(bars, counts):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                         str(count), ha='center', va='bottom')

        # Plot 3: Cabin Crew Assignment
        ax3 = axes[2]
        if all_cabin_assignments:
            cabin_counts = Counter(all_cabin_assignments)
            cabin_ids = list(cabin_counts.keys())
            counts = list(cabin_counts.values())

            bars = ax3.bar(range(len(cabin_ids)), counts, color='coral', alpha=0.7)
            ax3.set_xlabel('Cabin Crew ID')
            ax3.set_ylabel('Times Assigned')
            ax3.set_title('Cabin Crew Assignment to Flights')
            ax3.set_xticks(range(len(cabin_ids)))
            ax3.set_xticklabels([f'C{i + 1}' for i in range(len(cabin_ids))], rotation=45)

            # Add value labels
            for bar, count in zip(bars, counts):
                ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                         str(count), ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'assignment_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📈 Assignment plots saved to: {save_dir}/assignment_analysis.png")


def diagnose_dimension_mismatch(model_dir: str, test_env):
    """Diagnose dimension mismatches between training and test environments"""
    print("\n🔍 DIMENSION MISMATCH DIAGNOSIS")
    print("=" * 50)

    for agent_name in test_env.agents:
        model_path = os.path.join(model_dir, f'{agent_name}_actor_final.pth')

        if os.path.exists(model_path):
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')

            # Get training dimensions
            first_layer_weight = checkpoint['feature_extractor.0.weight']
            trained_obs_dim = first_layer_weight.shape[1]

            # Get current dimensions
            current_obs_dim = test_env.observation_spaces[agent_name].shape[0]
            current_action_space = test_env.action_spaces[agent_name]

            print(f"\n📊 {agent_name.upper()}:")
            print(f"  Observation Dimension:")
            print(f"    Training: {trained_obs_dim}")
            print(f"    Test:     {current_obs_dim}")
            print(f"    Match:    {'✅ YES' if trained_obs_dim == current_obs_dim else '❌ NO'}")

            # Action space analysis
            if 'action_head.weight' in checkpoint:
                trained_action_dim = checkpoint['action_head.weight'].shape[0]
                from gymnasium import spaces
                if isinstance(current_action_space, spaces.MultiBinary):
                    current_action_dim = current_action_space.n
                elif isinstance(current_action_space, spaces.MultiDiscrete):
                    current_action_dim = sum(current_action_space.nvec)

                print(f"  Action Dimension:")
                print(f"    Training: {trained_action_dim}")
                print(f"    Test:     {current_action_dim}")
                print(f"    Match:    {'✅ YES' if trained_action_dim == current_action_dim else '❌ NO'}")

    print("\n💡 SOLUTIONS:")
    print("-" * 30)
    print("1. Use the SAME CSV files for testing as training")
    print("2. OR retrain models with your new test data")
    print("3. OR ensure test data has same structure as training data")
    print("=" * 50)


def main():
    """Simple evaluation main function with dynamic dimension handling"""
    # Setup
    model_dir = "01_mappo_airline_models"

    print("🚀 Starting MAPPO evaluation with dynamic dimensions...")
    print("Using the EXACT same data as training...")

    try:
        # Use the SAME files as training (not test_ files)
        env = AirlineSchedulingMAEnvironment(
            flights_csv_path='flights_data.csv',  # Same as training
            aircraft_csv_path='aircraft_data.csv',  # Same as training
            crew_csv_path='crew_data.csv',  # Same as training
            weekly_budget=50000,
            max_emissions=5000
        )

        print(f"✅ Environment loaded:")
        if hasattr(env, 'flights_data'):
            print(f"   Flights: {len(env.flights_data)}")
        if hasattr(env, 'aircraft_data'):
            print(f"   Aircraft: {len(env.aircraft_data)}")
        if hasattr(env, 'crew_data'):
            print(f"   Crew: {len(env.crew_data)}")

        # Initialize evaluator and load models
        evaluator = SimpleEvaluator(model_dir)
        evaluator.load_models(env)

        # Run evaluation
        print("\n🔄 Running evaluation...")
        results = evaluator.evaluate(num_episodes=50)

        # Show results
        evaluator.print_metrics(results)
        evaluator.plot_assignments(results)

    except ValueError as e:
        print(f"\n❌ Dimension Error: {e}")
        print("\n🔍 TROUBLESHOOTING:")
        print("1. Make sure you're using the SAME CSV files as training")
        print("2. Check if the training script used different data")
        print("3. Verify model files exist and were saved correctly")

    except FileNotFoundError as e:
        print(f"\n❌ File Error: {e}")
        print("\n💡 Solutions:")
        print("1. Check if model files exist in 'mappo_airline_models' directory")
        print("2. Verify CSV files exist and have correct names")
        print("3. Make sure training completed successfully")

    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n🔍 Please check your environment setup and file paths")


if __name__ == "__main__":
    main()