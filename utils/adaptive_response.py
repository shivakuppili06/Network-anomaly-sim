import numpy as np
import json
import os
import matplotlib.pyplot as plt

# NOTE: This module demonstrates RL-based decision optimization using synthetic reward assumptions.
# It does NOT use real telecom outcome data.

class EpsilonGreedyBandit:
    def __init__(self, actions, contexts, epsilon=0.1):
        self.actions = actions
        self.contexts = contexts
        self.epsilon = epsilon
        
        # Q-values: Expected reward for each action in each context
        self.q_table = {c: {a: 0.0 for a in actions} for c in contexts}
        # Counts: Number of times each action was chosen in each context
        self.n_table = {c: {a: 0 for a in actions} for c in contexts}
        
    def select_action(self, context):
        if np.random.rand() < self.epsilon:
            # Explore
            return np.random.choice(self.actions)
        else:
            # Exploit
            q_values = self.q_table[context]
            max_q = max(q_values.values())
            # Handle ties randomly
            best_actions = [a for a, q in q_values.items() if q == max_q]
            return np.random.choice(best_actions)
            
    def update(self, context, action, reward):
        self.n_table[context][action] += 1
        n = self.n_table[context][action]
        q = self.q_table[context][action]
        # Incremental update formula
        self.q_table[context][action] = q + (1.0 / n) * (reward - q)

def run_simulation(num_events=5000):
    print("Starting Adaptive Response Simulation (Contextual Bandit)...")
    
    actions = ['reroute', 'restart', 'adjust_qos', 'escalate']
    contexts = ['high_latency', 'throughput_drop', 'jitter_spike', 'packet_loss_spike']
    
    # Synthetic reward probabilities (success rates) for (context, action)
    reward_probs = {
        'high_latency': {'reroute': 0.8, 'restart': 0.1, 'adjust_qos': 0.5, 'escalate': 0.6},
        'throughput_drop': {'reroute': 0.3, 'restart': 0.85, 'adjust_qos': 0.2, 'escalate': 0.6},
        'jitter_spike': {'reroute': 0.2, 'restart': 0.1, 'adjust_qos': 0.8, 'escalate': 0.6},
        'packet_loss_spike': {'reroute': 0.7, 'restart': 0.4, 'adjust_qos': 0.5, 'escalate': 0.6}
    }
    
    bandit = EpsilonGreedyBandit(actions, contexts, epsilon=0.15)
    
    bandit_rewards = []
    random_rewards = []
    
    cumulative_bandit_reward = 0
    cumulative_random_reward = 0
    
    history_bandit = []
    history_random = []
    
    # Generate random sequence of anomaly contexts
    np.random.seed(42) # For reproducibility of the simulation
    event_contexts = np.random.choice(contexts, num_events)
    
    for context in event_contexts:
        # 1. Bandit Agent
        action_b = bandit.select_action(context)
        prob_b = reward_probs[context][action_b]
        reward_b = 1 if np.random.rand() < prob_b else 0
        bandit.update(context, action_b, reward_b)
        cumulative_bandit_reward += reward_b
        history_bandit.append(cumulative_bandit_reward)
        bandit_rewards.append(reward_b)
        
        # 2. Random Baseline Agent
        action_r = np.random.choice(actions)
        prob_r = reward_probs[context][action_r]
        reward_r = 1 if np.random.rand() < prob_r else 0
        cumulative_random_reward += reward_r
        history_random.append(cumulative_random_reward)
        random_rewards.append(reward_r)
        
    os.makedirs('results', exist_ok=True)
    
    # Save Report
    report = {
        'total_events': num_events,
        'bandit_accuracy': cumulative_bandit_reward / num_events,
        'random_accuracy': cumulative_random_reward / num_events,
        'final_q_values': bandit.q_table
    }
    
    with open('results/bandit_report.json', 'w') as f:
        json.dump(report, f, indent=4)
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(history_bandit, label=f'Contextual Bandit (Epsilon-Greedy)', linewidth=2)
    plt.plot(history_random, label='Random Baseline', linestyle='--', linewidth=2)
    plt.xlabel('Simulated Anomaly Events')
    plt.ylabel('Cumulative Reward (Successful Fixes)')
    plt.title('Adaptive Response: Bandit vs Baseline')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/bandit_performance.png')
    
    print(f"Simulation Complete. Bandit Accuracy: {report['bandit_accuracy']:.2%} | Random: {report['random_accuracy']:.2%}")
    print("Saved results to results/bandit_report.json and results/bandit_performance.png")

if __name__ == '__main__':
    run_simulation()
