import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

print("=" * 80)
print("DISCRETE EVENT SIMULATION - SIMPY")
print("Sri Lanka Intercity Express Bus Network")
print("=" * 80)

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

# Load real data to calibrate parameters
df = pd.read_csv('busnetworkdata.csv')
peak_data = df[df['PeakPeriod'] == 'Peak']

# Simulation time (in minutes) - simulate peak period (2 hours)
SIM_TIME = 120  # 2 hours = 120 minutes

# Calculate arrival rate from data (buses per minute during peak)
peak_trips = len(peak_data)
peak_hours = 2 * 5  # 2 peak hours per day × 5 days
ARRIVAL_RATE = peak_trips / (peak_hours * 60)  # buses per minute

# Service time parameters (waiting time at terminal)
MEAN_SERVICE_TIME = peak_data['WaitingTime'].mean()  # minutes
STD_SERVICE_TIME = peak_data['WaitingTime'].std()

# Number of terminals (servers)
NUM_TERMINALS = 4

# Number of replications for statistical confidence
NUM_REPLICATIONS = 10

print(f"\nSimulation Parameters:")
print(f"  Simulation Time: {SIM_TIME} minutes")
print(f"  Arrival Rate: {ARRIVAL_RATE:.4f} buses/minute ({ARRIVAL_RATE*60:.2f} buses/hour)")
print(f"  Mean Service Time: {MEAN_SERVICE_TIME:.2f} minutes")
print(f"  Std Dev Service Time: {STD_SERVICE_TIME:.2f} minutes")
print(f"  Number of Terminals: {NUM_TERMINALS}")
print(f"  Replications: {NUM_REPLICATIONS}")

# ============================================================================
# SIMULATION MODEL
# ============================================================================

class BusTerminalSimulation:
    def __init__(self, env, num_terminals, arrival_rate, mean_service, std_service):
        self.env = env
        self.terminals = simpy.Resource(env, capacity=num_terminals)
        self.arrival_rate = arrival_rate
        self.mean_service = mean_service
        self.std_service = std_service
        
        # Metrics
        self.buses_arrived = 0
        self.buses_served = 0
        self.waiting_times = []
        self.queue_lengths = []
        self.service_times = []
        self.total_time_in_system = []
        
    def bus_arrival(self, bus_id):
        """Process each bus arrival"""
        arrival_time = self.env.now
        self.buses_arrived += 1
        
        # Record queue length at arrival
        queue_length = len(self.terminals.queue)
        self.queue_lengths.append(queue_length)
        
        # Request terminal
        with self.terminals.request() as request:
            yield request
            
            # Calculate waiting time
            wait_time = self.env.now - arrival_time
            self.waiting_times.append(wait_time)
            
            # Service time (normal distribution, minimum 1 minute)
            service_time = max(1, np.random.normal(self.mean_service, self.std_service))
            self.service_times.append(service_time)
            
            # Process the bus
            yield self.env.timeout(service_time)
            
            # Bus departs
            self.buses_served += 1
            total_time = self.env.now - arrival_time
            self.total_time_in_system.append(total_time)
    
    def generate_arrivals(self):
        """Generate bus arrivals using Poisson process"""
        bus_id = 0
        while True:
            # Exponential inter-arrival time
            inter_arrival = np.random.exponential(1 / self.arrival_rate)
            yield self.env.timeout(inter_arrival)
            
            bus_id += 1
            self.env.process(self.bus_arrival(bus_id))

def run_simulation(num_terminals, arrival_rate, mean_service, std_service, sim_time):
    """Run a single simulation replication"""
    env = simpy.Environment()
    terminal_sim = BusTerminalSimulation(env, num_terminals, arrival_rate, 
                                         mean_service, std_service)
    env.process(terminal_sim.generate_arrivals())
    env.run(until=sim_time)
    
    return {
        'buses_arrived': terminal_sim.buses_arrived,
        'buses_served': terminal_sim.buses_served,
        'avg_wait_time': np.mean(terminal_sim.waiting_times) if terminal_sim.waiting_times else 0,
        'max_wait_time': np.max(terminal_sim.waiting_times) if terminal_sim.waiting_times else 0,
        'avg_queue_length': np.mean(terminal_sim.queue_lengths) if terminal_sim.queue_lengths else 0,
        'max_queue_length': np.max(terminal_sim.queue_lengths) if terminal_sim.queue_lengths else 0,
        'avg_service_time': np.mean(terminal_sim.service_times) if terminal_sim.service_times else 0,
        'avg_system_time': np.mean(terminal_sim.total_time_in_system) if terminal_sim.total_time_in_system else 0,
        'utilization': (sum(terminal_sim.service_times) / (num_terminals * sim_time)) * 100 if terminal_sim.service_times else 0
    }

def run_multiple_replications(num_terminals, arrival_rate, mean_service, std_service, 
                               sim_time, num_reps, scenario_name):
    """Run multiple replications and compute statistics"""
    print(f"\nRunning {scenario_name}...")
    results = []
    
    for rep in range(num_reps):
        result = run_simulation(num_terminals, arrival_rate, mean_service, std_service, sim_time)
        results.append(result)
        if (rep + 1) % 5 == 0:
            print(f"  Completed {rep + 1}/{num_reps} replications")
    
    # Compute statistics
    df_results = pd.DataFrame(results)
    
    stats_summary = {
        'scenario': scenario_name,
        'num_terminals': num_terminals,
        'avg_wait_mean': df_results['avg_wait_time'].mean(),
        'avg_wait_std': df_results['avg_wait_time'].std(),
        'avg_wait_ci': 1.96 * df_results['avg_wait_time'].std() / np.sqrt(num_reps),
        'avg_queue_mean': df_results['avg_queue_length'].mean(),
        'avg_queue_std': df_results['avg_queue_length'].std(),
        'buses_served_mean': df_results['buses_served'].mean(),
        'buses_served_std': df_results['buses_served'].std(),
        'utilization_mean': df_results['utilization'].mean(),
        'utilization_std': df_results['utilization'].std()
    }
    
    return stats_summary, df_results

# ============================================================================
# RUN SCENARIOS
# ============================================================================
print("\n" + "=" * 80)
print("SCENARIO SIMULATIONS")
print("=" * 80)

scenarios_results = []

# Current System
print("\n" + "-" * 80)
print("SCENARIO: CURRENT SYSTEM (4 Terminals)")
print("-" * 80)
current_stats, current_df = run_multiple_replications(
    NUM_TERMINALS, ARRIVAL_RATE, MEAN_SERVICE_TIME, STD_SERVICE_TIME,
    SIM_TIME, NUM_REPLICATIONS, "Current System"
)
scenarios_results.append(current_stats)

# Scenario 1: Add 1 Terminal
print("\n" + "-" * 80)
print("SCENARIO 1: ADD 1 TERMINAL (5 Terminals)")
print("-" * 80)
scenario1_stats, scenario1_df = run_multiple_replications(
    5, ARRIVAL_RATE, MEAN_SERVICE_TIME, STD_SERVICE_TIME,
    SIM_TIME, NUM_REPLICATIONS, "Scenario 1: +1 Terminal"
)
scenarios_results.append(scenario1_stats)

# Scenario 2: Process Improvement (+20% faster service)
improved_service_time = MEAN_SERVICE_TIME * 0.8  # 20% reduction in service time
improved_std = STD_SERVICE_TIME * 0.8
print("\n" + "-" * 80)
print("SCENARIO 2: PROCESS IMPROVEMENT (+20% Service Speed)")
print("-" * 80)
scenario2_stats, scenario2_df = run_multiple_replications(
    NUM_TERMINALS, ARRIVAL_RATE, improved_service_time, improved_std,
    SIM_TIME, NUM_REPLICATIONS, "Scenario 2: +20% Speed"
)
scenarios_results.append(scenario2_stats)

# Scenario 3: Combined Approach
print("\n" + "-" * 80)
print("SCENARIO 3: COMBINED (5 Terminals + 20% Speed)")
print("-" * 80)
scenario3_stats, scenario3_df = run_multiple_replications(
    5, ARRIVAL_RATE, improved_service_time, improved_std,
    SIM_TIME, NUM_REPLICATIONS, "Scenario 3: Combined"
)
scenarios_results.append(scenario3_stats)

# ============================================================================
# RESULTS SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SIMULATION RESULTS SUMMARY")
print("=" * 80)

results_df = pd.DataFrame(scenarios_results)

print("\n" + "-" * 80)
print("Average Waiting Time (minutes)")
print("-" * 80)
for _, row in results_df.iterrows():
    ci = row['avg_wait_ci']
    print(f"{row['scenario']:30s}: {row['avg_wait_mean']:6.2f} ± {ci:5.2f} "
          f"(95% CI: [{row['avg_wait_mean']-ci:.2f}, {row['avg_wait_mean']+ci:.2f}])")

print("\n" + "-" * 80)
print("Average Queue Length (buses)")
print("-" * 80)
for _, row in results_df.iterrows():
    print(f"{row['scenario']:30s}: {row['avg_queue_mean']:6.2f} ± {row['avg_queue_std']:5.2f}")

print("\n" + "-" * 80)
print("Buses Served")
print("-" * 80)
for _, row in results_df.iterrows():
    print(f"{row['scenario']:30s}: {row['buses_served_mean']:6.1f} ± {row['buses_served_std']:5.2f}")

print("\n" + "-" * 80)
print("Terminal Utilization (%)")
print("-" * 80)
for _, row in results_df.iterrows():
    print(f"{row['scenario']:30s}: {row['utilization_mean']:6.2f}% ± {row['utilization_std']:5.2f}%")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Chart 1: Waiting Time with Confidence Intervals
ax1 = axes[0, 0]
scenarios = results_df['scenario'].tolist()
wait_means = results_df['avg_wait_mean'].tolist()
wait_cis = results_df['avg_wait_ci'].tolist()

x_pos = np.arange(len(scenarios))
bars1 = ax1.bar(x_pos, wait_means, yerr=wait_cis, capsize=5,
                color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'], 
                alpha=0.7, edgecolor='black')
ax1.axhline(8, color='red', linestyle='--', linewidth=2, label='Target: 8 min')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(scenarios, rotation=15, ha='right')
ax1.set_ylabel('Average Waiting Time (minutes)', fontsize=11)
ax1.set_title('Waiting Time Comparison (with 95% CI)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + wait_cis[i],
             f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Chart 2: Queue Length
ax2 = axes[0, 1]
queue_means = results_df['avg_queue_mean'].tolist()
bars2 = ax2.bar(x_pos, queue_means,
                color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'], 
                alpha=0.7, edgecolor='black')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(scenarios, rotation=15, ha='right')
ax2.set_ylabel('Average Queue Length (buses)', fontsize=11)
ax2.set_title('Queue Length Comparison', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}', ha='center', va='bottom', fontsize=9)

# Chart 3: Buses Served
ax3 = axes[1, 0]
served_means = results_df['buses_served_mean'].tolist()
bars3 = ax3.bar(x_pos, served_means,
                color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'], 
                alpha=0.7, edgecolor='black')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(scenarios, rotation=15, ha='right')
ax3.set_ylabel('Buses Served', fontsize=11)
ax3.set_title('Throughput Comparison', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

for i, bar in enumerate(bars3):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}', ha='center', va='bottom', fontsize=9)

# Chart 4: Improvement Percentage
ax4 = axes[1, 1]
baseline = wait_means[0]
improvements = [(baseline - wt) / baseline * 100 for wt in wait_means[1:]]
bars4 = ax4.bar(range(len(improvements)), improvements,
                color=['#3498db', '#2ecc71', '#f39c12'], 
                alpha=0.7, edgecolor='black')
ax4.set_xticks(range(len(improvements)))
ax4.set_xticklabels(scenarios[1:], rotation=15, ha='right')
ax4.set_ylabel('Waiting Time Reduction (%)', fontsize=11)
ax4.set_title('Improvement vs Current System', fontsize=12, fontweight='bold')
ax4.axhline(0, color='black', linewidth=1)
ax4.grid(True, alpha=0.3, axis='y')

for i, bar in enumerate(bars4):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', 
             va='bottom' if height > 0 else 'top', fontsize=9)

plt.suptitle('SimPy Simulation Results - Bus Terminal System', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('simulation_results.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'simulation_results.png'")
plt.show()

# Save results to CSV
results_df.to_csv('simulation_results.csv', index=False)
print("✓ Results saved to 'simulation_results.csv'")

print("\n" + "=" * 80)
print("SIMULATION COMPLETE!")
print("=" * 80)