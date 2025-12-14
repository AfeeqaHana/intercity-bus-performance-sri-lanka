import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import factorial
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('busnetworkdata.csv')

print("=" * 80)
print("QUEUING THEORY ANALYSIS - M/M/c MODEL")
print("Sri Lanka Intercity Express Bus Network")
print("=" * 80)

# ============================================================================
# 1. PARAMETER ESTIMATION
# ============================================================================
print("\n" + "=" * 80)
print("1. SYSTEM PARAMETERS")
print("=" * 80)

# Separate peak and off-peak data
peak_data = df[df['PeakPeriod'] == 'Peak']
offpeak_data = df[df['PeakPeriod'] == 'Off-Peak']

# Calculate arrival rate (λ) - buses arriving at terminal per hour
# Assuming analysis per terminal (4 terminals, distributed trips)
total_hours = 17  # Operating hours per day (5 AM - 10 PM)
total_days = 5

# Average buses per hour
buses_per_hour = len(df) / (total_days * total_hours)

# Service rate (μ) - rate at which buses are processed/dispatched per hour
# Based on waiting time at terminal
avg_waiting_time_hours = df['WaitingTime'].mean() / 60  # Convert to hours
service_rate = 1 / avg_waiting_time_hours if avg_waiting_time_hours > 0 else 0

print(f"\nOverall System:")
print(f"  Total Trips: {len(df)}")
print(f"  Operating Period: {total_days} days × {total_hours} hours = {total_days * total_hours} hours")
print(f"  Average Waiting Time: {df['WaitingTime'].mean():.2f} minutes")
print(f"  Arrival Rate (λ): {buses_per_hour:.3f} buses/hour")
print(f"  Service Rate (μ): {service_rate:.3f} buses/hour")

# Peak period analysis
peak_waiting_hours = peak_data['WaitingTime'].mean() / 60
peak_service_rate = 1 / peak_waiting_hours if peak_waiting_hours > 0 else 0
peak_arrival_rate = len(peak_data) / (total_days * 2)  # 2 peak hours per day

print(f"\nPeak Period (7-8 AM, 5-6 PM):")
print(f"  Peak Trips: {len(peak_data)}")
print(f"  Average Waiting Time: {peak_data['WaitingTime'].mean():.2f} minutes")
print(f"  Arrival Rate (λ): {peak_arrival_rate:.3f} buses/hour")
print(f"  Service Rate (μ): {peak_service_rate:.3f} buses/hour")

# Off-peak period analysis
offpeak_waiting_hours = offpeak_data['WaitingTime'].mean() / 60
offpeak_service_rate = 1 / offpeak_waiting_hours if offpeak_waiting_hours > 0 else 0
offpeak_arrival_rate = len(offpeak_data) / (total_days * 15)  # 15 off-peak hours

print(f"\nOff-Peak Period:")
print(f"  Off-Peak Trips: {len(offpeak_data)}")
print(f"  Average Waiting Time: {offpeak_data['WaitingTime'].mean():.2f} minutes")
print(f"  Arrival Rate (λ): {offpeak_arrival_rate:.3f} buses/hour")
print(f"  Service Rate (μ): {offpeak_service_rate:.3f} buses/hour")

# ============================================================================
# 2. M/M/c MODEL FUNCTIONS
# ============================================================================

def calculate_p0(lambda_rate, mu, c):
    """Calculate probability of 0 customers in system (P0)"""
    rho = lambda_rate / mu
    
    # Sum for n = 0 to c-1
    sum_term = sum((rho ** n) / factorial(n) for n in range(c))
    
    # Term for n >= c
    if rho < c:  # Stability condition
        last_term = (rho ** c) / (factorial(c) * (1 - rho / c))
    else:
        return 0  # System unstable
    
    p0 = 1 / (sum_term + last_term)
    return p0

def calculate_lq(lambda_rate, mu, c, p0):
    """Calculate average queue length (Lq)"""
    rho = lambda_rate / mu
    
    if rho >= c:  # System unstable
        return float('inf')
    
    numerator = (rho ** c) * rho * p0
    denominator = factorial(c) * ((1 - rho / c) ** 2)
    
    lq = numerator / denominator
    return lq

def calculate_wq(lq, lambda_rate):
    """Calculate average waiting time in queue (Wq)"""
    if lambda_rate == 0:
        return 0
    return lq / lambda_rate

def calculate_utilization(lambda_rate, mu, c):
    """Calculate system utilization"""
    return lambda_rate / (c * mu)

def analyze_mmc_system(lambda_rate, mu, c, label="System"):
    """Perform complete M/M/c analysis"""
    print(f"\n{label}:")
    print(f"  λ (arrival rate): {lambda_rate:.3f} buses/hour")
    print(f"  μ (service rate): {mu:.3f} buses/hour per server")
    print(f"  c (servers): {c}")
    
    rho = calculate_utilization(lambda_rate, mu, c)
    print(f"  ρ (utilization): {rho:.3f} ({rho*100:.1f}%)")
    
    if rho >= 1.0:
        print(f"  ⚠ WARNING: System is UNSTABLE (ρ ≥ 1.0)")
        print(f"  Queue will grow indefinitely!")
        return None
    
    p0 = calculate_p0(lambda_rate, mu, c)
    print(f"  P₀ (idle probability): {p0:.4f}")
    
    lq = calculate_lq(lambda_rate, mu, c, p0)
    print(f"  Lq (avg queue length): {lq:.2f} buses")
    
    wq = calculate_wq(lq, lambda_rate)
    print(f"  Wq (avg waiting time): {wq:.2f} hours = {wq*60:.2f} minutes")
    
    ls = lq + (lambda_rate / mu)
    print(f"  Ls (avg in system): {ls:.2f} buses")
    
    ws = wq + (1 / mu)
    print(f"  Ws (avg time in system): {ws:.2f} hours = {ws*60:.2f} minutes")
    
    return {
        'lambda': lambda_rate,
        'mu': mu,
        'c': c,
        'rho': rho,
        'P0': p0,
        'Lq': lq,
        'Wq': wq * 60,  # in minutes
        'Ls': ls,
        'Ws': ws * 60   # in minutes
    }

# ============================================================================
# 3. CURRENT SYSTEM ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("2. CURRENT SYSTEM ANALYSIS (4 Major Terminals)")
print("=" * 80)

# Using 4 terminals as servers
c_current = 4

# Overall analysis
current_overall = analyze_mmc_system(buses_per_hour, service_rate, c_current, 
                                     "Current System - Overall")

# Peak analysis
current_peak = analyze_mmc_system(peak_arrival_rate, peak_service_rate, c_current, 
                                  "Current System - Peak Period")

# Off-peak analysis
current_offpeak = analyze_mmc_system(offpeak_arrival_rate, offpeak_service_rate, c_current, 
                                     "Current System - Off-Peak Period")

# ============================================================================
# 4. SCENARIO ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("3. SCENARIO ANALYSIS - OPTIMIZATION STRATEGIES")
print("=" * 80)

scenarios = []

# Scenario 1: Add 1 terminal (5 terminals)
print("\n" + "-" * 80)
print("SCENARIO 1: Add 1 Additional Terminal (c = 5)")
print("-" * 80)
scenario1 = analyze_mmc_system(peak_arrival_rate, peak_service_rate, 5, 
                               "Peak Period with 5 Terminals")
if scenario1:
    scenarios.append(('Scenario 1: +1 Terminal', scenario1))

# Scenario 2: Improve process efficiency (20% faster service)
improved_service_rate = peak_service_rate * 1.2
print("\n" + "-" * 80)
print("SCENARIO 2: Process Improvement (+20% Service Rate)")
print("-" * 80)
scenario2 = analyze_mmc_system(peak_arrival_rate, improved_service_rate, c_current, 
                               "Peak with Improved Process")
if scenario2:
    scenarios.append(('Scenario 2: +20% Speed', scenario2))

# Scenario 3: Combined approach
print("\n" + "-" * 80)
print("SCENARIO 3: Combined Approach (5 terminals + 20% faster)")
print("-" * 80)
scenario3 = analyze_mmc_system(peak_arrival_rate, improved_service_rate, 5, 
                               "Peak with Combined Improvements")
if scenario3:
    scenarios.append(('Scenario 3: Combined', scenario3))

# Scenario 4: Reduce demand (through scheduling)
reduced_arrival = peak_arrival_rate * 0.8  # 20% demand reduction
print("\n" + "-" * 80)
print("SCENARIO 4: Demand Management (-20% Peak Arrivals)")
print("-" * 80)
scenario4 = analyze_mmc_system(reduced_arrival, peak_service_rate, c_current, 
                               "Peak with Demand Management")
if scenario4:
    scenarios.append(('Scenario 4: -20% Demand', scenario4))

# ============================================================================
# 5. COMPARISON VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("4. SCENARIO COMPARISON")
print("=" * 80)

if current_peak and scenarios:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Prepare data for comparison
    scenario_names = ['Current'] + [s[0] for s in scenarios]
    waiting_times = [current_peak['Wq']] + [s[1]['Wq'] for s in scenarios]
    queue_lengths = [current_peak['Lq']] + [s[1]['Lq'] for s in scenarios]
    utilizations = [current_peak['rho']*100] + [s[1]['rho']*100 for s in scenarios]
    
    # Chart 1: Waiting Time Comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(scenario_names)), waiting_times, 
                    color=['red'] + ['green']*len(scenarios), alpha=0.7, edgecolor='black')
    ax1.axhline(8, color='orange', linestyle='--', linewidth=2, label='Target: 8 min')
    ax1.set_xticks(range(len(scenario_names)))
    ax1.set_xticklabels(scenario_names, rotation=15, ha='right')
    ax1.set_ylabel('Average Waiting Time (minutes)', fontsize=11)
    ax1.set_title('Waiting Time Comparison', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Chart 2: Queue Length Comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(scenario_names)), queue_lengths, 
                    color=['red'] + ['blue']*len(scenarios), alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(scenario_names)))
    ax2.set_xticklabels(scenario_names, rotation=15, ha='right')
    ax2.set_ylabel('Average Queue Length (buses)', fontsize=11)
    ax2.set_title('Queue Length Comparison', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Chart 3: Utilization Comparison
    ax3 = axes[1, 0]
    bars3 = ax3.bar(range(len(scenario_names)), utilizations, 
                    color=['red'] + ['purple']*len(scenarios), alpha=0.7, edgecolor='black')
    ax3.axhline(80, color='green', linestyle='--', linewidth=2, label='Target: 70-80%')
    ax3.set_xticks(range(len(scenario_names)))
    ax3.set_xticklabels(scenario_names, rotation=15, ha='right')
    ax3.set_ylabel('Utilization (%)', fontsize=11)
    ax3.set_title('System Utilization Comparison', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
    
    # Chart 4: Improvement Percentage
    ax4 = axes[1, 1]
    if waiting_times[0] > 0:
        improvements = [(waiting_times[0] - wt) / waiting_times[0] * 100 
                       for wt in waiting_times[1:]]
        bars4 = ax4.bar(range(len(improvements)), improvements, 
                        color='teal', alpha=0.7, edgecolor='black')
        ax4.set_xticks(range(len(improvements)))
        ax4.set_xticklabels([s[0] for s in scenarios], rotation=15, ha='right')
        ax4.set_ylabel('Waiting Time Reduction (%)', fontsize=11)
        ax4.set_title('Improvement vs Current System', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.axhline(0, color='black', linewidth=1)
        
        for i, bar in enumerate(bars4):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
                     fontsize=9)
    
    plt.suptitle('Queuing Theory - Scenario Comparison (Peak Period)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('queuing_model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Comparison chart saved as 'queuing_model_comparison.png'")
    plt.show()

print("\n" + "=" * 80)
print("Queuing theory analysis complete!")
print("=" * 80)