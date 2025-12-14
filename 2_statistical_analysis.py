import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Load data
df = pd.read_csv('busnetworkdata.csv')

print("=" * 70)
print("STATISTICAL ANALYSIS - SRI LANKA INTERCITY EXPRESS BUS NETWORK")
print("=" * 70)

# ============================================================================
# 1. BASIC STATISTICS
# ============================================================================
print("\n" + "=" * 70)
print("1. BASIC STATISTICS")
print("=" * 70)

print("\nWaiting Time Statistics:")
print(f"  Mean: {df['WaitingTime'].mean():.2f} minutes")
print(f"  Std Dev: {df['WaitingTime'].std():.2f} minutes")
print(f"  Minimum: {df['WaitingTime'].min():.2f} minutes")
print(f"  Maximum: {df['WaitingTime'].max():.2f} minutes")
print(f"  Median: {df['WaitingTime'].median():.2f} minutes")

print("\nDelay Statistics:")
print(f"  Mean: {df['DelayMinutes'].mean():.2f} minutes")
print(f"  Std Dev: {df['DelayMinutes'].std():.2f} minutes")
print(f"  Minimum: {df['DelayMinutes'].min():.2f} minutes")
print(f"  Maximum: {df['DelayMinutes'].max():.2f} minutes")

print("\nOccupancy Statistics:")
print(f"  Mean: {df['OccupancyPercent'].mean():.2f}%")
print(f"  Std Dev: {df['OccupancyPercent'].std():.2f}%")
print(f"  Minimum: {df['OccupancyPercent'].min():.2f}%")
print(f"  Maximum: {df['OccupancyPercent'].max():.2f}%")

print("\nPassengers per Trip:")
print(f"  Mean: {df['Passengers'].mean():.2f}")
print(f"  Total Passengers (5 days): {df['Passengers'].sum()}")

# ============================================================================
# 2. PEAK vs OFF-PEAK ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("2. PEAK vs OFF-PEAK COMPARISON")
print("=" * 70)

peak_analysis = df.groupby('PeakPeriod').agg({
    'WaitingTime': ['mean', 'max', 'std'],
    'DelayMinutes': ['mean', 'max'],
    'OccupancyPercent': 'mean',
    'Passengers': 'mean',
    'TripID': 'count'
}).round(2)

print("\n", peak_analysis)

peak_data = df[df['PeakPeriod'] == 'Peak']
offpeak_data = df[df['PeakPeriod'] == 'Off-Peak']

print(f"\nPeak Period:")
print(f"  Average Waiting Time: {peak_data['WaitingTime'].mean():.2f} minutes")
print(f"  Average Delay: {peak_data['DelayMinutes'].mean():.2f} minutes")
print(f"  Average Occupancy: {peak_data['OccupancyPercent'].mean():.2f}%")

print(f"\nOff-Peak Period:")
print(f"  Average Waiting Time: {offpeak_data['WaitingTime'].mean():.2f} minutes")
print(f"  Average Delay: {offpeak_data['DelayMinutes'].mean():.2f} minutes")
print(f"  Average Occupancy: {offpeak_data['OccupancyPercent'].mean():.2f}%")

# Performance gap
wait_gap = ((peak_data['WaitingTime'].mean() - offpeak_data['WaitingTime'].mean()) / 
            offpeak_data['WaitingTime'].mean() * 100)
print(f"\nWaiting Time Increase in Peak: {wait_gap:.1f}%")

# ============================================================================
# 3. ROUTE ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("3. ROUTE-WISE ANALYSIS")
print("=" * 70)

df['Route'] = df['DepartureTerminal'] + ' → ' + df['ArrivalTerminal']

route_analysis = df.groupby('Route').agg({
    'Passengers': 'mean',
    'OccupancyPercent': 'mean',
    'DelayMinutes': 'mean',
    'WaitingTime': 'mean',
    'TripID': 'count'
}).round(2)

route_analysis.columns = ['Avg Passengers', 'Avg Occupancy %', 'Avg Delay (min)', 
                          'Avg Wait (min)', 'Trips']
print("\n", route_analysis.sort_values('Avg Occupancy %', ascending=False))

# ============================================================================
# 4. TERMINAL ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("4. TERMINAL-WISE ANALYSIS")
print("=" * 70)

terminal_analysis = df.groupby('DepartureTerminal').agg({
    'TripID': 'count',
    'Passengers': 'sum',
    'DelayMinutes': 'mean',
    'WaitingTime': 'mean'
}).round(2)

terminal_analysis.columns = ['Total Trips', 'Total Passengers', 
                             'Avg Delay (min)', 'Avg Waiting (min)']
print("\n", terminal_analysis.sort_values('Total Passengers', ascending=False))

# ============================================================================
# 5. PERFORMANCE METRICS EVALUATION
# ============================================================================
print("\n" + "=" * 70)
print("5. PERFORMANCE METRICS vs TARGETS")
print("=" * 70)

metrics = {
    'Average Waiting Time': {
        'Current': df['WaitingTime'].mean(),
        'Target': 5.0,
        'Unit': 'minutes'
    },
    'Peak Waiting Time': {
        'Current': peak_data['WaitingTime'].mean(),
        'Target': 8.0,
        'Unit': 'minutes'
    },
    'Average Occupancy': {
        'Current': df['OccupancyPercent'].mean(),
        'Target': 85.0,
        'Unit': '%'
    },
    'Average Delay': {
        'Current': df['DelayMinutes'].mean(),
        'Target': 10.0,
        'Unit': 'minutes'
    }
}

for metric, values in metrics.items():
    current = values['Current']
    target = values['Target']
    unit = values['Unit']
    
    if metric == 'Average Occupancy':
        gap = current - target
        status = "✓ MET" if current >= target else "✗ NOT MET"
    else:
        gap = current - target
        status = "✓ MET" if current <= target else "✗ NOT MET"
    
    print(f"\n{metric}:")
    print(f"  Current: {current:.2f} {unit}")
    print(f"  Target: {target:.2f} {unit}")
    print(f"  Gap: {gap:+.2f} {unit}")
    print(f"  Status: {status}")

# ============================================================================
# 6. CORRELATION ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("6. CORRELATION ANALYSIS")
print("=" * 70)

correlation_vars = ['WaitingTime', 'DelayMinutes', 'OccupancyPercent', 'Passengers']
corr_matrix = df[correlation_vars].corr()

print("\nCorrelation Matrix:")
print(corr_matrix.round(3))

print("\nKey Correlations:")
print(f"  Waiting Time ↔ Delay: {corr_matrix.loc['WaitingTime', 'DelayMinutes']:.3f}")
print(f"  Waiting Time ↔ Occupancy: {corr_matrix.loc['WaitingTime', 'OccupancyPercent']:.3f}")
print(f"  Delay ↔ Occupancy: {corr_matrix.loc['DelayMinutes', 'OccupancyPercent']:.3f}")

# ============================================================================
# 7. DISTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("7. DISTRIBUTION ANALYSIS")
print("=" * 70)

# Test for normality
stat_wait, p_wait = stats.shapiro(df['WaitingTime'].sample(min(5000, len(df))))
stat_delay, p_delay = stats.shapiro(df['DelayMinutes'].sample(min(5000, len(df))))

print("\nShapiro-Wilk Normality Test:")
print(f"  Waiting Time: statistic={stat_wait:.4f}, p-value={p_wait:.4f}")
print(f"  Delay: statistic={stat_delay:.4f}, p-value={p_delay:.4f}")

if p_wait < 0.05:
    print("  → Waiting Time is NOT normally distributed")
else:
    print("  → Waiting Time is approximately normally distributed")

# ============================================================================
# 8. ON-TIME PERFORMANCE
# ============================================================================
print("\n" + "=" * 70)
print("8. ON-TIME PERFORMANCE")
print("=" * 70)

# Buses with delay <= 5 minutes considered on-time
df['OnTime'] = df['DelayMinutes'] <= 5
ontime_rate = (df['OnTime'].sum() / len(df)) * 100

print(f"\nOn-Time Departure Rate: {ontime_rate:.2f}%")
print(f"Target: 95%")
print(f"Gap: {ontime_rate - 95:.2f}%")
print(f"Status: {'✓ MET' if ontime_rate >= 95 else '✗ NOT MET'}")

ontime_by_period = df.groupby('PeakPeriod')['OnTime'].apply(lambda x: (x.sum()/len(x)*100)).round(2)
print(f"\nOn-Time Rate by Period:")
print(ontime_by_period)

# Save summary to file
summary_output = f"""
STATISTICAL ANALYSIS SUMMARY
Sri Lanka Intercity Express Bus Network
{'=' * 70}

OVERALL PERFORMANCE:
- Total Trips Analyzed: {len(df)}
- Total Passengers: {df['Passengers'].sum():,}
- Average Waiting Time: {df['WaitingTime'].mean():.2f} minutes
- Average Delay: {df['DelayMinutes'].mean():.2f} minutes
- Average Occupancy: {df['OccupancyPercent'].mean():.2f}%
- On-Time Rate: {ontime_rate:.2f}%

PEAK vs OFF-PEAK:
- Peak Waiting Time: {peak_data['WaitingTime'].mean():.2f} minutes
- Off-Peak Waiting Time: {offpeak_data['WaitingTime'].mean():.2f} minutes
- Performance Gap: {wait_gap:.1f}% increase during peak

TARGET ACHIEVEMENT:
- Waiting Time Target (< 5 min): {'MET' if df['WaitingTime'].mean() <= 5 else 'NOT MET'}
- Occupancy Target (> 85%): {'MET' if df['OccupancyPercent'].mean() >= 85 else 'NOT MET'}
- Delay Target (< 10 min): {'MET' if df['DelayMinutes'].mean() <= 10 else 'NOT MET'}
- On-Time Target (95%): {'MET' if ontime_rate >= 95 else 'NOT MET'}
"""

with open('analysis_summary.txt', 'w') as f:
    f.write(summary_output)

print("\n" + "=" * 70)
print("Analysis complete! Summary saved to 'analysis_summary.txt'")
print("=" * 70)