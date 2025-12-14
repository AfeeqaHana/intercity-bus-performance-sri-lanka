import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Load data
df = pd.read_csv('busnetworkdata.csv')
df['Route'] = df['DepartureTerminal'] + ' → ' + df['ArrivalTerminal']

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))

# ============================================================================
# 1. Waiting Time Distribution
# ============================================================================
ax1 = plt.subplot(3, 3, 1)
sns.histplot(data=df, x='WaitingTime', bins=30, kde=True, ax=ax1, color='steelblue')
ax1.axvline(df['WaitingTime'].mean(), color='red', linestyle='--', 
            label=f'Mean: {df["WaitingTime"].mean():.2f} min')
ax1.axvline(5, color='green', linestyle='--', label='Target: 5 min')
ax1.set_xlabel('Waiting Time (minutes)', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Distribution of Waiting Time', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# ============================================================================
# 2. Peak vs Off-Peak Waiting Time
# ============================================================================
ax2 = plt.subplot(3, 3, 2)
peak_offpeak = df.groupby('PeakPeriod')['WaitingTime'].mean()
bars = ax2.bar(peak_offpeak.index, peak_offpeak.values, 
               color=['#e74c3c', '#3498db'], alpha=0.7, edgecolor='black')
ax2.axhline(5, color='green', linestyle='--', linewidth=2, label='Target: 5 min')
ax2.set_ylabel('Average Waiting Time (minutes)', fontsize=11)
ax2.set_title('Peak vs Off-Peak Waiting Time', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# ============================================================================
# 3. Delay Distribution
# ============================================================================
ax3 = plt.subplot(3, 3, 3)
sns.histplot(data=df, x='DelayMinutes', bins=30, kde=True, ax=ax3, color='coral')
ax3.axvline(df['DelayMinutes'].mean(), color='red', linestyle='--', 
            label=f'Mean: {df["DelayMinutes"].mean():.2f} min')
ax3.axvline(10, color='green', linestyle='--', label='Target: 10 min')
ax3.set_xlabel('Delay (minutes)', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=11)
ax3.set_title('Distribution of Delays', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# ============================================================================
# 4. Occupancy Distribution
# ============================================================================
ax4 = plt.subplot(3, 3, 4)
sns.histplot(data=df, x='OccupancyPercent', bins=30, kde=True, ax=ax4, color='mediumseagreen')
ax4.axvline(df['OccupancyPercent'].mean(), color='red', linestyle='--', 
            label=f'Mean: {df["OccupancyPercent"].mean():.2f}%')
ax4.axvline(85, color='green', linestyle='--', label='Target: 85%')
ax4.set_xlabel('Occupancy (%)', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title('Distribution of Bus Occupancy', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# ============================================================================
# 5. Route-wise Occupancy
# ============================================================================
ax5 = plt.subplot(3, 3, 5)
route_occupancy = df.groupby('Route')['OccupancyPercent'].mean().sort_values(ascending=True)
bars = ax5.barh(route_occupancy.index, route_occupancy.values, 
                color='teal', alpha=0.7, edgecolor='black')
ax5.axvline(85, color='green', linestyle='--', linewidth=2, label='Target: 85%')
ax5.set_xlabel('Average Occupancy (%)', fontsize=11)
ax5.set_title('Average Occupancy by Route', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax5.text(width, bar.get_y() + bar.get_height()/2., 
             f'{width:.1f}%', ha='left', va='center', fontsize=9)

# ============================================================================
# 6. Peak vs Off-Peak Delay
# ============================================================================
ax6 = plt.subplot(3, 3, 6)
peak_delay = df.groupby('PeakPeriod')['DelayMinutes'].mean()
bars = ax6.bar(peak_delay.index, peak_delay.values, 
               color=['#e67e22', '#16a085'], alpha=0.7, edgecolor='black')
ax6.axhline(10, color='green', linestyle='--', linewidth=2, label='Target: 10 min')
ax6.set_ylabel('Average Delay (minutes)', fontsize=11)
ax6.set_title('Peak vs Off-Peak Delay', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# ============================================================================
# 7. Waiting Time vs Occupancy Scatter
# ============================================================================
ax7 = plt.subplot(3, 3, 7)
scatter = ax7.scatter(df['OccupancyPercent'], df['WaitingTime'], 
                     c=df['DelayMinutes'], cmap='YlOrRd', 
                     alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
ax7.set_xlabel('Occupancy (%)', fontsize=11)
ax7.set_ylabel('Waiting Time (minutes)', fontsize=11)
ax7.set_title('Waiting Time vs Occupancy (colored by Delay)', 
              fontsize=12, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax7)
cbar.set_label('Delay (min)', fontsize=10)
ax7.grid(True, alpha=0.3)

# ============================================================================
# 8. Terminal-wise Performance
# ============================================================================
ax8 = plt.subplot(3, 3, 8)
terminal_wait = df.groupby('DepartureTerminal')['WaitingTime'].mean().sort_values(ascending=True)
bars = ax8.barh(terminal_wait.index, terminal_wait.values, 
                color='mediumpurple', alpha=0.7, edgecolor='black')
ax8.axvline(5, color='green', linestyle='--', linewidth=2, label='Target: 5 min')
ax8.set_xlabel('Average Waiting Time (minutes)', fontsize=11)
ax8.set_title('Waiting Time by Departure Terminal', fontsize=12, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3, axis='x')

for i, bar in enumerate(bars):
    width = bar.get_width()
    ax8.text(width, bar.get_y() + bar.get_height()/2., 
             f'{width:.2f}', ha='left', va='center', fontsize=9)

# ============================================================================
# 9. Correlation Heatmap
# ============================================================================
ax9 = plt.subplot(3, 3, 9)
corr_vars = ['WaitingTime', 'DelayMinutes', 'OccupancyPercent', 'Passengers']
corr_matrix = df[corr_vars].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, ax=ax9, cbar_kws={'shrink': 0.8},
            linewidths=1, linecolor='black')
ax9.set_title('Correlation Matrix', fontsize=12, fontweight='bold')

# Overall title
fig.suptitle('Sri Lanka Intercity Express Bus Network - Performance Analysis Dashboard', 
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig('bus_analysis_visualizations.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'bus_analysis_visualizations.png'")
plt.show()

# ============================================================================
# Additional Chart: Time Series Pattern (if needed)
# ============================================================================
fig2, axes = plt.subplots(2, 2, figsize=(16, 10))

# Chart 1: Daily Passengers
ax1 = axes[0, 0]
daily_passengers = df.groupby('Date')['Passengers'].sum()
ax1.plot(daily_passengers.index, daily_passengers.values, 
         marker='o', linewidth=2, markersize=8, color='dodgerblue')
ax1.set_xlabel('Date', fontsize=11)
ax1.set_ylabel('Total Passengers', fontsize=11)
ax1.set_title('Daily Passenger Volume', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Chart 2: Daily Average Occupancy
ax2 = axes[0, 1]
daily_occupancy = df.groupby('Date')['OccupancyPercent'].mean()
ax2.plot(daily_occupancy.index, daily_occupancy.values, 
         marker='s', linewidth=2, markersize=8, color='seagreen')
ax2.axhline(85, color='red', linestyle='--', label='Target: 85%')
ax2.set_xlabel('Date', fontsize=11)
ax2.set_ylabel('Average Occupancy (%)', fontsize=11)
ax2.set_title('Daily Average Occupancy', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# Chart 3: Box plot of waiting time by period
ax3 = axes[1, 0]
df.boxplot(column='WaitingTime', by='PeakPeriod', ax=ax3, 
           patch_artist=True, grid=True)
ax3.set_xlabel('Period', fontsize=11)
ax3.set_ylabel('Waiting Time (minutes)', fontsize=11)
ax3.set_title('Waiting Time Distribution by Period', fontsize=12, fontweight='bold')
plt.sca(ax3)
plt.xticks([1, 2], ['Off-Peak', 'Peak'])

# Chart 4: Route popularity
ax4 = axes[1, 1]
route_trips = df['Route'].value_counts()
ax4.pie(route_trips.values, labels=route_trips.index, autopct='%1.1f%%',
        startangle=90, colors=sns.color_palette('Set3'))
ax4.set_title('Trip Distribution by Route', fontsize=12, fontweight='bold')

fig2.suptitle('Sri Lanka Intercity Express Bus Network - Temporal & Distribution Analysis', 
              fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('bus_temporal_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Additional visualization saved as 'bus_temporal_analysis.png'")
plt.show()

print("\n" + "=" * 70)
print("All visualizations generated successfully!")
print("=" * 70)