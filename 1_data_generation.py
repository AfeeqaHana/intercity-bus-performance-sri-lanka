import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# System Parameters
NUM_DAYS = 5
BUSES_PER_DAY = 100  # Total trips per day across all routes
TOTAL_TRIPS = NUM_DAYS * BUSES_PER_DAY  # 500 trips
BUS_CAPACITY = 45

# Define routes
routes = [
    ("Colombo", "Kandy"),
    ("Colombo", "Galle"),
    ("Colombo", "Jaffna"),
    ("Kandy", "Jaffna"),
    ("Galle", "Kandy")
]

# Time periods
PEAK_HOURS = [(7, 8), (17, 18)]  # 7-8 AM and 5-6 PM
OFF_PEAK_HOURS = [(5, 7), (8, 17), (18, 22)]  # Rest of operating hours

def is_peak_time(hour):
    """Check if given hour is peak time"""
    for start, end in PEAK_HOURS:
        if start <= hour < end:
            return True
    return False

def generate_scheduled_time(day, trip_num):
    """Generate scheduled departure time"""
    # Spread trips across operating hours (5 AM to 10 PM = 17 hours)
    base_hour = 5
    hour_increment = trip_num % 17
    hour = base_hour + hour_increment
    minute = random.choice([0, 15, 30, 45])
    
    scheduled_time = datetime(2025, 10, 27 + day, hour, minute, 0)
    return scheduled_time

def generate_delay(is_peak):
    """Generate delay based on peak/off-peak period"""
    if is_peak:
        # Peak: higher delays (mean=12.3 min, range 5-25 min)
        delay = np.random.gamma(shape=3, scale=4) + 5
        delay = min(delay, 25)  # Cap at 25 minutes
    else:
        # Off-peak: lower delays (mean=3.8 min, range 0-10 min)
        delay = np.random.gamma(shape=2, scale=1.5)
        delay = min(delay, 10)  # Cap at 10 minutes
    
    return round(delay, 2)

def generate_passengers(route, is_peak, bus_capacity):
    """Generate number of passengers based on route popularity and time"""
    # Route popularity factors
    route_factors = {
        ("Colombo", "Kandy"): 1.2,  # Most popular
        ("Colombo", "Galle"): 1.1,
        ("Colombo", "Jaffna"): 0.9,
        ("Kandy", "Jaffna"): 0.7,
        ("Galle", "Kandy"): 0.8
    }
    
    base_factor = route_factors.get(route, 1.0)
    
    if is_peak:
        # Peak: 75-95% occupancy
        mean_occupancy = 0.85 * base_factor
    else:
        # Off-peak: 40-70% occupancy
        mean_occupancy = 0.55 * base_factor
    
    # Add randomness
    occupancy = np.random.normal(mean_occupancy, 0.1)
    occupancy = max(0.3, min(occupancy, 1.0))  # Keep between 30% and 100%
    
    passengers = int(occupancy * bus_capacity)
    return min(passengers, bus_capacity)

def generate_waiting_time(delay, passengers, bus_capacity):
    """Generate waiting time (time passengers wait at terminal)"""
    # Base waiting time from delay
    base_wait = delay * 0.6
    
    # Add factor based on terminal congestion (passenger load)
    occupancy_factor = (passengers / bus_capacity) * 5
    
    waiting_time = base_wait + occupancy_factor + np.random.normal(0, 2)
    waiting_time = max(1, waiting_time)  # Minimum 1 minute
    
    return round(waiting_time, 2)

# Generate dataset
data = []
trip_id = 1

for day in range(NUM_DAYS):
    trips_today = BUSES_PER_DAY
    
    for trip in range(trips_today):
        # Select route
        route = random.choice(routes)
        departure_terminal = route[0]
        arrival_terminal = route[1]
        
        # Generate scheduled time
        scheduled_time = generate_scheduled_time(day, trip)
        hour = scheduled_time.hour
        is_peak = is_peak_time(hour)
        
        # Determine peak period label
        peak_period = "Peak" if is_peak else "Off-Peak"
        
        # Generate delay
        delay = generate_delay(is_peak)
        
        # Calculate actual departure
        actual_time = scheduled_time + timedelta(minutes=delay)
        
        # Generate passengers
        passengers = generate_passengers(route, is_peak, BUS_CAPACITY)
        
        # Generate waiting time
        waiting_time = generate_waiting_time(delay, passengers, BUS_CAPACITY)
        
        # Calculate occupancy percentage
        occupancy_pct = round((passengers / BUS_CAPACITY) * 100, 2)
        
        # Create record
        record = {
            'TripID': trip_id,
            'Date': scheduled_time.date(),
            'DepartureTerminal': departure_terminal,
            'ArrivalTerminal': arrival_terminal,
            'ScheduledDeparture': scheduled_time.strftime('%H:%M:%S'),
            'ActualDeparture': actual_time.strftime('%H:%M:%S'),
            'Passengers': passengers,
            'BusCapacity': BUS_CAPACITY,
            'OccupancyPercent': occupancy_pct,
            'DelayMinutes': delay,
            'WaitingTime': waiting_time,
            'PeakPeriod': peak_period
        }
        
        data.append(record)
        trip_id += 1

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('busnetworkdata.csv', index=False)

print("=" * 70)
print("DATA GENERATION COMPLETE")
print("=" * 70)
print(f"\nTotal Records Generated: {len(df)}")
print(f"\nDataset Preview:")
print(df.head(10))
print(f"\n\nBasic Statistics:")
print(df[['Passengers', 'OccupancyPercent', 'DelayMinutes', 'WaitingTime']].describe())
print(f"\n\nPeak vs Off-Peak Distribution:")
print(df['PeakPeriod'].value_counts())
print(f"\n\nRoute Distribution:")
route_dist = df.groupby(['DepartureTerminal', 'ArrivalTerminal']).size()
print(route_dist)
print("\n" + "=" * 70)
print("Dataset saved as 'busnetworkdata.csv'")
print("=" * 70)