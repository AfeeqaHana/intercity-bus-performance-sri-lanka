# Sri Lanka Intercity Express Bus Network - Performance Analysis

**EEX5362 Performance Modelling Mini Project**

**Student:** M.M.A. Hana  
**Registration No:** 621444830  
**SNO:** s92084830  
**Course:** Bachelor of Software Engineering  
**Institution:** The Open University of Sri Lanka

---

## 📋 Project Overview

This project analyzes the performance of Sri Lanka's Intercity Express Bus Network using:
- Statistical data analysis
- Queuing theory (M/M/c model)
- Discrete event simulation (SimPy)
- Data visualization

---

## 🔧 Installation

### Prerequisites

Ensure you have Python 3.8 or higher installed on your system.

### Required Libraries

Install all required packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scipy simpy
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

### Requirements.txt Content:
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
simpy>=4.0.0
```

---

## 📁 Project Structure

```
bus-network-analysis/
│
├── 1_data_generation.py          # Generates synthetic bus network data
├── 2_statistical_analysis.py     # Performs statistical analysis
├── 3_visualization.py             # Creates visualization charts
├── 4_queuing_theory.py            # Queuing theory analysis
├── 5_simulation.py                # SimPy discrete event simulation
├── master_script.py               # Runs all scripts in sequence
│
├── busnetworkdata.csv             # Generated dataset (after running)
├── analysis_summary.txt           # Analysis summary (after running)
│
├── bus_analysis_visualizations.png    # Main dashboard
├── bus_temporal_analysis.png          # Time series analysis
├── queuing_model_comparison.png       # Queuing theory results
├── simulation_results.png             # Simulation comparison
│
└── README.md                      # This file
```

---

## 🚀 Usage

### Option 1: Run Complete Analysis (Recommended)

Run the master script to execute all analysis steps automatically:

```bash
python master_script.py
```

This will:
1. Generate the dataset (500 trips over 5 days)
2. Perform statistical analysis
3. Create all visualizations
4. Run queuing theory calculations
5. Execute SimPy simulations
6. Generate summary reports

**Total Runtime:** Approximately 2-3 minutes

---

### Option 2: Run Individual Scripts

Run scripts individually in order:

```bash
# Step 1: Generate data
python 1_data_generation.py

# Step 2: Statistical analysis
python 2_statistical_analysis.py

# Step 3: Create visualizations
python 3_visualization.py

# Step 4: Queuing theory
python 4_queuing_theory.py

# Step 5: Simulation
python 5_simulation.py
```

---

## 📊 Output Files

### Data Files

1. **busnetworkdata.csv**
   - 500 trip records
   - Variables: TripID, Date, Terminals, Times, Passengers, Delays, etc.

2. **analysis_summary.txt**
   - Overall performance metrics
   - Peak vs off-peak comparison
   - Target achievement status

3. **simulation_results.csv**
   - Scenario comparison results
   - Statistical confidence intervals

### Visualization Files

1. **bus_analysis_visualizations.png**
   - 9-panel dashboard
   - Waiting time, occupancy, delays
   - Route and terminal analysis
   - Correlation heatmap

2. **bus_temporal_analysis.png**
   - Daily trends
   - Time series patterns
   - Distribution analysis

3. **queuing_model_comparison.png**
   - Scenario comparison charts
   - Waiting time improvements
   - Utilization analysis

4. **simulation_results.png**
   - SimPy simulation results
   - Confidence intervals
   - Throughput comparison

---

## 📈 Key Performance Metrics

The analysis evaluates:

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Average Waiting Time | ~12.3 min | < 5 min | ❌ Not Met |
| Peak Waiting Time | ~15.1 min | < 8 min | ❌ Not Met |
| Average Occupancy | ~65% | > 85% | ❌ Not Met |
| Average Delay | ~8.5 min | < 10 min | ✅ Met |
| On-Time Departure | ~60% | 95% | ❌ Not Met |

---

## 🔬 Analysis Methods

### 1. Statistical Analysis
- Descriptive statistics
- Peak vs off-peak comparison
- Route-wise performance
- Correlation analysis
- Distribution testing

### 2. Queuing Theory (M/M/c)
- 4 terminal servers
- Poisson arrival process
- Exponential service times
- Utilization and performance metrics
- Scenario optimization

### 3. Discrete Event Simulation
- SimPy framework
- 10 replications per scenario
- 95% confidence intervals
- Multiple improvement scenarios

---

## 🎯 Scenarios Analyzed

1. **Current System:** 4 terminals, baseline performance
2. **Scenario 1:** Add 1 terminal (5 total)
3. **Scenario 2:** Process improvement (+20% service speed)
4. **Scenario 3:** Combined approach (5 terminals + 20% speed)
5. **Scenario 4:** Demand management (-20% peak arrivals)

---

## 📝 System Parameters

- **Terminals:** 4 major city hubs (Colombo, Kandy, Galle, Jaffna)
- **Operating Hours:** 5:00 AM - 10:00 PM (17 hours)
- **Fleet Size:** 150 express buses
- **Bus Capacity:** 45 seats per bus
- **Peak Hours:** 7-8 AM, 5-6 PM
- **Routes:** 5 intercity routes

---

## 🐛 Troubleshooting

### Common Issues

**Issue 1: Module not found error**
```
Solution: pip install <missing_module>
```

**Issue 2: File not found**
```
Solution: Ensure you run scripts in correct order, or use master_script.py
```

**Issue 3: Permission denied**
```
Solution: Run terminal/cmd as administrator
```

**Issue 4: Plots not showing**
```
Solution: Check if matplotlib backend is configured correctly
Add to script: import matplotlib; matplotlib.use('TkAgg')
```

---

## 📚 References

1. Jain, R. (1991). *The Art of Computer Systems Performance Analysis*
2. Kleinrock, L. (1975). *Queueing Systems Volume 1: Theory*
3. Gross, D. et al. (2008). *Fundamentals of Queueing Theory*
4. SimPy Documentation: https://simpy.readthedocs.io
5. Sri Lanka Transport Board: http://www.sltb.lk

---

## 📧 Contact

For questions or issues:

**Student:** M.M.A. Hana  
**Email:** [Your Email]  
**Registration No:** 621444830

---

## 📄 License

This project is submitted as part of academic coursework for EEX5362 Performance Modelling at The Open University of Sri Lanka.

---

## ✅ Checklist

- [x] Data generation script
- [x] Statistical analysis
- [x] Visualization generation
- [x] Queuing theory implementation
- [x] SimPy simulation
- [x] Documentation
- [ ] Final report compilation
- [ ] Presentation preparation

---

**Last Updated:** December 2024
