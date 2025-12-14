"""
Master Script for Bus Network Performance Analysis
Sri Lanka Intercity Express Bus Network
EEX5362 Performance Modelling Mini Project

This script runs the complete analysis pipeline:
1. Data Generation
2. Statistical Analysis
3. Visualizations
4. Queuing Theory Analysis
5. SimPy Simulation

Author: M.M.A. Hana
Registration No: 621444830
"""

import subprocess
import sys
import os
from datetime import datetime

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print_header(f"STEP: {description}")
    print(f"Running: {script_name}\n")
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, 
                              text=True, 
                              check=True)
        print(f"\n✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running {script_name}")
        print(f"Error: {e}")
        return False
    except FileNotFoundError:
        print(f"\n✗ File not found: {script_name}")
        print(f"Please ensure all script files are in the same directory.")
        return False

def main():
    """Main execution function"""
    start_time = datetime.now()
    
    print_header("SRI LANKA INTERCITY EXPRESS BUS NETWORK")
    print_header("PERFORMANCE MODELLING & ANALYSIS")
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Define analysis pipeline
    pipeline = [
        ("1_data_generation.py", "Data Generation"),
        ("2_statistical_analysis.py", "Statistical Analysis"),
        ("3_visualization.py", "Data Visualization"),
        ("4_queuing_theory.py", "Queuing Theory Analysis"),
        ("5_simulation.py", "SimPy Simulation")
    ]
    
    # Track results
    results = []
    
    # Execute pipeline
    for script, description in pipeline:
        success = run_script(script, description)
        results.append((description, success))
        
        if not success:
            print(f"\n⚠ Warning: {description} failed!")
            user_input = input("Continue with remaining steps? (y/n): ")
            if user_input.lower() != 'y':
                break
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print_header("ANALYSIS COMPLETE")
    print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Duration: {duration}\n")
    
    print("Summary of Results:")
    print("-" * 80)
    for description, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {description:40s} {status}")
    print("-" * 80)
    
    # List output files
    print("\nGenerated Files:")
    print("-" * 80)
    output_files = [
        "busnetworkdata.csv",
        "analysis_summary.txt",
        "bus_analysis_visualizations.png",
        "bus_temporal_analysis.png",
        "queuing_model_comparison.png",
        "simulation_results.png",
        "simulation_results.csv"
    ]
    
    for file in output_files:
        if os.path.exists(file):
            file_size = os.path.getsize(file) / 1024  # KB
            print(f"  ✓ {file:40s} ({file_size:.1f} KB)")
        else:
            print(f"  ✗ {file:40s} (Not found)")
    
    print("-" * 80)
    print("\nAll analysis complete! You can now:")
    print("  1. Review the generated CSV data file")
    print("  2. Examine the visualization PNG files")
    print("  3. Check analysis_summary.txt for key metrics")
    print("  4. Use the results for your final report")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()