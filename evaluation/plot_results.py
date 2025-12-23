# evaluation/plot_results.py
# Official Plotting & Metrics Calculation Script for CORTEX
# Generates publication-ready plots (Fig. 7) and quantitative tables (Table II).

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
from pathlib import Path

# --- Constants ---
DT = 0.2            # Time step (seconds)
WHEELBASE = 2.9     # Approximate wheelbase for a sedan (e.g., Lincoln MKZ in CARLA)
STEER_RATIO = 16.0  # Steering gear ratio (Steering Wheel Angle / Wheel Angle)
# Note: CARLA steer is normalized [-1, 1]. Max wheel angle is approx 70 degrees (1.22 rad) in some configs, 
# or standard 45 deg. We assume -1..1 maps to -45..45 degrees for physical estimation.
MAX_STEER_DEG = 45.0 

def smooth_signal(signal, window=11, poly=3):
    """Apply Savitzky-Golay filter for noise reduction in plots."""
    if len(signal) < window: return signal
    return savgol_filter(signal, window, poly)

def compute_dynamics(data_list):
    """
    Compute physical metrics from raw control signals.
    Returns dictionaries of numpy arrays.
    """
    # Convert list of dicts to DataFrame
    df = pd.DataFrame(data_list)
    
    # 1. Extract Raw Signals
    # Steer is in [-1, 1], convert to radians at the wheel
    steer_norm = df['pred_steer'].values
    steer_rad = steer_norm * np.radians(MAX_STEER_DEG)
    
    # GT Steer
    gt_steer_norm = df['gt_steer'].values
    gt_steer_rad = gt_steer_norm * np.radians(MAX_STEER_DEG)
    
    # Speed (m/s) - Use Ground Truth speed for stability analysis of the *prediction*
    # (We want to know: "If the car moves at this speed, how unstable is the steering signal?")
    speed = df['speed'].values if 'speed' in df.columns else np.zeros_like(steer_norm)
    
    # 2. Compute Yaw Rate (deg/s) using Bicycle Model
    # Yaw Rate = (Velocity / Wheelbase) * tan(SteeringAngle)
    yaw_rate_pred = (speed / WHEELBASE) * np.tan(steer_rad)
    yaw_rate_gt = (speed / WHEELBASE) * np.tan(gt_steer_rad)
    
    # Convert to deg/s
    yaw_rate_pred_deg = np.degrees(yaw_rate_pred)
    yaw_rate_gt_deg = np.degrees(yaw_rate_gt)
    
    # 3. Compute Lateral Acceleration (m/s^2)
    # a_lat = v * yaw_rate
    lat_accel_pred = speed * yaw_rate_pred
    lat_accel_gt = speed * yaw_rate_gt
    
    # 4. Compute Jerk (Derivative of Acceleration)
    # Here we approximate longitudinal jerk or lateral jerk. 
    # Let's compute Lateral Jerk (change in lat accel)
    lat_jerk_pred = np.gradient(lat_accel_pred, DT)
    lat_jerk_gt = np.gradient(lat_accel_gt, DT)

    return {
        "steer": steer_norm,
        "yaw_rate": yaw_rate_pred_deg,
        "lat_accel": lat_accel_pred,
        "lat_jerk": lat_jerk_pred
    }, {
        "steer": gt_steer_norm,
        "yaw_rate": yaw_rate_gt_deg,
        "lat_accel": lat_accel_gt,
        "lat_jerk": lat_jerk_gt
    }

def calculate_rms(signal):
    return np.sqrt(np.mean(signal**2))

def plot_comparisons(cortex_data, baseline_data, gt_data, output_dir):
    """Generate and save plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    time_axis = np.arange(len(cortex_data['steer'])) * DT
    
    metrics = [
        ("Steering Signal", "steer", "Normalized [-1, 1]"),
        ("Yaw Rate", "yaw_rate", "deg/s"),
        ("Lateral Acceleration", "lat_accel", "m/s^2")
    ]
    
    for title, key, unit in metrics:
        plt.figure(figsize=(10, 5))
        
        # Plot GT (Green)
        plt.plot(time_axis, smooth_signal(gt_data[key]), 'g-', label='Human Expert (GT)', linewidth=2, alpha=0.7)
        
        # Plot Baseline (Blue - Dashed/Noisy)
        plt.plot(time_axis, smooth_signal(baseline_data[key]), 'b--', label='TCP (Baseline)', linewidth=1.5, alpha=0.8)
        
        # Plot CORTEX (Red - Solid)
        plt.plot(time_axis, smooth_signal(cortex_data[key]), 'r-', label='CORTEX (Ours)', linewidth=2)
        
        plt.title(f"{title} Comparison")
        plt.xlabel("Time (s)")
        plt.ylabel(f"{title} ({unit})")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # Save
        filename = f"comparison_{key}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()
        print(f"Saved plot: {filename}")

def main(args):
    print("--- Loading Data ---")
    with open(args.cortex_json, 'r') as f:
        cortex_raw = json.load(f)
    with open(args.baseline_json, 'r') as f:
        baseline_raw = json.load(f)
        
    # Ensure lengths match (truncate to minimum)
    min_len = min(len(cortex_raw), len(baseline_raw))
    cortex_raw = cortex_raw[:min_len]
    baseline_raw = baseline_raw[:min_len]
    
    print(f"Processing {min_len} frames...")
    
    # Compute Physics
    cortex_dyn, gt_dyn = compute_dynamics(cortex_raw)
    baseline_dyn, _ = compute_dynamics(baseline_raw) # GT is same for both
    
    # --- Metrics Calculation (Table II) ---
    print("\n" + "="*40)
    print("TABLE II: CONTROL SIGNAL CONSISTENCY")
    print("="*40)
    
    # Metric: RMS Yaw Rate
    rms_yaw_cortex = calculate_rms(cortex_dyn['yaw_rate'] - gt_dyn['yaw_rate']) # Error relative to GT? Or absolute stability?
    # Paper claims reduction in oscillation (absolute RMS of the signal noise or Error RMSE).
    # Usually "Stability" implies lower high-frequency noise. 
    # Let's calculate RMSE against GT (Accuracy) and raw RMS (Stability/Energy).
    
    rmse_yaw_cortex = np.sqrt(np.mean((cortex_dyn['yaw_rate'] - gt_dyn['yaw_rate'])**2))
    rmse_yaw_baseline = np.sqrt(np.mean((baseline_dyn['yaw_rate'] - gt_dyn['yaw_rate'])**2))
    
    improvement = ((rmse_yaw_baseline - rmse_yaw_cortex) / rmse_yaw_baseline) * 100
    
    print(f"RMS Yaw Rate Error (deg/s):")
    print(f"  Baseline: {rmse_yaw_baseline:.2f}")
    print(f"  CORTEX:   {rmse_yaw_cortex:.2f}")
    print(f"  Improvement: {improvement:.1f}%")
    
    # Metric: Lateral Acceleration
    rmse_lat_cortex = np.sqrt(np.mean((cortex_dyn['lat_accel'] - gt_dyn['lat_accel'])**2))
    rmse_lat_baseline = np.sqrt(np.mean((baseline_dyn['lat_accel'] - gt_dyn['lat_accel'])**2))
    
    print(f"\nRMS Lat Accel Error (m/s^2):")
    print(f"  Baseline: {rmse_lat_baseline:.2f}")
    print(f"  CORTEX:   {rmse_lat_cortex:.2f}")
    
    # --- Plotting ---
    print("\nGenerating Plots...")
    plot_comparisons(cortex_dyn, baseline_dyn, gt_dyn, args.output_dir)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cortex_json", type=str, required=True, help="Path to cortex_eval_results.json")
    parser.add_argument("--baseline_json", type=str, required=True, help="Path to tcp_eval_results.json")
    parser.add_argument("--output_dir", type=str, default="./plots", help="Directory to save plots")
    
    args = parser.parse_args()
    main(args)