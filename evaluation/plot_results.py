"""
CORTEX: Global Dataset Metrics Summarizer and Table Generator
Parses evaluation output JSON files, computes global Town05 averages, 
and generates IEEE Access compliant summary CSV tables.
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def calculate_rms(data_list):
    valid_data = [x for x in data_list if x is not None and not np.isnan(x) and not np.isinf(x)]
    if not valid_data: return np.nan
    return np.sqrt(np.mean(np.square(valid_data)))


def main():
    parser = argparse.ArgumentParser(description="CORTEX Evaluation Summary & Table Generator")
    parser.add_argument("--input_json", type=str, default="cortex_ultimate_ablation_matrix_results.json", help="Path to evaluation JSON results")
    parser.add_argument("--output_csv", type=str, default="summary_statistics_table_ALL_MODES.csv", help="Path to save summary CSV")
    args = parser.parse_args()

    if not os.path.exists(args.input_json):
        print(f"Error: Results file '{args.input_json}' not found.")
        return

    print(f"Loading matrix results from {args.input_json}...")
    with open(args.input_json, "r", encoding="utf-8") as f:
        matrix_data = json.load(f)

    df = pd.DataFrame(matrix_data)
    print(f"Successfully loaded {len(df)} logged frame evaluation records.")

    modes = ["Ego-only", "V2I-Fixed-Sync", "CORTEX-Ultimate"]
    summary_data = []

    for mode in modes:
        mode_df = df[df['evaluation_mode'] == mode]
        if mode_df.empty: continue

        ades, fdes = [], []
        for _, row in mode_df.iterrows():
            gt_path = row.get("ground_truth", {}).get("gt_path_xy", [])
            pred_path = row.get("prediction", {}).get("pred_path_xy", [])
            if isinstance(gt_path, list) and isinstance(pred_path, list) and len(gt_path) > 0 and len(pred_path) > 0:
                gt_np, pred_np = np.array(gt_path), np.array(pred_path)
                min_len = min(len(gt_np), len(pred_np))
                dists = np.linalg.norm(gt_np[:min_len] - pred_np[:min_len], axis=1)
                ades.append(np.mean(dists))
                fdes.append(dists[-1])

        summary_data.append({
            "Evaluation Mode": mode,
            "Global ADE (m)": np.nanmean(ades) if ades else np.nan,
            "Global FDE (m)": np.nanmean(fdes) if fdes else np.nan,
            "Total Processed Frames": len(mode_df)
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(args.output_csv, index=False, float_format="%.4f")

    print("\n" + "="*70)
    print(" CORTEX GLOBAL TOWN05 METRICS SUMMARY TABLE")
    print("="*70)
    print(summary_df.to_string(index=False))
    print(f"\nSummary table written to: {args.output_csv}")


if __name__ == "__main__":
    main()