#!/usr/bin/env python

"""
Script to analyze trigger prescale factors from JSON correction files.

This script reads the prescale JSON files referenced in triggers_prescales_run3.yaml,
loops over run numbers and lumi sections, and calculates average prescale factors
averaged over different luminosity sections.
"""

import os
import json
import yaml
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

def load_yaml_config(config_path):
    """Load the YAML configuration file with prescales."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def resolve_config_path(path_string, config_dir):
    """Resolve ${config_dir:} placeholders in file paths."""
    if "${config_dir:}" in path_string:
        return path_string.replace("${config_dir:}", config_dir)
    return path_string

def load_prescale_json(json_path):
    """Load and parse a prescale JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def parse_prescale_data(data):
    """Parse prescale data to extract run numbers, lumi sections, and weights."""
    prescale_info = []
    
    # Navigate the JSON structure
    corrections = data.get("corrections", [])
    for correction in corrections:
        if correction.get("name") == "prescaleWeight":
            content = correction.get("data", {}).get("content", [])
            
            for run_entry in content:
                run_number = run_entry["key"]
                path_content = run_entry["value"]["content"]
                
                for path_entry in path_content:
                    hlt_path = path_entry["key"]
                    weight_data = path_entry["value"]
                    
                    # Handle different weight data structures
                    if isinstance(weight_data, (int, float)):
                        # Simple constant weight
                        prescale_info.append({
                            'run': run_number,
                            'hlt_path': hlt_path,
                            'lumi_start': 1,
                            'lumi_end': float('inf'),
                            'weight': float(weight_data)
                        })
                    elif isinstance(weight_data, dict) and weight_data.get("nodetype") == "binning":
                        # Binned weights by lumi section
                        edges = weight_data["edges"]
                        content = weight_data["content"]
                        
                        for i, weight in enumerate(content):
                            lumi_start = edges[i]
                            lumi_end = edges[i+1] if i+1 < len(edges) else float('inf')
                            
                            prescale_info.append({
                                'run': run_number,
                                'hlt_path': hlt_path,
                                'lumi_start': lumi_start,
                                'lumi_end': lumi_end,
                                'weight': float(weight)
                            })
    
    return prescale_info

def calculate_averages(prescale_data):
    """Calculate various averages of prescale factors."""
    df = pd.DataFrame(prescale_data)
    
    results = {}
    
    # Overall average by HLT path
    path_averages = df.groupby('hlt_path')['weight'].agg(['mean', 'std', 'count']).round(4)
    results['by_hlt_path'] = path_averages
    
    # Average by run number
    run_averages = df.groupby('run')['weight'].agg(['mean', 'std', 'count']).round(4)
    results['by_run'] = run_averages
    
    # Average by run and HLT path
    run_path_averages = df.groupby(['run', 'hlt_path'])['weight'].agg(['mean', 'std', 'count']).round(4)
    results['by_run_and_path'] = run_path_averages
    
    # Overall statistics
    overall_stats = {
        'total_entries': len(df),
        'unique_runs': df['run'].nunique(),
        'unique_hlt_paths': df['hlt_path'].nunique(),
        'overall_mean': df['weight'].mean(),
        'overall_std': df['weight'].std(),
        'min_weight': df['weight'].min(),
        'max_weight': df['weight'].max(),
        'zero_weight_fraction': (df['weight'] == 0).mean()
    }
    results['overall'] = overall_stats
    
    return results, df

def print_summary(results):
    """Print a summary of the results."""
    print("="*80)
    print("TRIGGER PRESCALE ANALYSIS SUMMARY")
    print("="*80)
    
    # Overall statistics
    overall = results['overall']
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total entries: {overall['total_entries']}")
    print(f"  Unique runs: {overall['unique_runs']}")
    print(f"  Unique HLT paths: {overall['unique_hlt_paths']}")
    print(f"  Overall mean prescale: {overall['overall_mean']:.4f}")
    print(f"  Overall std prescale: {overall['overall_std']:.4f}")
    print(f"  Min prescale: {overall['min_weight']:.4f}")
    print(f"  Max prescale: {overall['max_weight']:.4f}")
    print(f"  Fraction with zero prescale: {overall['zero_weight_fraction']:.4f}")
    
    # By HLT path
    print(f"\nAVERAGE PRESCALES BY HLT PATH:")
    print("-" * 60)
    by_path = results['by_hlt_path']
    for path, stats in by_path.iterrows():
        print(f"  {path}:")
        print(f"    Mean: {stats['mean']:.4f} ± {stats['std']:.4f} ({stats['count']} entries)")
    
    # Top 10 runs with highest average prescales
    print(f"\nTOP 10 RUNS WITH HIGHEST AVERAGE PRESCALES:")
    print("-" * 60)
    top_runs = results['by_run'].nlargest(10, 'mean')
    for run, stats in top_runs.iterrows():
        print(f"  Run {run}: {stats['mean']:.4f} ± {stats['std']:.4f} ({stats['count']} entries)")
    
    # Runs with zero prescales
    zero_runs = results['by_run'][results['by_run']['mean'] == 0]
    if len(zero_runs) > 0:
        print(f"\nRUNS WITH ZERO AVERAGE PRESCALES:")
        print("-" * 60)
        for run, stats in zero_runs.iterrows():
            print(f"  Run {run}: {stats['count']} entries")

def save_detailed_results(results, df, output_dir):
    """Save detailed results to CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save raw data
    df.to_csv(output_dir / "prescale_raw_data.csv", index=False)
    
    # Save averages
    results['by_hlt_path'].to_csv(output_dir / "averages_by_hlt_path.csv")
    results['by_run'].to_csv(output_dir / "averages_by_run.csv") 
    results['by_run_and_path'].to_csv(output_dir / "averages_by_run_and_path.csv")
    
    # Save overall statistics
    with open(output_dir / "overall_statistics.json", 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        overall_json = {k: float(v) if isinstance(v, np.floating) else 
                       int(v) if isinstance(v, np.integer) else v 
                       for k, v in results['overall'].items()}
        json.dump(overall_json, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Analyze trigger prescale factors")
    parser.add_argument("--config", default="mutag_calib/configs/params/triggers_prescales_2024.yaml",
                       help="Path to triggers prescales YAML config file")
    parser.add_argument("--output-dir", "-o", default="prescale_analysis",
                       help="Output directory for detailed results")
    parser.add_argument("--year", default=None,
                       help="Specific year to analyze (e.g., '2022_preEE'). If not specified, analyze all years.")
    parser.add_argument("--trigger-group", default=None, 
                       help="Specific trigger group to analyze (e.g., 'BTagMu'). If not specified, analyze all groups.")
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file {config_path} does not exist")
        return
    
    config = load_yaml_config(config_path)
    config_dir = config_path.parent
    
    # Extract prescale information
    hlt_triggers_prescales = config.get("HLT_triggers_prescales", {})
    
    all_prescale_data = []
    processed_files = set()
    
    print(f"Processing prescale files...")
    
    for year, year_data in hlt_triggers_prescales.items():
        if args.year and year != args.year:
            continue
            
        print(f"\nProcessing year: {year}")
        
        for trigger_group, triggers in year_data.items():
            if args.trigger_group and trigger_group != args.trigger_group:
                continue
                
            print(f"  Processing trigger group: {trigger_group}")
            
            for trigger_name, json_path in triggers.items():
                resolved_path = resolve_config_path(json_path, str(config_dir))
                json_file = Path(resolved_path)
                
                if not json_file.exists():
                    print(f"    Warning: JSON file not found: {json_file}")
                    continue
                
                # Avoid processing the same file multiple times
                if str(json_file) in processed_files:
                    continue
                processed_files.add(str(json_file))
                
                print(f"    Processing: {trigger_name} -> {json_file.name}")
                
                try:
                    prescale_json = load_prescale_json(json_file)
                    prescale_info = parse_prescale_data(prescale_json)
                    
                    # Add metadata
                    for entry in prescale_info:
                        entry['year'] = year
                        entry['trigger_group'] = trigger_group
                        entry['trigger_name'] = trigger_name
                        entry['json_file'] = json_file.name
                    
                    all_prescale_data.extend(prescale_info)
                    print(f"      Found {len(prescale_info)} prescale entries")
                    
                except Exception as e:
                    print(f"      Error processing {json_file}: {e}")
    
    if not all_prescale_data:
        print("No prescale data found!")
        return
    
    print(f"\nTotal prescale entries collected: {len(all_prescale_data)}")
    
    # Calculate averages
    print("Calculating averages...")
    results, df = calculate_averages(all_prescale_data)
    
    # Print summary
    print_summary(results)
    
    # Save detailed results
    save_detailed_results(results, df, args.output_dir)

if __name__ == "__main__":
    main()
