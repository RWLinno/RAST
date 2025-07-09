#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prepare AirQuality dataset for basicts.
This script downloads the AirQuality dataset from TSL and prepares it for use with basicts.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def prepare_air_quality_data(data_dir, output_dir=None, small=False):
    """
    Prepare AirQuality dataset for basicts.
    
    Args:
        data_dir: Directory to store the TSL dataset
        output_dir: Directory to store the processed data for basicts
        small: Whether to use the small version of the dataset (36 nodes in Beijing)
    """
    try:
        from tsl.datasets import AirQuality as TSLAirQuality
    except ImportError:
        print("TSL package is required to use AirQuality dataset.")
        print("Please install it with `pip install tsl`.")
        sys.exit(1)
    
    # Create output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(data_dir, 'basicts_format')
    
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading AirQuality dataset {'(small version)' if small else ''}...")
    
    # Create TSL AirQuality dataset (this will download the data if not already present)
    tsl_dataset = TSLAirQuality(root=data_dir, small=small)
    
    # Get data
    df = tsl_dataset.dataframe()  # Get the dataframe with PM2.5 values
    mask = tsl_dataset.mask  # Get the mask (1 if value is valid, 0 if missing)
    eval_mask = tsl_dataset.eval_mask  # Get the evaluation mask
    dist = tsl_dataset.dist  # Get the distance matrix
    
    print(f"Dataset shape: {df.shape}")
    print(f"Number of nodes: {df.shape[1]}")
    print(f"Number of time steps: {df.shape[0]}")
    print(f"Missing values: {(~mask.values).sum() / mask.values.size:.2%}")
    
    # Save data in basicts format
    dataset_name = 'AirQuality_small' if small else 'AirQuality'
    
    # Create dataset directory
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Save data
    print(f"Saving data to {dataset_dir}...")
    
    # Save raw data
    df.to_csv(os.path.join(dataset_dir, 'pm25.csv'))
    
    # Save mask
    mask_df = pd.DataFrame(mask.values, index=df.index, columns=df.columns)
    mask_df.to_csv(os.path.join(dataset_dir, 'mask.csv'))
    
    # Save eval mask
    eval_mask_df = pd.DataFrame(eval_mask, index=df.index, columns=df.columns)
    eval_mask_df.to_csv(os.path.join(dataset_dir, 'eval_mask.csv'))
    
    # Save distance matrix
    np.save(os.path.join(dataset_dir, 'dist.npy'), dist)
    
    # Create adjacency matrix from distance matrix
    # Use Gaussian kernel to convert distance to similarity
    sigma = np.std(dist)
    adj = np.exp(-dist**2 / (2 * sigma**2))
    
    # Save adjacency matrix
    np.save(os.path.join(dataset_dir, 'adj_mx.npy'), adj)
    
    # Create time features
    hour_of_day = np.array([t.hour for t in df.index])
    day_of_week = np.array([t.dayofweek for t in df.index])
    
    # Normalize time features
    hour_of_day = hour_of_day / 24.0  # Normalize to [0, 1)
    day_of_week = day_of_week / 7.0   # Normalize to [0, 1)
    
    # Save time features
    time_features = pd.DataFrame({
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week
    }, index=df.index)
    time_features.to_csv(os.path.join(dataset_dir, 'time_features.csv'))
    
    # Create dataset info
    info = {
        'name': dataset_name,
        'num_nodes': df.shape[1],
        'num_time_steps': df.shape[0],
        'missing_values': (~mask.values).sum() / mask.values.size,
        'sampling_rate': '1 hour',
        'start_time': str(df.index[0]),
        'end_time': str(df.index[-1])
    }
    
    # Save info
    with open(os.path.join(dataset_dir, 'info.txt'), 'w') as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Dataset prepared successfully: {dataset_name}")
    print(f"Data saved to {dataset_dir}")
    
    return dataset_dir

def main():
    parser = argparse.ArgumentParser(description='Prepare AirQuality dataset for basicts')
    parser.add_argument('--data_dir', type=str, default='./datasets',
                        help='Directory to store the TSL dataset')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to store the processed data for basicts')
    parser.add_argument('--small', action='store_true',
                        help='Use the small version of the dataset (36 nodes in Beijing)')
    
    args = parser.parse_args()
    
    prepare_air_quality_data(args.data_dir, args.output_dir, args.small)

if __name__ == '__main__':
    main() 