#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert AirQuality dataset to basicts standard format.
This script directly downloads and converts the AirQuality dataset to the standard format used by basicts.
Implementation follows the TSL library's AirQuality dataset format.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import h5py
import urllib.request
import zipfile
import io

def infer_mask(df, infer_from='next'):
    """Infer evaluation mask from DataFrame."""
    mask = (~df.isna()).astype('uint8')
    eval_mask = pd.DataFrame(index=mask.index, columns=mask.columns,
                             data=0).astype('uint8')
    if infer_from == 'previous':
        offset = -1
    elif infer_from == 'next':
        offset = 1
    else:
        raise ValueError('`infer_from` can only be one of {}'.format(
            ['previous', 'next']))
    months = sorted(set(zip(mask.index.year, mask.index.month)))
    length = len(months)
    for i in range(length):
        j = (i + offset) % length
        year_i, month_i = months[i]
        year_j, month_j = months[j]
        cond_j = (mask.index.year == year_j) & (mask.index.month == month_j)
        mask_j = mask[cond_j]
        offset_i = 12 * (year_i - year_j) + (month_i - month_j)
        mask_i = mask_j.shift(1, pd.DateOffset(months=offset_i))
        mask_i = mask_i[~mask_i.index.duplicated(keep='first')]
        mask_i = mask_i[np.in1d(mask_i.index, mask.index)]
        i_idx = mask_i.index
        eval_mask.loc[i_idx] = ~mask_i.loc[i_idx] & mask.loc[i_idx]
    return eval_mask

def geographical_distance(coords, to_rad=True):
    """Calculate geographical distance between coordinates."""
    if to_rad:
        coords = np.radians(coords)
    
    lat = coords['latitude'].values
    lon = coords['longitude'].values
    
    dlat = lat[:, None] - lat
    dlon = lon[:, None] - lon
    
    a = np.sin(dlat/2)**2 + np.cos(lat[:, None]) * np.cos(lat) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Earth's radius in kilometers
    
    return c * r

def temporal_mean(df):
    """Calculate temporal mean by hour and day of week."""
    return df.groupby([df.index.dayofweek, df.index.hour]).transform('mean')

def convert_air_quality_to_basicts(data_dir, output_dir=None, small=False):
    """Convert AirQuality dataset to basicts standard format."""
    # Create output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(data_dir, 'basicts_format')
    
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Download URL from TSL
    tsl_url = "https://drive.switch.ch/index.php/s/W0fRqotjHxIndPj/download"
    zip_path = os.path.join(data_dir, 'data.zip')
    
    # Download and extract data
    print(f"Downloading AirQuality dataset {'(small version)' if small else ''}...")
    urllib.request.urlretrieve(tsl_url, zip_path)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    os.unlink(zip_path)
    
    # Load data based on small flag
    if small:
        h5_path = os.path.join(data_dir, 'small36.h5')
        df = pd.read_hdf(h5_path, 'pm25')
        eval_mask = pd.read_hdf(h5_path, 'eval_mask')
        stations = pd.read_hdf(h5_path, 'stations')
    else:
        h5_path = os.path.join(data_dir, 'full437.h5')
        df = pd.read_hdf(h5_path, 'pm25')
        stations = pd.read_hdf(h5_path, 'stations')
        eval_mask = infer_mask(df, infer_from='next')
    
    # Create mask (1 if value is valid, 0 if missing)
    mask = (~df.isna()).astype('uint8')
    missing_ratio = df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100
    
    # Calculate geographical distances
    st_coord = stations.loc[:, ['latitude', 'longitude']]
    dist = geographical_distance(st_coord, to_rad=True)
    
    # Fill missing values with temporal mean
    df = df.fillna(temporal_mean(df))
    
    print(f"Dataset shape: {df.shape}")
    print(f"Number of nodes: {df.shape[1]}")
    print(f"Number of time steps: {df.shape[0]}")
    print(f"Missing values: {missing_ratio:.2f}%")
    
    # Create time features
    hour_of_day = np.array([t.hour for t in df.index])
    day_of_week = np.array([t.dayofweek for t in df.index])
    
    # Normalize time features
    hour_of_day = hour_of_day / 24.0  # Normalize to [0, 1)
    day_of_week = day_of_week / 7.0   # Normalize to [0, 1)
    
    # Reshape to [T, N, 1]
    hour_of_day = np.tile(hour_of_day.reshape(-1, 1, 1), (1, df.shape[1], 1))
    day_of_week = np.tile(day_of_week.reshape(-1, 1, 1), (1, df.shape[1], 1))
    
    # Create 3D array [T, N, C] where C=3 (PM2.5, hour_of_day, day_of_week)
    data_3d = np.zeros((df.shape[0], df.shape[1], 3), dtype=np.float32)
    data_3d[:, :, 0] = df.values
    data_3d[:, :, 1] = hour_of_day.reshape(df.shape[0], df.shape[1])
    data_3d[:, :, 2] = day_of_week.reshape(df.shape[0], df.shape[1])
    
    # Save data in basicts format
    dataset_name = 'AirQuality_small' if small else 'AirQuality'
    
    # Create dataset directory
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Save data.dat file (memory-mapped file)
    data_file_path = os.path.join(dataset_dir, 'data.dat')
    data_shape = data_3d.shape
    data_memmap = np.memmap(data_file_path, dtype='float32', mode='w+', shape=data_shape)
    data_memmap[:] = data_3d[:]
    data_memmap.flush()
    
    # Save description file
    desc = {
        'name': dataset_name,
        'shape': list(data_shape),
        'num_nodes': df.shape[1],
        'num_time_steps': df.shape[0],
        'num_features': 3,  # PM2.5, hour_of_day, day_of_week
        'missing_values': missing_ratio / 100,  # Convert to ratio
        'sampling_rate': '1 hour',
        'start_time': str(df.index[0]),
        'end_time': str(df.index[-1])
    }
    
    with open(os.path.join(dataset_dir, 'desc.json'), 'w') as f:
        json.dump(desc, f, indent=4)
    
    # Save adjacency matrix using Gaussian kernel
    # Use same theta for both air and air36 as in TSL implementation
    theta = np.std(dist[:36, :36])
    adj = np.exp(-dist**2 / (2 * theta**2))
    
    # Save adjacency matrix
    np.save(os.path.join(dataset_dir, 'adj_mx.npy'), adj)
    
    # Save mask
    mask_values = mask.values.astype(np.float32)
    mask_file_path = os.path.join(dataset_dir, 'mask.dat')
    mask_memmap = np.memmap(mask_file_path, dtype='float32', mode='w+', shape=(mask_values.shape[0], mask_values.shape[1], 1))
    mask_memmap[:, :, 0] = mask_values
    mask_memmap.flush()
    
    # Save eval mask
    eval_mask_values = eval_mask.values.astype(np.float32)
    eval_mask_file_path = os.path.join(dataset_dir, 'eval_mask.dat')
    eval_mask_memmap = np.memmap(eval_mask_file_path, dtype='float32', mode='w+', shape=(eval_mask_values.shape[0], eval_mask_values.shape[1], 1))
    eval_mask_memmap[:, :, 0] = eval_mask_values
    eval_mask_memmap.flush()
    
    print(f"Dataset converted successfully: {dataset_name}")
    print(f"Data saved to {dataset_dir}")
    
    return dataset_dir

def main():
    parser = argparse.ArgumentParser(description='Convert AirQuality dataset to basicts standard format')
    parser.add_argument('--data_dir', type=str, default='./datasets',
                        help='Directory to store the downloaded dataset')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to store the processed data for basicts')
    parser.add_argument('--small', action='store_true',
                        help='Use the small version of the dataset (36 nodes in Beijing)')
    
    args = parser.parse_args()
    
    convert_air_quality_to_basicts(args.data_dir, args.output_dir, args.small)

if __name__ == '__main__':
    main() 