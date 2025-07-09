import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Union, List, Tuple

class AirQualityDataset(Dataset):
    """
    AirQuality dataset adapter for basicts.
    This class adapts the TSL AirQuality dataset to the format expected by basicts.
    
    Args:
        data_dir: Directory where the AirQuality dataset is stored
        mode: One of ['train', 'val', 'test']
        input_len: Length of the input sequence
        output_len: Length of the output sequence
        small: Whether to use the small version of the dataset (36 nodes in Beijing)
        add_time_features: Whether to add time features (hour of day, day of week)
    """
    
    def __init__(self, 
                 data_dir: str,
                 mode: str = 'train',
                 input_len: int = 12,
                 output_len: int = 12,
                 small: bool = False,
                 add_time_features: bool = True):
        super().__init__()
        
        self.data_dir = data_dir
        self.mode = mode
        self.input_len = input_len
        self.output_len = output_len
        self.small = small
        self.add_time_features = add_time_features
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load AirQuality dataset from TSL format"""
        try:
            from tsl.datasets import AirQuality as TSLAirQuality
        except ImportError:
            raise ImportError("TSL package is required to use AirQuality dataset. "
                             "Please install it with `pip install tsl`.")
        
        # Create TSL AirQuality dataset
        tsl_dataset = TSLAirQuality(root=self.data_dir, small=self.small)
        
        # Get data
        df = tsl_dataset.dataframe()  # Get the dataframe with PM2.5 values
        mask = tsl_dataset.mask  # Get the mask (1 if value is valid, 0 if missing)
        eval_mask = tsl_dataset.eval_mask  # Get the evaluation mask
        
        # Convert to numpy arrays
        self.data = df.values  # Shape: [T, N]
        self.mask = mask.values if mask is not None else np.ones_like(self.data)
        self.eval_mask = eval_mask if eval_mask is not None else np.ones_like(self.data)
        
        # Get timestamps
        self.timestamps = df.index
        
        # Create time features
        if self.add_time_features:
            # Hour of day (0-23)
            hour_of_day = np.array([t.hour for t in self.timestamps])
            # Day of week (0-6)
            day_of_week = np.array([t.dayofweek for t in self.timestamps])
            
            # Normalize time features
            hour_of_day = hour_of_day / 24.0  # Normalize to [0, 1)
            day_of_week = day_of_week / 7.0   # Normalize to [0, 1)
            
            # Reshape to [T, N, 1]
            hour_of_day = np.tile(hour_of_day.reshape(-1, 1, 1), (1, self.data.shape[1], 1))
            day_of_week = np.tile(day_of_week.reshape(-1, 1, 1), (1, self.data.shape[1], 1))
            
            # Add time features to data
            self.data = np.expand_dims(self.data, axis=-1)  # [T, N, 1]
            self.data = np.concatenate([self.data, hour_of_day, day_of_week], axis=-1)  # [T, N, 3]
        else:
            # Just reshape to [T, N, 1]
            self.data = np.expand_dims(self.data, axis=-1)  # [T, N, 1]
        
        # Split data into train/val/test
        self._split_data()
        
    def _split_data(self):
        """Split data into train/val/test sets"""
        # Default split: 70% train, 10% val, 20% test
        n_samples = len(self.timestamps)
        train_ratio, val_ratio = 0.7, 0.1
        
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        if self.mode == 'train':
            self.data = self.data[:train_end]
            self.mask = self.mask[:train_end]
            self.eval_mask = self.eval_mask[:train_end]
            self.timestamps = self.timestamps[:train_end]
        elif self.mode == 'val':
            self.data = self.data[train_end:val_end]
            self.mask = self.mask[train_end:val_end]
            self.eval_mask = self.eval_mask[train_end:val_end]
            self.timestamps = self.timestamps[train_end:val_end]
        elif self.mode == 'test':
            self.data = self.data[val_end:]
            self.mask = self.mask[val_end:]
            self.eval_mask = self.eval_mask[val_end:]
            self.timestamps = self.timestamps[val_end:]
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        # Create valid indices (ensure we have enough data for input and output)
        self.valid_indices = self._get_valid_indices()
        
    def _get_valid_indices(self):
        """Get valid indices for sampling"""
        total_len = self.input_len + self.output_len
        valid_indices = []
        
        for i in range(len(self.data) - total_len + 1):
            # Check if we have enough valid data in both input and output
            input_mask = self.mask[i:i+self.input_len].mean() > 0.8  # At least 80% valid
            output_mask = self.mask[i+self.input_len:i+total_len].mean() > 0.8
            
            if input_mask and output_mask:
                valid_indices.append(i)
        
        return valid_indices
    
    def __len__(self):
        """Return the number of valid samples"""
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """Get a sample"""
        # Get the starting index
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.input_len + self.output_len
        
        # Get data
        data_slice = self.data[start_idx:end_idx]
        
        # Split into history and future
        history_data = data_slice[:self.input_len]  # [input_len, N, C]
        future_data = data_slice[self.input_len:]   # [output_len, N, C]
        
        # Transpose to [B, L, N, C]
        history_data = np.transpose(history_data, (1, 0, 2))  # [N, input_len, C]
        future_data = np.transpose(future_data, (1, 0, 2))    # [N, output_len, C]
        
        # Add batch dimension
        history_data = np.expand_dims(history_data, axis=0)  # [1, N, input_len, C]
        future_data = np.expand_dims(future_data, axis=0)    # [1, N, output_len, C]
        
        # Transpose to final shape [B, L, N, C]
        history_data = np.transpose(history_data, (0, 2, 1, 3))  # [1, input_len, N, C]
        future_data = np.transpose(future_data, (0, 2, 1, 3))    # [1, output_len, N, C]
        
        # Convert to torch tensors
        history_data = torch.FloatTensor(history_data)
        future_data = torch.FloatTensor(future_data)
        
        return history_data, future_data

def load_air_quality_dataset(data_dir: str, 
                           batch_size: int = 32, 
                           input_len: int = 12,
                           output_len: int = 12,
                           small: bool = False,
                           add_time_features: bool = True,
                           train_ratio: float = 0.7,
                           val_ratio: float = 0.1,
                           test_ratio: float = 0.2):
    """
    Load AirQuality dataset for basicts.
    
    Args:
        data_dir: Directory where the AirQuality dataset is stored
        batch_size: Batch size for dataloaders
        input_len: Length of the input sequence
        output_len: Length of the output sequence
        small: Whether to use the small version of the dataset (36 nodes in Beijing)
        add_time_features: Whether to add time features (hour of day, day of week)
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        test_ratio: Ratio of data to use for testing
        
    Returns:
        train_loader, val_loader, test_loader: DataLoaders for train, val, test sets
    """
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = AirQualityDataset(
        data_dir=data_dir,
        mode='train',
        input_len=input_len,
        output_len=output_len,
        small=small,
        add_time_features=add_time_features
    )
    
    val_dataset = AirQualityDataset(
        data_dir=data_dir,
        mode='val',
        input_len=input_len,
        output_len=output_len,
        small=small,
        add_time_features=add_time_features
    )
    
    test_dataset = AirQualityDataset(
        data_dir=data_dir,
        mode='test',
        input_len=input_len,
        output_len=output_len,
        small=small,
        add_time_features=add_time_features
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader 