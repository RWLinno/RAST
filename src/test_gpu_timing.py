#!/usr/bin/env python3
"""
Quick test script for RAST GPU and timing functionality
Usage: python test_gpu_timing.py
"""

import torch
import numpy as np
import sys
import os

# Add BasicTS to path
sys.path.append(os.path.abspath('.'))

# Import RAST model
from RAST.arch import RAST

def test_gpu_timing():
    """Test GPU device setting and timing mode"""
    print("🧪 Testing RAST GPU and Timing Functionality")
    print("=" * 50)
    
    # Check CUDA availability
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_properties(i).name}")
    
    # Test model parameters
    model_args = {
        "num_nodes": 307,
        "input_len": 12,
        "output_len": 12,
        "input_dim": 3,
        "output_dim": 1,
        "embed_dim": 128,
        "retrieval_dim": 128,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "prompt_domain": "PEMS04",
        "top_k": 3,
        "dropout": 0.1,
        "update_interval": 15,
        # Test parameters
        "device_id": 2,  # Use GPU 2
        "timing_mode": True,  # Enable timing analysis
        "use_amp": False,  # Disable AMP
        "llm_model": "bert-base-uncased",
        "llm_dim": 768,
    }
    
    print("\n🚀 Initializing RAST model...")
    try:
        model = RAST(**model_args)
        print("✅ Model initialized successfully")
        print(f"Model device: {next(model.parameters()).device}")
        
        # Create dummy input data
        batch_size = 4
        seq_len = 12
        num_nodes = 307
        input_dim = 3
        
        # Generate dummy data on the correct device
        device = next(model.parameters()).device
        history_data = torch.randn(batch_size, seq_len, num_nodes, input_dim).to(device)
        future_data = torch.randn(batch_size, seq_len, num_nodes, input_dim).to(device)
        
        print(f"Input data shape: {history_data.shape}")
        print(f"Input data device: {history_data.device}")
        
        print("\n⏱️ Running forward pass with timing analysis...")
        
        # This should trigger timing analysis and exit
        with torch.no_grad():
            result = model.forward(
                history_data=history_data,
                future_data=future_data,
                batch_seen=0,
                epoch=1,
                train=True
            )
        
        # This should not be reached due to timing_mode exit
        print("❌ Timing mode should have exited before this point")
        
    except SystemExit:
        print("✅ Timing mode successfully analyzed and exited")
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gpu_timing() 