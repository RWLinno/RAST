import time
import functools
import numpy as np
import torch
import os
import glob

from baselines.STID.arch.stid_arch import STID
from baselines.AGCRN.arch.agcrn_arch import AGCRN
from .model_configs import get_model_config, infer_dataset_from_path

MODEL_REGISTRY = {    
    'STID': STID,
    'AGCRN': AGCRN
}

def register_model(name, model_class):
    MODEL_REGISTRY[name] = model_class
    print(f"Registered model: {name}")

def load_pretrained_stgnn(pretrain_path, model_name):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not supported. Available: {list(MODEL_REGISTRY.keys())}")
    
    dataset_name = infer_dataset_from_path(pretrain_path)
    print(f"Inferred dataset: {dataset_name}")
    
    model_config = get_model_config(model_name, dataset_name)
    print(f"Loading {model_name} with config: {model_config}")
    
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(**model_config)
    
    _load_checkpoint(model, pretrain_path)
    
    model.eval()
    print(f"Successfully loaded pre-trained {model_name} from {pretrain_path}")
    return model

def _load_checkpoint(model, checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {str(e)}")



def timing_decorator(func_name, module_timings):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            result = func(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            
            if func_name not in module_timings:
                module_timings[func_name] = []
            module_timings[func_name].append(end_time - start_time)
            
            return result
        return wrapper
    return decorator

def count_parameters(module):
    if hasattr(module, 'parameters'):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return 0

def analyze_model_efficiency(model):
    print("\n" + "="*80)
    print("🔍 RAST MODEL EFFICIENCY ANALYSIS")
    print("="*80)
    
    modules = {
        'temporal_series_encoder': model.temporal_series_encoder,
        'spatial_node_embeddings': model.spatial_node_embeddings,
        'feature_encoder_layers': model.feature_encoder_layers,
        'query_proj': model.query_proj,
        'key_proj': model.key_proj,
        'value_proj': model.value_proj,
        'cross_attention': model.cross_attention,
        'out_proj': model.out_proj,
        'llm_encoder': model.llm_encoder,
        'llm_proj': model.llm_proj,
    }
    
    total_params = 0
    print(f"{'Module':<25} {'Parameters':<15} {'Avg Time (ms)':<15} {'Efficiency':<15}")
    print("-" * 70)
    
    for module_name, module in modules.items():
        param_count = count_parameters(module)
        total_params += param_count
        
        if hasattr(model, 'module_timings') and module_name in model.module_timings:
            avg_time = np.mean(model.module_timings[module_name]) * 1000  # Convert to ms
            efficiency = param_count / avg_time if avg_time > 0 else float('inf')
            print(f"{module_name:<25} {param_count:<15,} {avg_time:<15.3f} {efficiency:<15,.0f}")
        else:
            print(f"{module_name:<25} {param_count:<15,} {'N/A':<15} {'N/A':<15}")
    
    print("-" * 70)
    print(f"{'TOTAL':<25} {total_params:<15,}")
    
    if hasattr(model, 'module_timings'):
        print("\n📊 TIMING BREAKDOWN:")
        total_time = sum(np.mean(times) for times in model.module_timings.values()) * 1000
        
        for module_name, times in model.module_timings.items():
            avg_time = np.mean(times) * 1000
            percentage = (avg_time / total_time) * 100 if total_time > 0 else 0
            print(f"  {module_name:<25}: {avg_time:>8.3f}ms ({percentage:>5.1f}%)")
        
        print(f"\n⚡ Total Forward Time: {total_time:.3f}ms")
    
    if hasattr(model, 'device'):
        print(f"🚀 GPU Device: {model.device}")
    if hasattr(model, 'use_amp'):
        print(f"🔧 Mixed Precision: {'Enabled' if model.use_amp else 'Disabled'}")
    
    if hasattr(model, 'module_timings'):
        print("\n💡 PERFORMANCE RECOMMENDATIONS:")
        sorted_times = sorted(model.module_timings.items(), 
                            key=lambda x: np.mean(x[1]), reverse=True)
        
        print("  Top 3 Time-Consuming Modules:")
        for i, (module_name, times) in enumerate(sorted_times[:3]):
            avg_time = np.mean(times) * 1000
            print(f"    {i+1}. {module_name}: {avg_time:.3f}ms")
    
    print("\n🚨 TIMING MODE - EXITING AFTER ANALYSIS")
    print("="*80)

def record_timing(module_timings, module_name: str, duration: float):
    if module_name not in module_timings:
        module_timings[module_name] = []
    module_timings[module_name].append(duration)
