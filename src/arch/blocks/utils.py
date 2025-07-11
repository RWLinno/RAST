import time
import functools
import numpy as np
import torch
import os

from baselines.STID.arch.stid_arch import STID
from baselines.AGCRN.arch.agcrn_arch import AGCRN

def load_model(model_path, model_name, device):
    """Load a model from a path"""
    model_dict = {    
        'STID': STID,
        'AGCRN': AGCRN
    }
    model = _build_model(model_dict[model_name])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def _build_model(self):
    raise NotImplementedError
    return None

def timing_decorator(func_name, module_timings):
    """Decorator for timing model components"""
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
    """Count parameters in a module"""
    if hasattr(module, 'parameters'):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return 0

def analyze_model_efficiency(model):
    """Analyze module efficiency and exit"""
    print("\n" + "="*80)
    print("🔍 RAST MODEL EFFICIENCY ANALYSIS")
    print("="*80)
    
    # Collect parameter counts
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
    
    # Timing analysis
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
    
    # Performance recommendations
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
    """Record timing for a module"""
    if module_name not in module_timings:
        module_timings[module_name] = []
    module_timings[module_name].append(duration)
