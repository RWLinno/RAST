#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAST Retrieval Store Analysis Script
====================================

This script analyzes the contents of a RAST model's retrieval store saved in NPZ format.
It provides visualizations and statistics about the stored temporal and spatial patterns.

Usage:
    python analyze_retrieval_store.py --file path/to/SD_store_epoch_1.npz [--output output_dir]
"""

import os
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze RAST retrieval store contents")
    parser.add_argument("--file", type=str, required=True, help="Path to NPZ file (e.g., SD_store_epoch_1.npz)")
    parser.add_argument("--output", type=str, default="./analysis_output", help="Output directory for analysis results")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--sample", type=int, default=12, help="Number of samples to show in text output")
    return parser.parse_args()

def load_npz_file(file_path: str) -> Dict[str, Any]:
    print(f"Loading file: {file_path}")
    
    try:
        with np.load(file_path, allow_pickle=True) as data:
            epoch = data.get('epoch')
            domain = str(data.get('domain'))
            
            temporal_vectors = data['temporal_vectors']
            spatial_vectors = data['spatial_vectors']
            
            try:
                temporal_values = json.loads(str(data['temporal_values']))
                spatial_values = json.loads(str(data['spatial_values']))
            except json.JSONDecodeError:
                print("Warning: Could not parse JSON values, using empty lists.")
                temporal_values = []
                spatial_values = []
            
            return {
                'epoch': epoch,
                'domain': domain,
                'temporal_vectors': temporal_vectors,
                'spatial_vectors': spatial_vectors,
                'temporal_values': temporal_values,
                'spatial_values': spatial_values
            }
    except Exception as e:
        print(f"Error loading NPZ file: {e}")
        raise

def compute_vector_statistics(vectors: np.ndarray) -> Dict[str, Any]:
    if len(vectors) == 0:
        return {'count': 0}
    
    stats = {
        'count': len(vectors),
        'dim': vectors.shape[1],
        'mean': np.mean(vectors, axis=0),
        'std': np.std(vectors, axis=0),
        'min': np.min(vectors, axis=0),
        'max': np.max(vectors, axis=0),
        'norm_mean': np.mean(np.linalg.norm(vectors, axis=1)),
        'norm_std': np.std(np.linalg.norm(vectors, axis=1)),
        'similarity_matrix': compute_similarity_matrix(vectors)
    }
    
    return stats

def compute_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1)
    normalized = vectors / norms
    
    similarity = np.dot(normalized, normalized.T)
    return similarity

def analyze_values(values: List[Dict]) -> Dict[str, Any]:
    if not values:
        return {'count': 0}
    
    sample = values[0]
    keys = list(sample.keys())
    
    results = {'count': len(values), 'keys': keys}
    
    for key in keys:
        if key in sample and isinstance(sample[key], (int, float, list)):
            if isinstance(sample[key], list):
                try:
                    values_array = np.array([item[key] for item in values])
                    if values_array.dtype.kind in 'iuf':
                        results[f'{key}_mean'] = np.mean(values_array, axis=0).tolist() if values_array.size > 0 else []
                        results[f'{key}_std'] = np.std(values_array, axis=0).tolist() if values_array.size > 0 else []
                except (ValueError, TypeError):
                    pass
            elif isinstance(sample[key], (int, float)):
                try:
                    values_array = np.array([item[key] for item in values])
                    results[f'{key}_mean'] = float(np.mean(values_array))
                    results[f'{key}_min'] = float(np.min(values_array))
                    results[f'{key}_max'] = float(np.max(values_array))
                except (ValueError, TypeError):
                    pass
    
    return results

def plot_vector_projections(vectors: np.ndarray, values: List[Dict], output_dir: str, prefix: str):
    if len(vectors) < 2:
        print(f"Not enough {prefix} vectors to plot projections")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    pca = PCA(n_components=2)
    vectors_pca = pca.fit_transform(vectors)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(vectors_pca[:, 0], vectors_pca[:, 1], alpha=0.7)
    
    labels = []
    for i, v in enumerate(values):
        if 'timestamp' in v:
            labels.append(f"T{v['timestamp']}")
        elif 'node_id' in v:
            labels.append(f"N{v['node_id']}")
        else:
            labels.append(str(i))
    
    max_labels = min(25, len(labels))
    step = len(labels) // max_labels if len(labels) > max_labels else 1
    for i in range(0, len(labels), step):
        plt.annotate(labels[i], (vectors_pca[i, 0], vectors_pca[i, 1]))
    
    plt.title(f'PCA projection of {prefix} vectors')
    plt.xlabel(f'PC1 (variance: {pca.explained_variance_ratio_[0]:.2f})')
    plt.ylabel(f'PC2 (variance: {pca.explained_variance_ratio_[1]:.2f})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_pca.png'), dpi=300)
    plt.close()
    
    if len(vectors) > 1000:
        print(f"Skipping t-SNE for {prefix} due to large number of vectors: {len(vectors)}")
        return
    
    try:
        tsne = TSNE(n_components=2, perplexity=min(30, len(vectors)-1), random_state=42)
        vectors_tsne = tsne.fit_transform(vectors)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(vectors_tsne[:, 0], vectors_tsne[:, 1], alpha=0.7)
        
        for i in range(0, len(labels), step):
            plt.annotate(labels[i], (vectors_tsne[i, 0], vectors_tsne[i, 1]))
        
        plt.title(f't-SNE projection of {prefix} vectors')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{prefix}_tsne.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error computing t-SNE for {prefix}: {e}")

def plot_similarity_matrix(similarity: np.ndarray, output_dir: str, prefix: str):
    if len(similarity) == 0:
        return
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity, cmap='viridis', vmin=-1, vmax=1)
    plt.title(f'Similarity matrix for {prefix} vectors')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_similarity.png'), dpi=300)
    plt.close()

def plot_metadata_stats(values: List[Dict], output_dir: str, prefix: str):
    if not values:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    if 'importance' in values[0]:
        importances = [v['importance'] for v in values]
        ids = [v.get('timestamp', v.get('node_id', i)) for i, v in enumerate(values)]
        
        df = pd.DataFrame({
            'ID': ids,
            'Importance': importances
        })
        df = df.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(df)), df['Importance'])
        plt.xticks(range(len(df)), df['ID'], rotation=90)
        plt.title(f'Importance scores for {prefix}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{prefix}_importance.png'), dpi=300)
        plt.close()
    
    if 'mean' in values[0] and isinstance(values[0]['mean'], list):
        means = np.array([v['mean'] for v in values])
        stds = np.array([v['std'] for v in values]) if 'std' in values[0] else None
        
        n_features = means.shape[1] if means.size > 0 else 0
        
        if n_features > 0:
            plt.figure(figsize=(12, 8))
            sns.heatmap(means, cmap='coolwarm', center=0)
            plt.title(f'Mean values heatmap for {prefix}')
            plt.xlabel('Feature dimension')
            if prefix == 'temporal':
                plt.ylabel('Timestep')
            else:
                plt.ylabel('Node ID')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{prefix}_means_heatmap.png'), dpi=300)
            plt.close()
            
            max_dims = min(5, n_features)
            plt.figure(figsize=(12, 6))
            for i in range(max_dims):
                plt.plot(means[:, i], label=f'Dim {i}')
            plt.title(f'Mean values for first {max_dims} dimensions ({prefix})')
            if prefix == 'temporal':
                plt.xlabel('Timestep')
            else:
                plt.xlabel('Node ID')
            plt.ylabel('Mean value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{prefix}_means_line.png'), dpi=300)
            plt.close()

def print_text_summary(data: Dict[str, Any], temporal_stats: Dict[str, Any], 
                     spatial_stats: Dict[str, Any], temporal_analysis: Dict[str, Any],
                     spatial_analysis: Dict[str, Any], sample_count: int):
    print("\n" + "="*80)
    print(f"RAST RETRIEVAL STORE ANALYSIS - Domain: {data['domain']}, Epoch: {data['epoch']}")
    print("="*80)
    
    print("\n--- GENERAL INFORMATION ---")
    print(f"Domain: {data['domain']}")
    print(f"Epoch: {data['epoch']}")
    print(f"Temporal vectors: {temporal_stats['count']} (dim: {temporal_stats.get('dim', 'N/A')})")
    print(f"Spatial vectors: {spatial_stats['count']} (dim: {spatial_stats.get('dim', 'N/A')})")
    
    print("\n--- VECTOR STATISTICS ---")
    print("\nTemporal vectors:")
    if temporal_stats['count'] > 0:
        print(f"  Mean vector norm: {temporal_stats['norm_mean']:.4f} (std: {temporal_stats['norm_std']:.4f})")
        print(f"  Feature range: [{np.min(temporal_stats['min']):.4f}, {np.max(temporal_stats['max']):.4f}]")
    else:
        print("  No temporal vectors found")
    
    print("\nSpatial vectors:")
    if spatial_stats['count'] > 0:
        print(f"  Mean vector norm: {spatial_stats['norm_mean']:.4f} (std: {spatial_stats['norm_std']:.4f})")
        print(f"  Feature range: [{np.min(spatial_stats['min']):.4f}, {np.max(spatial_stats['max']):.4f}]")
    else:
        print("  No spatial vectors found")
    
    print("\n--- METADATA ANALYSIS ---")
    
    print("\nTemporal metadata:")
    if temporal_analysis['count'] > 0:
        print(f"  Count: {temporal_analysis['count']}")
        print(f"  Available fields: {', '.join(temporal_analysis['keys'])}")
        
        if 'importance_mean' in temporal_analysis:
            print(f"  Mean importance: {temporal_analysis['importance_mean']:.4f}")
            print(f"  Min importance: {temporal_analysis['importance_min']:.4f}")
            print(f"  Max importance: {temporal_analysis['importance_max']:.4f}")
        
        indices = sorted(random.sample(range(len(data['temporal_values'])), 
                                 min(sample_count, len(data['temporal_values']))))
        print(f"\n  Random samples ({len(indices)} total):")
        for i in indices:
            item = data['temporal_values'][i]
            print(f"    Timestep {i}: ", end="")
            info = []
            if 'timestamp' in item:
                info.append(f"timestamp: {item['timestamp']}")
            if 'importance' in item:
                info.append(f"importance: {item['importance']:.4f}")
            if 'mean' in item:
                info.append(f"mean values: [{', '.join([f'{v:.3f}' for v in item['mean'][:3]])}...]")
            if 'std' in item:
                info.append(f"std values: [{', '.join([f'{v:.3f}' for v in item['std'][:3]])}...]")
            print(", ".join(info))
    else:
        print("  No temporal metadata found")
    
    print("\nSpatial metadata:")
    if spatial_analysis['count'] > 0:
        print(f"  Count: {spatial_analysis['count']}")
        print(f"  Available fields: {', '.join(spatial_analysis['keys'])}")
        
        if 'importance_mean' in spatial_analysis:
            print(f"  Mean importance: {spatial_analysis['importance_mean']:.4f}")
            print(f"  Min importance: {spatial_analysis['importance_min']:.4f}")
            print(f"  Max importance: {spatial_analysis['importance_max']:.4f}")
        
        indices = sorted(random.sample(range(len(data['spatial_values'])), 
                                 min(sample_count, len(data['spatial_values']))))
        print(f"\n  Random node samples ({len(indices)} total):")
        for i in indices:
            item = data['spatial_values'][i]
            print(f"    Node {i}: ", end="")
            info = []
            if 'node_id' in item:
                info.append(f"node_id: {item['node_id']}")
            if 'importance' in item:
                info.append(f"importance: {item['importance']:.4f}")
            if 'mean' in item:
                info.append(f"mean values: [{', '.join([f'{v:.3f}' for v in item['mean'][:3]])}...]")
            if 'std' in item:
                info.append(f"std values: [{', '.join([f'{v:.3f}' for v in item['std'][:3]])}...]")
            print(", ".join(info))
    else:
        print("  No spatial metadata found")
    
    print("\n" + "="*80)
    print("Analysis complete! See output directory for visualizations.")
    print("="*80 + "\n")

def analyze_sample_features(data, sample_index, is_temporal=True):
    values = data['temporal_values'] if is_temporal else data['spatial_values']
    vectors = data['temporal_vectors'] if is_temporal else data['spatial_vectors']
    
    if sample_index >= len(values):
        print(f"Sample index {sample_index} out of range")
        return
        
    sample = values[sample_index]
    vector = vectors[sample_index]
    
    print(f"\nDetailed feature analysis for {'timestep' if is_temporal else 'node'} {sample_index}:")
    print("-" * 60)
    
    for key, value in sample.items():
        if isinstance(value, list):
            print(f"{key}: [mean: {np.mean(value):.4f}, min: {np.min(value):.4f}, max: {np.max(value):.4f}]")
        else:
            print(f"{key}: {value}")
    
    print(f"\nVector statistics (dimension: {len(vector)}):")
    print(f"  Norm: {np.linalg.norm(vector):.4f}")
    print(f"  Mean: {np.mean(vector):.4f}")
    print(f"  Std dev: {np.std(vector):.4f}")
    print(f"  Max value: {np.max(vector):.4f}")
    print(f"  Min value: {np.min(vector):.4f}")
    print(f"  First 5 values: {vector[:5]}")

def main():
    args = parse_args()
    
    file_path = args.file
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return
    
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    data = load_npz_file(file_path)
    
    temporal_stats = compute_vector_statistics(data['temporal_vectors'])
    spatial_stats = compute_vector_statistics(data['spatial_vectors'])
    
    temporal_analysis = analyze_values(data['temporal_values'])
    spatial_analysis = analyze_values(data['spatial_values'])
    
    if args.plot:
        plot_vector_projections(data['temporal_vectors'], data['temporal_values'], output_dir, 'temporal')
        plot_vector_projections(data['spatial_vectors'], data['spatial_values'], output_dir, 'spatial')
        
        if temporal_stats['count'] > 0:
            plot_similarity_matrix(temporal_stats['similarity_matrix'], output_dir, 'temporal')
        if spatial_stats['count'] > 0:
            plot_similarity_matrix(spatial_stats['similarity_matrix'], output_dir, 'spatial')
        
        plot_metadata_stats(data['temporal_values'], output_dir, 'temporal')
        plot_metadata_stats(data['spatial_values'], output_dir, 'spatial')
    
    print_text_summary(data, temporal_stats, spatial_stats, temporal_analysis, 
                     spatial_analysis, args.sample)
    
    summary_path = os.path.join(output_dir, 'analysis_summary.txt')
    with open(summary_path, 'w') as f:
        import sys
        old_stdout = sys.stdout
        sys.stdout = f
        print_text_summary(data, temporal_stats, spatial_stats, temporal_analysis, 
                         spatial_analysis, args.sample)
        sys.stdout = old_stdout
    
    print(f"Summary saved to {summary_path}")

    if temporal_stats['count'] > 0:
        random_temporal = np.random.randint(0, temporal_stats['count'])
        analyze_sample_features(data, random_temporal, is_temporal=True)
    
    if spatial_stats['count'] > 0:
        random_spatial = np.random.randint(0, spatial_stats['count'])
        analyze_sample_features(data, random_spatial, is_temporal=False)

if __name__ == "__main__":
    main()