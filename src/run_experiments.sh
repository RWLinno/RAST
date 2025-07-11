#!/bin/bash

# RAST Experiment Runner Script
# =============================
# This script runs all ablation studies and parameter sensitivity analyses

echo "🧪 RAST Comprehensive Experiment Runner"
echo "========================================"

GPU_ID=${1:-2}  # Default to GPU 2, can be overridden with first argument

echo "🖥️ Using GPU: $GPU_ID"

# Ablation Studies
echo ""
echo "🔬 Starting Ablation Studies..."
echo "================================"

echo "▶️ Running: Without Query Embedding"
python ../experiments/train.py -c PEMS04_without_query_embedding.py -g $GPU_ID
echo "✅ Completed: Without Query Embedding"

echo "▶️ Running: Without Retrieval Embedding"
python ../experiments/train.py -c PEMS04_without_retrieval_embedding.py -g $GPU_ID
echo "✅ Completed: Without Retrieval Embedding"

# Parameter Sensitivity Studies
echo ""
echo "📊 Starting Parameter Sensitivity Studies..."
echo "============================================="

echo "▶️ Running: Retrieval Dimension = 64"
python ../experiments/train.py -c PEMS04_retrieval_embedding_64.py -g $GPU_ID
echo "✅ Completed: Retrieval Dimension = 64"

echo "▶️ Running: Update Interval = 5"
python ../experiments/train.py -c PEMS04_update_interval_5.py -g $GPU_ID
echo "✅ Completed: Update Interval = 5"

echo ""
echo "🎉 All experiments completed!"
echo "=========================="
echo "📊 Results can be found in the respective checkpoint directories."
echo "📁 Ablation studies: checkpoints/ablation_studies/"
echo "📁 Parameter sensitivity: checkpoints/parameter_sensitivity/" 