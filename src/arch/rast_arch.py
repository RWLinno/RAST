import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import faiss
import numpy as np
import json
import os
import glob
import traceback
import time
import functools
from .blocks.Embed import *
from .blocks.Pretrain import *
from .blocks.Text_Encoder import *
from .blocks.RetrievalStore import *

class RAST(nn.Module):
    """
    Retrieval-Augmented Spatio-Temporal Model (RAST)
    
    A neural network architecture that combines:
    1. Pre-trained transformer encoder for spatio-temporal feature extraction
    2. Retrieval-based augmentation using domain-specific prompts
    3. Cross-attention mechanism for feature fusion
    """
    
    def __init__(self, **model_args):
        """
        Initialize RAST model
        
        Args:
            model_args: Dictionary containing model parameters including:
                - num_nodes: Number of spatial nodes
                - input_dim: Input feature dimension
                - output_dim: Output prediction dimension
                - embed_dim: Embedding dimension
                - input_len: Input sequence length
                - output_len: Prediction horizon
                - encoder_layers: Number of transformer encoder layers
                - decoder_layers: Number of decoder layers
                - patch_size: Size of patches for embedding
                - prompt_domain: Domain for prompt generation
                - top_k: Number of nearest neighbors for retrieval
                - batch_size: Batch size for prompt processing (default: 32)
        """
        super().__init__()
        # Model parameters
        self.num_nodes = model_args['num_nodes'] 
        self.input_dim = model_args['input_dim']
        self.output_dim = model_args['output_dim']
        self.embed_dim = model_args['embed_dim']
        self.retrieval_dim = model_args['retrieval_dim']
        self.temporal_dim = model_args.get('temporal_dim', 64)
        self.spatial_dim = model_args.get('spatial_dim', 32)
        self.seq_len = model_args['input_len']
        self.horizon = model_args['output_len']
        self.encoder_layers = model_args['encoder_layers']
        self.decoder_layers = model_args['decoder_layers']
        self.patch_size = model_args['patch_size']
        self.domain = model_args['prompt_domain']
        self.top_k = model_args['top_k']
        self.dropout = model_args['dropout']
        self.prompt_batch_size = model_args.get('batch_size', 32)
        
        self.debug_prompts = 1
        self.update_interval = model_args.get('update_interval', 5)  # Default to 20 epochs for better training speed
        self.output_type = model_args.get('output_type', 'full')  # 'full', 'only_data_embed', or 'only_retrieval_embed'
        
        # GPU device setting and timing mode
        self.device_id = model_args.get('device_id', 0)  # Single GPU device ID
        self.timing_mode = model_args.get('timing_mode', False)  # Enable detailed timing analysis
        self.use_amp = model_args.get('use_amp', False)  # Disable AMP by default to fix GPU issues
        
        # LLM configuration
        self.llm_model = model_args.get('llm_model', "bert-base-uncased")
        self.llm_dim = model_args.get('llm_dim', 768)  # Default BERT dimension
        
        # Pre-training configuration
        self.from_scratch = model_args.get('from_scratch', True)
        self.pre_train_path = model_args.get('pre_train_path', None)
        self.database_path = model_args.get('database_path', './database')
        
        # Ensure database directory exists
        os.makedirs(self.database_path, exist_ok=True)
        print(f"Database path: {self.database_path}")
        
        # Initialize device and timing
        self._init_timing()
        
        # Initialize components
        self._init_components()
        self._init_llm_encoder()
        self._freeze_components()
        
        # Create ablation-specific components
        self._init_ablation_components()
        
        # Training settings
        self.prompt_cache = {}  # Cache for prompt generation
        self.retrieval_cache = {}  # Cache for retrieval results
    
    def _init_timing(self):
        """Initialize device settings and timing utilities with explicit CUDA setup"""
        import time
        import functools
        
        # Timing utilities
        self.module_timings = {}
        self.module_params = {}
        
        if self.timing_mode:
            print("⏱️ Timing mode enabled - will analyze module efficiency")
            
            def timing_decorator(func_name):
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
                        
                        if func_name not in self.module_timings:
                            self.module_timings[func_name] = []
                        self.module_timings[func_name].append(end_time - start_time)
                        
                        return result
                    return wrapper
                return decorator
            
            self.timing_decorator = timing_decorator

    def _init_components(self):
        """Initialize model components with proper weight initialization and regularization"""
        # Pre-trained encoder
        self.pre_train_encoder = self._load_weight()
        print(f"Pre-trained encoder initialized with {self.encoder_layers} layers")
        
        # Retrieval component with GPU support
        self.retrieval_store = RetrievalStore(
            self.retrieval_dim, 
            doc_dir=self.database_path,
            max_files=5,
            num_nodes=self.num_nodes,
            seq_len=self.seq_len,
            use_gpu=torch.cuda.is_available()  # Enable GPU for retrieval
        )
        
        # Projection layers with orthogonal initialization for better gradient flow
        self.query_proj = nn.Linear(self.embed_dim, self.retrieval_dim)
        self.key_proj = nn.Linear(self.retrieval_dim, self.retrieval_dim)
        self.value_proj = nn.Linear(self.retrieval_dim, self.retrieval_dim)
        
        # Initialize projection layers with orthogonal weights
        nn.init.orthogonal_(self.query_proj.weight)
        nn.init.orthogonal_(self.key_proj.weight)
        nn.init.orthogonal_(self.value_proj.weight)
        nn.init.zeros_(self.query_proj.bias)
        nn.init.zeros_(self.key_proj.bias)
        nn.init.zeros_(self.value_proj.bias)
        
        # Optimized attention mechanism 
        self.cross_attention = nn.MultiheadAttention(
            self.retrieval_dim, num_heads=4, batch_first=True, dropout=self.dropout  # Reduced heads for speed
        )
        
        # Combined dimension
        self.output_dim_combined = self.embed_dim + self.retrieval_dim
        
        # Simplified output projection for better performance
        self.out_proj = nn.Sequential(
            nn.Linear(self.output_dim_combined, self.horizon * self.output_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Initialize output projection layers
        for m in self.out_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Spatio-temporal feature processing
        self.fusion_dim = self.temporal_dim + self.spatial_dim  

        self.spatial_node_embeddings = nn.Parameter(torch.empty(self.num_nodes, self.spatial_dim))
        nn.init.xavier_uniform_(self.spatial_node_embeddings)
        
        self.temporal_series_encoder = nn.Conv2d(
            in_channels=self.input_dim * self.seq_len,
            out_channels=self.temporal_dim,
            kernel_size=(1, 1),
            bias=True
        )
        nn.init.kaiming_normal_(self.temporal_series_encoder.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.temporal_series_encoder.bias)
        
        # 4. Simplified feature encoding for better performance
        self.feature_encoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.fusion_dim, self.fusion_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ) for _ in range(max(1, self.encoder_layers // 2))  # Reduce layers for speed
        ])
        
        # Initialize feature encoder layers
        for layer in self.feature_encoder_layers:
            for m in layer.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        # 5. Regression layer with proper initialization
        self.regression_layer = nn.Linear(self.fusion_dim, self.horizon)
        nn.init.xavier_normal_(self.regression_layer.weight)
        nn.init.zeros_(self.regression_layer.bias)
        
        # 6. Prediction generator with proper initialization
        self.prediction_generator = nn.Conv2d(
            in_channels=self.fusion_dim,
            out_channels=self.horizon,
            kernel_size=(1, 1),
            bias=True
        )
        nn.init.kaiming_normal_(self.prediction_generator.weight, mode='fan_out', nonlinearity='linear')
        nn.init.zeros_(self.prediction_generator.bias)
        
        # 7. Output adapter with proper initialization
        self.output_projector = nn.Linear(self.output_dim, self.output_dim)
        nn.init.xavier_normal_(self.output_projector.weight)
        nn.init.zeros_(self.output_projector.bias)
        
        # Hidden to embedding projection with proper initialization
        self.hidden_to_embed_proj = nn.Linear(self.fusion_dim, self.embed_dim)
        nn.init.xavier_normal_(self.hidden_to_embed_proj.weight)
        nn.init.zeros_(self.hidden_to_embed_proj.bias)

    def _init_ablation_components(self):
        """Initialize components for different ablation modes with proper regularization"""
        # Use original dimensions (matching pre-trained weights)
        self.data_embed_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.LayerNorm(self.embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim * 2, self.output_dim_combined)
        )
        
        # For only_retrieval_embed mode - use original dimensions
        self.retrieval_embed_mlp = nn.Sequential(
            nn.Linear(self.retrieval_dim, self.retrieval_dim * 2),
            nn.LayerNorm(self.retrieval_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.retrieval_dim * 2, self.output_dim_combined)
        )
        
        # For full mode - use original dimensions
        self.combined_embed_mlp = nn.Sequential(
            nn.Linear(self.output_dim_combined, self.output_dim_combined),
            nn.LayerNorm(self.output_dim_combined),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.output_dim_combined, self.output_dim_combined)
        )
        
        # Initialize all MLP layers with proper weight initialization
        for module in [self.data_embed_mlp, self.retrieval_embed_mlp, self.combined_embed_mlp]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _init_llm_encoder(self):
        """Initialize LLM encoder"""
        try:
            print(f"Initializing LLM Encoder with model: {self.llm_model}")
            self.llm_encoder = LLMEncoder(model_name=self.llm_model)
            self.llm_proj = nn.Linear(self.llm_dim, self.retrieval_dim)
            print("LLM Encoder initialized successfully")
        except Exception as e:
            print(f"Error initializing LLM Encoder: {e}")
            print("Using a simplified encoder instead")
            self.llm_encoder = nn.Sequential(
                nn.Linear(self.retrieval_dim, self.llm_dim),
                nn.ReLU(),
                nn.Linear(self.llm_dim, self.llm_dim)
            )
            self.llm_proj = nn.Linear(self.llm_dim, self.retrieval_dim)
    
    def _freeze_components(self):
        """Freeze pre-trained components"""
        # Freeze pre-trained encoder
        for param in self.pre_train_encoder.parameters():
            param.requires_grad = False
        
        # Freeze LLM encoder
        for param in self.llm_encoder.parameters():
            param.requires_grad = False
            
        print("Pre-trained components frozen")

    def _load_weight(self):
        """Load pre-trained weights or initialize new encoder"""
        encoder = nn.ModuleList([
            TransformerEncoder(self.embed_dim, num_heads=8)
            for _ in range(self.encoder_layers)
        ])
        
        if not self.from_scratch and self.pre_train_path:
            try:
                state_dict = torch.load(self.pre_train_path)
                encoder.load_state_dict(state_dict)
                print(f"Successfully loaded pre-trained weights from {self.pre_train_path}")
            except Exception as e:
                print(f"Failed to load pre-trained weights: {e}")
                print("Training from scratch instead")
        
        return encoder
            
    def _generate_prompt(self, domain: str, data: torch.Tensor) -> Tuple[List[str], List[str]]:
        """Generate prompts for temporal and spatial dimensions
        
        Args:
            domain: Domain identifier
            data: Input tensor of shape [B, T, N, D]
            
        Returns:
            Tuple of (temporal_prompts, spatial_prompts):
            - temporal_prompts: List[str] of length T, prompts for each timestep
            - spatial_prompts: List[str] of length N, prompts for each node
        """
        B, T, N, D = data.shape
        
        # Average across batch dimension to get [T, N, D]
        data_mean = torch.mean(data, dim=0)
        
        # 1. Generate temporal prompts
        temporal_prompts = []
        for t in range(T):
            # Get data for current timestep [N, D]
            time_data = data_mean[t]
            
            # Calculate statistics for current timestep
            t_mean = torch.mean(time_data, dim=0)  # [D]
            t_std = torch.std(time_data, dim=0)    # [D]
            
            # Find max/min nodes for each feature dimension
            max_nodes, _ = torch.max(time_data, dim=0)  # [D]
            min_nodes, _ = torch.min(time_data, dim=0)  # [D]
            
            # Find most significant nodes based on variance
            node_variance = torch.var(time_data, dim=1)  # [N]
            _, top_k_nodes = torch.topk(node_variance, min(self.top_k, N))
            
            # Generate temporal prompt
            prompt = (
                f"<|domain|>{domain} prediction task<|domain_end|>\n"
                f"<|temporal_info|>\n"
                f"Timestep {t}: Global pattern\n"
                f"Average values: {t_mean.tolist()}\n"
                f"Pattern variation: {t_std.tolist()}\n"
                f"Max feature values: {max_nodes.tolist()}\n"
                f"Min feature values: {min_nodes.tolist()}\n"
                f"Most active nodes: {top_k_nodes.tolist()}\n"
                f"<|temporal_info_end|>\n"
            )
            temporal_prompts.append(prompt)
        
        # 2. Generate spatial prompts
        spatial_prompts = []
        # Transpose to [N, T, D] for processing each node's time series
        node_data_mean = data_mean.transpose(0, 1)
        
        for n in range(N):
            # Get time series data for current node [T, D]
            node_data = node_data_mean[n]
            
            # Calculate statistics for current node
            n_mean = torch.mean(node_data, dim=0)  # [D]
            n_std = torch.std(node_data, dim=0)    # [D]
            
            # Find peak times for each feature dimension
            max_times, _ = torch.max(node_data, dim=0)  # [D]
            min_times, _ = torch.min(node_data, dim=0)  # [D]
            
            # Find most active time periods
            time_variance = torch.var(node_data, dim=1)  # [T]
            _, top_k_times = torch.topk(time_variance, min(self.top_k, T))
            
            # Generate spatial prompt
            prompt = (
                f"<|domain|>{domain} prediction task<|domain_end|>\n"
                f"<|spatial_info|>\n"
                f"Node {n}: Temporal pattern\n"
                f"Average values: {n_mean.tolist()}\n"
                f"Pattern variation: {n_std.tolist()}\n"
                f"Max feature times: {max_times.tolist()}\n"
                f"Min feature times: {min_times.tolist()}\n"
                f"Most active times: {top_k_times.tolist()}\n"
                f"<|spatial_info_end|>\n"
            )
            spatial_prompts.append(prompt)
        
        cache_key = f"{domain}_{data.shape}_{torch.mean(data).item():.4f}"
        if cache_key in self.prompt_cache:
            return self.prompt_cache[cache_key]
        
        # 存入缓存
        self.prompt_cache[cache_key] = (temporal_prompts, spatial_prompts)
        # 限制缓存大小
        if len(self.prompt_cache) > 100:  # 保留最近的100个结果
            self.prompt_cache.pop(next(iter(self.prompt_cache)))
        
        return temporal_prompts, spatial_prompts
        
    @torch.no_grad()
    def update_retrieval_store(self, values: torch.Tensor, epoch: int):
        """Update retrieval store with new values
        
        Args:
            values: Input tensor of shape [B, T, N, D]
            epoch: Current training epoch
        """
        try:
            # Check if vectors already exist in the store
            has_existing_vectors = len(self.retrieval_store.temporal_vectors) > 0 and len(self.retrieval_store.spatial_vectors) > 0
            
            # Generate prompts for temporal and spatial dimensions
            temporal_prompts, spatial_prompts = self._generate_prompt(self.domain, values)
            
            # Optimize batch size for processing
            batch_size = 128  # Increased batch size for better efficiency
            
            # Process all prompts in a single loop to reduce redundancy
            all_prompts = temporal_prompts + spatial_prompts
            all_embeddings = []
            
            # Process embeddings with optional mixed precision
            if self.use_amp and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    for i in range(0, len(all_prompts), batch_size):
                        batch_prompts = all_prompts[i:i+batch_size]
                        batch_embeddings = self.llm_encoder(batch_prompts)
                        # Project to retrieval dimension
                        batch_embeddings = self.llm_proj(batch_embeddings)
                        all_embeddings.append(batch_embeddings.cpu().numpy())
            else:
                for i in range(0, len(all_prompts), batch_size):
                    batch_prompts = all_prompts[i:i+batch_size]
                    batch_embeddings = self.llm_encoder(batch_prompts)
                    # Project to retrieval dimension
                    batch_embeddings = self.llm_proj(batch_embeddings)
                    all_embeddings.append(batch_embeddings.cpu().numpy())
            
            # Concatenate all embeddings
            all_embeddings = np.vstack(all_embeddings)
            
            # Separate temporal and spatial embeddings
            temporal_embeddings = all_embeddings[:len(temporal_prompts)]
            spatial_embeddings = all_embeddings[len(temporal_prompts):]
            
            # Update retrieval store using incremental update if available
            if has_existing_vectors and hasattr(self.retrieval_store, 'incremental_update'):
                # Use incremental update method
                self.retrieval_store.incremental_update(
                    temporal_embeddings, 
                    spatial_embeddings,
                    max_vectors=1000  # Limit vector count to control memory usage
                )
            else:
                # Full update of retrieval store and rebuild indices
                self.retrieval_store.temporal_vectors = [v for v in temporal_embeddings]
                self.retrieval_store.spatial_vectors = [v for v in spatial_embeddings]
                self.retrieval_store._rebuild_indices()
            
            # Save documents for future use
            self.retrieval_store.save_documents(self.domain, epoch, temporal_prompts, spatial_prompts)
            
            # Log statistics
            print(f"Successfully updated retrieval store at epoch {epoch}")
            print(f"Statistics: {self.retrieval_store.get_statistics()}")
            
        except Exception as e:
            print(f"Error in update_retrieval_store: {e}")
            traceback.print_exc()

    def retrieve(self, query_embed: torch.Tensor, history_data: torch.Tensor) -> torch.Tensor:
        """Highly optimized retrieve method with minimal overhead"""
        B, L, N, E = query_embed.shape
        device = query_embed.device
        
        # 1. Check if retrieval store is initialized
        if not (self.retrieval_store.temporal_vectors and self.retrieval_store.spatial_vectors):
            return torch.zeros(B, L, N, self.retrieval_dim, device=device, dtype=query_embed.dtype)

        try:
            # Generate simple cache key
            query_hash = hash(query_embed.data_ptr())
            batch_key = f"{B}_{L}_{N}_{query_hash}"
            
            # Check retrieval cache first
            if batch_key in self.retrieval_cache:
                return self.retrieval_cache[batch_key].to(device, non_blocking=True)
            
            # Prepare query efficiently - reduce to 2D early 
            query_flat = query_embed.view(-1, E)  # [B*L*N, E]
            
            # Project query
            query_projected = self.query_proj(query_flat)  # [B*L*N, retrieval_dim]
            query_np = query_projected.detach().cpu().numpy().astype(np.float32)
            
            # Limit k for performance
            k_limit = min(self.top_k, 2)  # Further reduced for speed
            
            # Fast search - only temporal for speed
            temporal_distances, temporal_indices = self.retrieval_store.search(query_np, k=k_limit, temporal=True)
            
            # Quick vector retrieval
            if len(self.retrieval_store.temporal_vectors) > 0:
                # Clip indices and get vectors
                valid_indices = np.clip(temporal_indices.flatten(), 0, len(self.retrieval_store.temporal_vectors)-1)
                retrieved_vecs = np.array([self.retrieval_store.temporal_vectors[i] for i in valid_indices])
                retrieved_vecs = retrieved_vecs.reshape(query_np.shape[0], k_limit, -1)
                
                # Convert to tensor and average
                retrieved_tensor = torch.from_numpy(retrieved_vecs).to(device, dtype=query_embed.dtype)
                retrieved_mean = torch.mean(retrieved_tensor, dim=1)  # [B*L*N, retrieval_dim]
                retrieved_mean = retrieved_mean.view(B, L, N, self.retrieval_dim)
            else:
                retrieved_mean = torch.zeros(B, L, N, self.retrieval_dim, device=device, dtype=query_embed.dtype)
            
            # Cache result (small cache for speed)
            if len(self.retrieval_cache) < 50:
                self.retrieval_cache[batch_key] = retrieved_mean.cpu()
            elif len(self.retrieval_cache) >= 100:
                # Clear cache periodically
                self.retrieval_cache.clear()
            
            return retrieved_mean
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return torch.zeros(B, L, N, self.retrieval_dim, device=device, dtype=query_embed.dtype)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, 
               batch_seen: int, epoch: int, train: bool, **kwargs) -> dict:
        """
        Optimized forward pass with reduced overhead and better GPU utilization
        """
        B, L, N, D = history_data.shape
    
        if self.use_amp and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                return self._forward_impl(history_data, future_data, batch_seen, epoch, train)
        else:
            return self._forward_impl(history_data, future_data, batch_seen, epoch, train)
    
    def _forward_impl(self, history_data: torch.Tensor, future_data: torch.Tensor, 
                     batch_seen: int, epoch: int, train: bool) -> dict:
        """Implementation of forward pass"""
        B, L, N, D = history_data.shape
        device = history_data.device
        input_data = history_data[..., range(self.input_dim)]
        
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(B, N, -1).transpose(1, 2).unsqueeze(-1)
        temp_embed = self.temporal_series_encoder(input_data)
        node_embed = self.spatial_node_embeddings.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1)

        hidden = torch.cat([temp_embed, node_embed], dim=1)
        
        hidden_permuted = hidden.squeeze(-1).transpose(1, 2)  # [B, N, fusion_dim]
        
        for layer in self.feature_encoder_layers:
            hidden_permuted = layer(hidden_permuted) + hidden_permuted  # Residual connection
        
        if self.output_type == "only_data_embed":
            prediction = self.regression_layer(hidden_permuted)
            prediction = prediction.unsqueeze(-1).expand(-1, -1, -1, self.output_dim)
            prediction = self.output_projector(prediction.reshape(-1, self.output_dim))
            prediction = prediction.reshape(B, N, self.horizon, self.output_dim)
            prediction = prediction.permute(0, 2, 1, 3)
            
            return {'prediction': prediction}
        
        # "full" or "only_retrieval_embed" mode
        # Project to embedding dimension
        query_embed = self.hidden_to_embed_proj(hidden_permuted)
        query_embed = query_embed.unsqueeze(1).expand(-1, L, -1, -1)
        
        # Initialize retrieval embeddings
        if self.output_type == "without_retrieval":
            retrieval_embed = torch.zeros(B, L, N, self.retrieval_dim, device=device, dtype=query_embed.dtype)
        else:
            retrieval_embed = self.retrieval_embed_mlp(query_embed)

        # Retrieval processing with smart updates
        if self.output_type in ["full", "only_retrieval_embed"]:
            should_update_retrieval = (
                train and 
                epoch % max(10, self.update_interval) == 0 and  # Use larger interval 
                batch_seen == 0 and
                not hasattr(self, f'updated_epoch_{epoch}')
            )
            
            if should_update_retrieval:
                print(f"Updating retrieval store at epoch {epoch}")
                self.update_retrieval_store(history_data, epoch)
                setattr(self, f'updated_epoch_{epoch}', True)
            
            # Perform retrieval
            if self.retrieval_store.temporal_vectors and self.retrieval_store.spatial_vectors:
                query = self.query_proj(query_embed)
                retrieval_embed = self.retrieve(query, history_data)

        # Process embeddings based on mode
        if self.output_type == "only_retrieval_embed":
            final_embed = self.retrieval_embed_mlp(retrieval_embed)
        else:  # "full" mode
            combined = torch.cat([query_embed, retrieval_embed], dim=-1)
            final_embed = self.combined_embed_mlp(combined)
        
        # Generate predictions efficiently
        B_new, L_new, N_new, E_new = final_embed.shape
        final_embed_flat = final_embed.reshape(-1, E_new)
        out_flat = self.out_proj(final_embed_flat)
        
        out = out_flat.reshape(B_new, L_new, N_new, self.horizon * self.output_dim)
        out = out.mean(dim=1)  # Aggregate time dimension
        out = out.reshape(B_new, N_new, self.horizon, self.output_dim)
        out = out.permute(0, 2, 1, 3)  # [B, horizon, N, output_dim]
        
        return {'prediction': out}

    def _record_timing(self, module_name: str, duration: float):
        if module_name not in self.module_timings:
            self.module_timings[module_name] = []
        self.module_timings[module_name].append(duration)
    
    def _count_parameters(self, module):
        """Count parameters in a module"""
        if hasattr(module, 'parameters'):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        return 0
    
    def _analyze_efficiency_and_exit(self):
        """Analyze module efficiency and exit"""
        print("\n" + "="*80)
        print("🔍 RAST MODEL EFFICIENCY ANALYSIS")
        print("="*80)
        
        # Collect parameter counts
        modules = {
            'temporal_series_encoder': self.temporal_series_encoder,
            'spatial_node_embeddings': self.spatial_node_embeddings,
            'feature_encoder_layers': self.feature_encoder_layers,
            'query_proj': self.query_proj,
            'key_proj': self.key_proj,
            'value_proj': self.value_proj,
            'cross_attention': self.cross_attention,
            'out_proj': self.out_proj,
            'llm_encoder': self.llm_encoder,
            'llm_proj': self.llm_proj,
        }
        
        total_params = 0
        print(f"{'Module':<25} {'Parameters':<15} {'Avg Time (ms)':<15} {'Efficiency':<15}")
        print("-" * 70)
        
        for module_name, module in modules.items():
            param_count = self._count_parameters(module)
            total_params += param_count
            
            if module_name in self.module_timings:
                avg_time = np.mean(self.module_timings[module_name]) * 1000  # Convert to ms
                efficiency = param_count / avg_time if avg_time > 0 else float('inf')
                print(f"{module_name:<25} {param_count:<15,} {avg_time:<15.3f} {efficiency:<15,.0f}")
            else:
                print(f"{module_name:<25} {param_count:<15,} {'N/A':<15} {'N/A':<15}")
        
        print("-" * 70)
        print(f"{'TOTAL':<25} {total_params:<15,}")
        
        # Timing analysis
        print("\n📊 TIMING BREAKDOWN:")
        total_time = sum(np.mean(times) for times in self.module_timings.values()) * 1000
        
        for module_name, times in self.module_timings.items():
            avg_time = np.mean(times) * 1000
            percentage = (avg_time / total_time) * 100 if total_time > 0 else 0
            print(f"  {module_name:<25}: {avg_time:>8.3f}ms ({percentage:>5.1f}%)")
        
        print(f"\n⚡ Total Forward Time: {total_time:.3f}ms")
        print(f"🚀 GPU Device: {self.device}")
        print(f"🔧 Mixed Precision: {'Enabled' if self.use_amp else 'Disabled'}")
        
        # Performance recommendations
        print("\n💡 PERFORMANCE RECOMMENDATIONS:")
        sorted_times = sorted(self.module_timings.items(), 
                            key=lambda x: np.mean(x[1]), reverse=True)
        
        print("  Top 3 Time-Consuming Modules:")
        for i, (module_name, times) in enumerate(sorted_times[:3]):
            avg_time = np.mean(times) * 1000
            print(f"    {i+1}. {module_name}: {avg_time:.3f}ms")
        
        print("\n🚨 TIMING MODE - EXITING AFTER ANALYSIS")
        print("="*80)
        exit(0)

    def _initialize_retrieval_store(self, history_data: torch.Tensor, epoch: int):
        """Initialize retrieval store with current batch data
        
        This method creates initial temporal and spatial embeddings for the retrieval store
        using the current batch data. It's called when the retrieval store is empty.
        
        Args:
            history_data: Historical input data [B, L, N, C]
            epoch: Current training epoch
        """
        try:
            # Generate temporal and spatial prompts
            temporal_prompts, spatial_prompts = self._generate_prompt(self.domain, history_data)
            
            # Initialize empty lists for embeddings
            temporal_embeddings = []
            spatial_embeddings = []
            
            # Process prompts in batches for efficiency
            batch_size = 128  # Increased batch size for better efficiency
            
            # Process temporal prompts
            for i in range(0, len(temporal_prompts), batch_size):
                batch_prompts = temporal_prompts[i:i+batch_size]
                with torch.no_grad():
                    batch_embeddings = self.llm_encoder(batch_prompts)
                    batch_embeddings = self.llm_proj(batch_embeddings)
                    temporal_embeddings.append(batch_embeddings.cpu().numpy())
            
            # Process spatial prompts
            for i in range(0, len(spatial_prompts), batch_size):
                batch_prompts = spatial_prompts[i:i+batch_size]
                with torch.no_grad():
                    batch_embeddings = self.llm_encoder(batch_prompts)
                    batch_embeddings = self.llm_proj(batch_embeddings)
                    spatial_embeddings.append(batch_embeddings.cpu().numpy())
            
            # Concatenate all embeddings
            if temporal_embeddings:
                temporal_embeddings = np.vstack(temporal_embeddings)
            else:
                temporal_embeddings = np.zeros((0, self.retrieval_dim))
            
            if spatial_embeddings:
                spatial_embeddings = np.vstack(spatial_embeddings)
            else:
                spatial_embeddings = np.zeros((0, self.retrieval_dim))
            
            # Update retrieval store
            self.retrieval_store.temporal_vectors = [v for v in temporal_embeddings]
            self.retrieval_store.spatial_vectors = [v for v in spatial_embeddings]
            
            # Update temporal and spatial values using the update_patterns method
            self.retrieval_store.update_patterns(history_data)
            
            # Rebuild indices
            self.retrieval_store._rebuild_indices()
            
            # Save documents
            self.retrieval_store.save_documents(
                self.domain, 
                epoch, 
                temporal_prompts, 
                spatial_prompts
            )
            
            print(f"Successfully initialized retrieval store at epoch {epoch}")
            print(f"Statistics: {self.retrieval_store.get_statistics()}")
            
        except Exception as e:
            print(f"Error initializing retrieval store: {e}")
            traceback.print_exc()
            # Create empty store as fallback
            self.retrieval_store.temporal_vectors = []
            self.retrieval_store.spatial_vectors = []
            self.retrieval_store.temporal_values = []
            self.retrieval_store.spatial_values = []