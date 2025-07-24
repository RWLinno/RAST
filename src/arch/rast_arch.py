import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np
import os
import traceback
from .blocks.Embed import *
from .blocks.Pretrain import *
from .blocks.RetrievalStore import *
from .blocks.Text_Encoder import *
from .blocks.utils import load_pretrained_stgnn
from .blocks.MLP import MultiLayerPerceptron
#from .blocks.MHA import _MultiheadAttention

class RAST(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.num_nodes = model_args['num_nodes'] 
        self.input_dim = model_args['input_dim']
        self.output_dim = model_args['output_dim']
        self.embed_dim = model_args.get('embed_dim', 128)
        self.query_dim = model_args.get('query_dim', 32)
        self.retrieval_dim = model_args.get('retrieval_dim', 64)
        self.temporal_dim = model_args.get('temporal_dim', 64)
        self.spatial_dim = model_args.get('spatial_dim', 64)
        self.seq_len = model_args['input_len']
        self.horizon = model_args['output_len']
        self.encoder_layers = model_args.get('encoder_layers', 1)
        self.top_k = model_args.get('top_k', 3)
        self.dropout = model_args.get('dropout', 0.1)
        self.batch_size = model_args.get('batch_size', 32)
        self.add_query = True

#        self.decoder_layers = model_args.get('decoder_layers', 1)
#        self.patch_size = model_args.get('patch_size', 32)
#        self.domain = model_args.get('prompt_domain', 'PEMS04')
#        self.llm_dim = model_args.get('llm_dim', 768)
#        self.device_id = model_args.get('device_id', 0)  # Single GPU device ID

        self.update_epoch = 0
        self.update_interval = model_args.get('update_interval', 5)  # Default to 20 epochs for better training speed
        self.output_type = 'full' #model_args.get('output_type', 'full')  # 'full', or 'only_retrieval_embed'
        print(f"Output type: {self.output_type}")

        self.timing_mode = model_args.get('timing_mode', False)  # Enable detailed timing analysis
        self.use_amp = model_args.get('use_amp', False)  # Disable AMP by default to fix GPU issues
        
        self.pre_train_model_name = model_args.get('pre_train_model_name', '')
        self.pre_train_path = model_args.get('pre_train_path', None)
        self.database_path = model_args.get('database_path', './database')

        if self.pre_train_path is not None and self.pre_train_model_name != '':
            try:
                self.backbone = load_pretrained_stgnn(
                    pretrain_path=self.pre_train_path, 
                    model_name=self.pre_train_model_name,
                )
                self.backbone.eval()
                for param in self.backbone.parameters():
                    param.requires_grad = False
                print(f"Loaded pre-trained {self.pre_train_model_name} from {self.pre_train_path}")
            except Exception as e:
                print(f"Error loading pre-trained model: {e}")
                print("Using default backbone")
        else:
            self.backbone = None

        self.mlp_predictor = MultiLayerPerceptron(
            input_dim=self.retrieval_dim + self.query_dim,
            output_dim=self.output_dim * self.horizon,
            dropout=self.dropout
        )

        os.makedirs(self.database_path, exist_ok=True)
        print(f"Database path: {self.database_path}")

        #self._init_timing()
        self._init_components()
        self.prompt_cache = {}
        self.retrieval_cache = {}
    
    def _init_timing(self):
        import time
        import functools
        
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
        self.retrieval_store = RetrievalStore(
            self.retrieval_dim, 
            doc_dir=self.database_path,
            max_files=10,
            num_nodes=self.num_nodes,
            seq_len=self.seq_len,
            use_gpu=torch.cuda.is_available()  # Enable GPU for retrieval
        )
        
        self.query_retrieval_proj = nn.Linear(self.query_dim, self.retrieval_dim)
        self.retrieval_ouput_proj = nn.Linear(self.retrieval_dim, self.horizon*self.output_dim)

        self.temporal_proj = nn.Linear(self.temporal_dim, self.retrieval_dim)
        self.spatial_proj = nn.Linear(self.spatial_dim, self.retrieval_dim)
        
        nn.init.orthogonal_(self.query_retrieval_proj.weight)
        nn.init.orthogonal_(self.temporal_proj.weight)
        nn.init.orthogonal_(self.spatial_proj.weight)
        nn.init.zeros_(self.query_retrieval_proj.bias)
        nn.init.zeros_(self.temporal_proj.bias)
        nn.init.zeros_(self.spatial_proj.bias)
        
        self.fusion_dim = self.temporal_dim + self.spatial_dim  

        self.spatial_encoder = nn.Parameter(torch.empty(self.num_nodes, self.spatial_dim))
        nn.init.xavier_uniform_(self.spatial_encoder)
        
        self.temporal_encoder = nn.Conv2d(
            in_channels=self.input_dim * self.seq_len,
            out_channels=self.temporal_dim,
            kernel_size=(1, 1),
            bias=True
        )
        nn.init.kaiming_normal_(self.temporal_encoder.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.temporal_encoder.bias)

        self.feature_encoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.fusion_dim, self.fusion_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ) for _ in range(self.encoder_layers)
        ])
        
        for layer in self.feature_encoder_layers:
            for m in layer.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        self.horizon_layer = nn.Linear(self.fusion_dim, self.horizon)
        nn.init.xavier_normal_(self.horizon_layer.weight)
        nn.init.zeros_(self.horizon_layer.bias)

        #self.attn = _MultiheadAttention(d_model=self.retrieval_dim, n_heads=8)
        self.attn = nn.MultiheadAttention(num_heads=4, embed_dim=self.retrieval_dim, dropout=0.1)
        self.output_projector = nn.Linear(self.output_dim, self.output_dim)
        nn.init.xavier_normal_(self.output_projector.weight)
        nn.init.zeros_(self.output_projector.bias)
        
        self.hidden_to_query_proj = nn.Linear(self.fusion_dim, self.query_dim)
        nn.init.xavier_normal_(self.hidden_to_query_proj.weight)
        nn.init.zeros_(self.hidden_to_query_proj.bias)

    @torch.no_grad()
    def _update_retrieval_tensors(self, temp_embed: torch.Tensor, node_embed: torch.Tensor, 
                                 history_data: torch.Tensor, epoch: int):
        """Update retrieval store with temporal and spatial four-dimensional tensors"""
        epoch_id = epoch % 50 # CLAIM: Limit 50 epoch store
        max_tensors = 20      # CLAIM: Keep 20 most recent tensor snapshots
        max_vectors = 1000    # CLAIM: Keep 1000 most recent vectors

        try:
            B, temporal_dim, N, _ = temp_embed.shape
            B, spatial_dim, N, _ = node_embed.shape
            
            temp_processed = temp_embed.squeeze(-1)  # [B, temporal_dim, N]
            node_processed = node_embed.squeeze(-1)  # [B, spatial_dim, N]
            
            temp_4d = temp_processed.unsqueeze(-1).expand(-1, -1, -1, epoch_id + 1)  # [B, temporal_dim, N, epoch_id]
            node_4d = node_processed.unsqueeze(-1).expand(-1, -1, -1, epoch_id + 1)  # [B, spatial_dim, N, epoch_id]
            
            temp_flat = temp_processed.reshape(-1, temporal_dim)  # [B*N, temporal_dim]
            node_flat = node_processed.reshape(-1, spatial_dim)    # [B*N, spatial_dim]
            
            temp_proj = self.temporal_proj(temp_flat)  # [B*N, retrieval_dim]
            node_proj = self.spatial_proj(node_flat)   # [B*N, retrieval_dim]
            
            # Store both projected vectors for FAISS and original 4D tensors for retrieval
            if not hasattr(self.retrieval_store, 'temporal_tensors'):
                self.retrieval_store.temporal_tensors = []
                self.retrieval_store.spatial_tensors = []
            
            # Convert to numpy for storage
            temp_tensor_np = temp_4d.detach().cpu().numpy().astype(np.float32)
            node_tensor_np = node_4d.detach().cpu().numpy().astype(np.float32)
            temp_proj_np = temp_proj.detach().cpu().numpy().astype(np.float32)
            node_proj_np = node_proj.detach().cpu().numpy().astype(np.float32)
            
            # Store 4D tensors for retrieval
            self.retrieval_store.temporal_tensors.append(temp_tensor_np)
            self.retrieval_store.spatial_tensors.append(node_tensor_np)
            
            # Update FAISS vectors for indexing
            for vec in temp_proj_np:
                self.retrieval_store.temporal_vectors.append(vec)
            for vec in node_proj_np:
                self.retrieval_store.spatial_vectors.append(vec)
            
            # Limit memory by keeping only recent tensors and vectors
            if len(self.retrieval_store.temporal_tensors) > max_tensors:
                self.retrieval_store.temporal_tensors = self.retrieval_store.temporal_tensors[-max_tensors:]
                self.retrieval_store.spatial_tensors = self.retrieval_store.spatial_tensors[-max_tensors:]
            
            if len(self.retrieval_store.temporal_vectors) > max_vectors:
                self.retrieval_store.temporal_vectors = self.retrieval_store.temporal_vectors[-max_vectors:]
            if len(self.retrieval_store.spatial_vectors) > max_vectors:
                self.retrieval_store.spatial_vectors = self.retrieval_store.spatial_vectors[-max_vectors:]

            self.retrieval_store._rebuild_indices() # CLAIM: for efficient similarity search
            
        except Exception as e:
            print(f"Error updating retrieval tensors: {e}")
            traceback.print_exc()

    def _retrieve_tensors(self, query: torch.Tensor, history_data: torch.Tensor, 
                         embed: torch.Tensor, temporal: bool = True) -> torch.Tensor:
        B, N, E = query.shape  # query is [B, N, query_dim]
        device = query.device
        
        if temporal:
            vectors = self.retrieval_store.temporal_vectors
            tensors = getattr(self.retrieval_store, 'temporal_tensors', [])
            proj_layer = self.temporal_proj
        else:
            vectors = self.retrieval_store.spatial_vectors
            tensors = getattr(self.retrieval_store, 'spatial_tensors', [])
            proj_layer = self.spatial_proj
        
        if not vectors or not tensors:
            return torch.zeros(B, N, self.retrieval_dim, device=device, dtype=query.dtype)
        
        query_flat = query.reshape(-1, E)  # [B*N, query_dim] -> 2D for FAISS
        query_projected = self.query_retrieval_proj(query_flat)  # [B*N, retrieval_dim]
        query_np = query_projected.detach().cpu().numpy().astype(np.float32)
        
        if not query_np.flags['C_CONTIGUOUS']:
            query_np = np.ascontiguousarray(query_np)
        
        if query_np.ndim != 2:
            raise ValueError(f"FAISS expects 2D array, got {query_np.ndim}D array with shape {query_np.shape}")
        
        k_limit = min(self.top_k, 3)
        distances, indices = self.retrieval_store.search(query_np, k=k_limit, temporal=temporal)
        
        if len(tensors) > 0:
            tensor_sample = tensors[0]
            num_vectors_per_tensor = tensor_sample.shape[0] * tensor_sample.shape[2]  # B*N
            
            tensor_indices = indices.flatten() // max(1, num_vectors_per_tensor)
            tensor_indices = np.clip(tensor_indices, 0, len(tensors)-1)
            unique_tensor_indices = np.unique(tensor_indices)[:k_limit]
            
            retrieved_tensors = []
            for tensor_idx in unique_tensor_indices:
                tensor_4d = tensors[tensor_idx]  # [B, embed_dim, N, epoch_steps]
                tensor_torch = torch.from_numpy(tensor_4d).to(device, dtype=query.dtype)
                
                tensor_avg = torch.mean(tensor_torch, dim=-1)  # [B, embed_dim, N]
                tensor_flat = tensor_avg.permute(0, 2, 1).reshape(-1, tensor_avg.shape[1])  # [B*N, embed_dim]
                tensor_proj = proj_layer(tensor_flat)  # [B*N, retrieval_dim]
                tensor_reshaped = tensor_proj.reshape(tensor_avg.shape[0], tensor_avg.shape[2], -1)  # [B, N, retrieval_dim]
                retrieved_tensors.append(tensor_reshaped)
            
            if retrieved_tensors:
                stacked_tensors = torch.stack(retrieved_tensors, dim=0)  # [k, B, N, retrieval_dim]
                averaged_tensor = torch.mean(stacked_tensors, dim=0)  # [B, N, retrieval_dim]
                return averaged_tensor
        
        return torch.zeros(B, N, self.retrieval_dim, device=device, dtype=query.dtype)

    def temporal_retriever(self, query: torch.Tensor, history_data: torch.Tensor, temp_embed: torch.Tensor) -> torch.Tensor:
        return self._retrieve_tensors(query, history_data, temp_embed, temporal=True)
    
    def spatial_retriever(self, query: torch.Tensor, history_data: torch.Tensor, node_embed: torch.Tensor) -> torch.Tensor:
        return self._retrieve_tensors(query, history_data, node_embed, temporal=False)
    
    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, 
               batch_seen: int, epoch: int, train: bool, **kwargs) -> dict:
        B, L, N, D = history_data.shape
    
        if self.use_amp and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                return self._forward_impl(history_data, future_data, batch_seen, epoch, train)
        else:
            return self._forward_impl(history_data, future_data, batch_seen, epoch, train)
    
    def _forward_impl(self, history_data: torch.Tensor, future_data: torch.Tensor, 
                     batch_seen: int, epoch: int, train: bool) -> dict:
        B, L, N, D = history_data.shape

        input_data = history_data[..., range(self.input_dim)]
        
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(B, N, -1).transpose(1, 2).unsqueeze(-1)
        
        temp_embed = self.temporal_encoder(input_data)
        node_embed = self.spatial_encoder.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1)

        should_update_retrieval = (
            train and self.output_type in ["full", "only_retrieval_embed","without_query_embedding"] and
            epoch % self.update_interval == 0 and self.update_epoch != epoch)
        
        if should_update_retrieval:
            self.update_epoch = epoch
            print(f"Updating retrieval tensors at epoch {epoch}")
            self._update_retrieval_tensors(temp_embed, node_embed, history_data, epoch)

        hidden = torch.cat([temp_embed, node_embed], dim=1)
        feature_fusion = hidden.squeeze(-1).transpose(1, 2)  # [B, N, fusion_dim]
        
        for layer in self.feature_encoder_layers:
            feature_fusion = layer(feature_fusion) + feature_fusion
        
        query_embed = self.horizon_layer(feature_fusion) # [B,N,horizon] => [64,307,12]

        prediction = query_embed.unsqueeze(-1).expand(-1, -1, -1, self.output_dim) # [B,N,horizon,output_dim] => [64,307,12,1]
        prediction = self.output_projector(query_embed.reshape(-1, self.output_dim))
        prediction = prediction.reshape(B, N, self.horizon, self.output_dim)
        prediction = prediction.permute(0, 2, 1, 3)

        if self.output_type == "only_query": # Ablation Study            
            return {'prediction': prediction}
        
        if self.backbone is not None:
            prediction = self.backbone(history_data, future_data, batch_seen, epoch, train)

        query_embed = self.hidden_to_query_proj(feature_fusion)  # [B, N, query_dim]

        temporal_retrieval = self.temporal_retriever(query_embed, history_data, temp_embed) #[B,N,retrieval_dim]
        spatial_retrieval = self.spatial_retriever(query_embed, history_data, node_embed) #[B,N,retrieval_dim]
        
        if self.output_type == "without_temporal_retrieval":
            temporal_retrieval = torch.zeros_like(temporal_retrieval)
        if self.output_type == "without_spatial_retrieval":
            spatial_retrieval = torch.zeros_like(spatial_retrieval)
        if self.output_type == "only_retrieval":
            query_embed = torch.zeros_like(query_embed)
            retrieval_embed = torch.cat([temporal_retrieval, spatial_retrieval], dim=-1)
            output = self.retrieval_ouput_proj(retrieval_embed)
            output = output.view(B, N, self.horizon, self.output_dim).transpose(1, 2)  # [B, Horizon, N, output_dim]
            return {'prediction': output}

        query_retrieval = self.query_retrieval_proj(query_embed)
        temporal_retrieval = self.attn(query_retrieval, temporal_retrieval, temporal_retrieval)[0]
        spatial_retrieval = self.attn(query_retrieval, spatial_retrieval, spatial_retrieval)[0]
        retrieval_embed = self.attn(query_retrieval, temporal_retrieval, spatial_retrieval)[0]

        if self.output_type == "without_retrieval_embedding": # Ablation
            retrieval_embed = torch.zeros_like(retrieval_embed) # [B,N,retrieval_dim]
        elif self.output_type == "without_query_embedding":
            query_embed = torch.zeros_like(query_embed) # [B,N,query_dim]

        combined_embed = torch.cat([retrieval_embed, query_embed], dim=-1) # [B,N,retrieval_dim+query_dim]
        
        output = self.mlp_predictor(combined_embed) # [B,N,retrieval_dim+query_dim] -> [B,N,H*self.output_dim]
        
        output = output.view(B, N, self.horizon, self.output_dim).transpose(1, 2)  # [B, Horizon, N, output_dim]
        
        if self.add_query:
            output = output + prediction
        
        return {'prediction': output}

    # def _init_encoder(self): # not important TODO: RAST utilized RAG augmenting pre-trained STGNNs in place of LLM.
    #     self.pre_train_encoder = nn.ModuleList([
    #         TransformerEncoder(self.embed_dim, num_heads=8)
    #         for _ in range(self.encoder_layers)
    #     ])
    #     if self.pre_train_path:
    #         try:
    #             state_dict = torch.load(self.pre_train_path)
    #             self.pre_train_encoder.load_state_dict(state_dict)
    #             print(f"Successfully loaded pre-trained weights from {self.pre_train_path}")
    #         except Exception as e:
    #             print(f"Failed to load pre-trained weights: {e}")
    #             print("Training from scratch instead")

    #     print(f"Pre-trained encoder initialized with {self.encoder_layers} layers")

    #     try:
    #         self.llm_encoder = nn.Sequential(
    #             nn.Linear(self.retrieval_dim, self.llm_dim),
    #             nn.ReLU(),
    #             nn.Linear(self.llm_dim, self.llm_dim)
    #         )
    #         self.llm_proj = nn.Linear(self.llm_dim, self.retrieval_dim)
    #     except Exception as e:
    #         print(f"Error initializing LLM Encoder: {e}")
    #         self.llm_encoder = nn.Sequential(
    #             nn.Linear(self.retrieval_dim, self.llm_dim),
    #             nn.ReLU(),
    #             nn.Linear(self.llm_dim, self.llm_dim)
    #         )
    #         self.llm_proj = nn.Linear(self.llm_dim, self.retrieval_dim)
        
    #     self._freeze_components()
    
    # def _freeze_components(self):
    #     for param in self.pre_train_encoder.parameters():
    #         param.requires_grad = False
        
    #     for param in self.llm_encoder.parameters():
    #         param.requires_grad = False

    # def _generate_prompt(self, domain: str, data: torch.Tensor) -> Tuple[List[str], List[str]]:
    #     """Generate prompts for temporal and spatial dimensions
        
    #     Args:
    #         domain: Domain identifier
    #         data: Input tensor of shape [B, T, N, D]
            
    #     Returns:
    #         Tuple of (temporal_prompts, spatial_prompts):
    #         - temporal_prompts: List[str] of length T, prompts for each timestep
    #         - spatial_prompts: List[str] of length N, prompts for each node
    #     """
    #     B, T, N, D = data.shape
    #     data_mean = torch.mean(data, dim=0)
    #     temporal_prompts = []
    #     for t in range(T):
    #         time_data = data_mean[t]
            
    #         t_mean = torch.mean(time_data, dim=0)  # [D]
    #         t_std = torch.std(time_data, dim=0)    # [D]
            
    #         max_nodes, _ = torch.max(time_data, dim=0)  # [D]
    #         min_nodes, _ = torch.min(time_data, dim=0)  # [D]
            
    #         node_variance = torch.var(time_data, dim=1)  # [N]
    #         _, top_k_nodes = torch.topk(node_variance, min(self.top_k, N))
            
    #         prompt = (
    #             f"<|domain|>{domain} prediction task<|domain_end|>\n"
    #             f"<|temporal_info|>\n"
    #             f"Timestep {t}: Global pattern\n"
    #             f"Average values: {t_mean.tolist()}\n"
    #             f"Pattern variation: {t_std.tolist()}\n"
    #             f"Max feature values: {max_nodes.tolist()}\n"
    #             f"Min feature values: {min_nodes.tolist()}\n"
    #             f"Most active nodes: {top_k_nodes.tolist()}\n"
    #             f"<|temporal_info_end|>\n"
    #         )
    #         temporal_prompts.append(prompt)
        
    #     spatial_prompts = []
    #     node_data_mean = data_mean.transpose(0, 1)
    #     for n in range(N):
    #         node_data = node_data_mean[n]
    #         n_mean = torch.mean(node_data, dim=0)  # [D]
    #         n_std = torch.std(node_data, dim=0)    # [D]
    #         max_times, _ = torch.max(node_data, dim=0)  # [D]
    #         min_times, _ = torch.min(node_data, dim=0)  # [D]
    #         time_variance = torch.var(node_data, dim=1)  # [T]
    #         _, top_k_times = torch.topk(time_variance, min(self.top_k, T))
    #         prompt = (
    #             f"<|domain|>{domain} prediction task<|domain_end|>\n"
    #             f"<|spatial_info|>\n"
    #             f"Node {n}: Temporal pattern\n"
    #             f"Average values: {n_mean.tolist()}\n"
    #             f"Pattern variation: {n_std.tolist()}\n"
    #             f"Max feature times: {max_times.tolist()}\n"
    #             f"Min feature times: {min_times.tolist()}\n"
    #             f"Most active times: {top_k_times.tolist()}\n"
    #             f"<|spatial_info_end|>\n"
    #         )
    #         spatial_prompts.append(prompt)
        
    #     cache_key = f"{domain}_{data.shape}_{torch.mean(data).item():.4f}"
    #     if cache_key in self.prompt_cache:
    #         return self.prompt_cache[cache_key]
        
    #     self.prompt_cache[cache_key] = (temporal_prompts, spatial_prompts)
    #     if len(self.prompt_cache) > 100:
    #         self.prompt_cache.pop(next(iter(self.prompt_cache)))
        
    #     return temporal_prompts, spatial_prompts

    # def _record_timing(self, module_name: str, duration: float):
    #     if module_name not in self.module_timings:
    #         self.module_timings[module_name] = []
    #     self.module_timings[module_name].append(duration)
    
    # def _count_parameters(self, module):
    #     """Count parameters in a module"""
    #     if hasattr(module, 'parameters'):
    #         return sum(p.numel() for p in module.parameters() if p.requires_grad)
    #     return 0
    
    # def _analyze_efficiency_and_exit(self):
    #     """Analyze module efficiency and exit"""
    #     print("\n" + "="*80)
    #     print("🔍 RAST MODEL EFFICIENCY ANALYSIS")
    #     print("="*80)
        
    #     # Collect parameter counts
    #     modules = {
    #         'temporal_encoder': self.temporal_encoder,
    #         'spatial_encoder': self.spatial_encoder,
    #         'feature_encoder_layers': self.feature_encoder_layers,
    #         'attn': self.attn,
    #         'backbone': self.backbone,
    #         'llm_encoder': self.llm_encoder,
    #         'llm_proj': self.llm_proj,
    #     }
        
    #     total_params = 0
    #     print(f"{'Module':<25} {'Parameters':<15} {'Avg Time (ms)':<15} {'Efficiency':<15}")
    #     print("-" * 70)
        
    #     for module_name, module in modules.items():
    #         param_count = self._count_parameters(module)
    #         total_params += param_count
            
    #         if module_name in self.module_timings:
    #             avg_time = np.mean(self.module_timings[module_name]) * 1000  # Convert to ms
    #             efficiency = param_count / avg_time if avg_time > 0 else float('inf')
    #             print(f"{module_name:<25} {param_count:<15,} {avg_time:<15.3f} {efficiency:<15,.0f}")
    #         else:
    #             print(f"{module_name:<25} {param_count:<15,} {'N/A':<15} {'N/A':<15}")
        
    #     print("-" * 70)
    #     print(f"{'TOTAL':<25} {total_params:<15,}")
        
    #     # Timing analysis
    #     print("\n📊 TIMING BREAKDOWN:")
    #     total_time = sum(np.mean(times) for times in self.module_timings.values()) * 1000
        
    #     for module_name, times in self.module_timings.items():
    #         avg_time = np.mean(times) * 1000
    #         percentage = (avg_time / total_time) * 100 if total_time > 0 else 0
    #         print(f"  {module_name:<25}: {avg_time:>8.3f}ms ({percentage:>5.1f}%)")
        
    #     print(f"\n⚡ Total Forward Time: {total_time:.3f}ms")
    #     print(f"🚀 GPU Device: {self.device}")
    #     print(f"🔧 Mixed Precision: {'Enabled' if self.use_amp else 'Disabled'}")
        
    #     # Performance recommendations
    #     print("\n💡 PERFORMANCE RECOMMENDATIONS:")
    #     sorted_times = sorted(self.module_timings.items(), 
    #                         key=lambda x: np.mean(x[1]), reverse=True)
        
    #     print("  Top 3 Time-Consuming Modules:")
    #     for i, (module_name, times) in enumerate(sorted_times[:3]):
    #         avg_time = np.mean(times) * 1000
    #         print(f"    {i+1}. {module_name}: {avg_time:.3f}ms")
        
    #     print("\n🚨 TIMING MODE - EXITING AFTER ANALYSIS")
    #     print("="*80)
    #     exit(0)

    # def _initialize_retrieval_store(self, history_data: torch.Tensor, epoch: int):
    #     """Initialize retrieval store with current batch data
        
    #     This method creates initial temporal and spatial embeddings for the retrieval store
    #     using the current batch data. It's called when the retrieval store is empty.
        
    #     Args:
    #         history_data: Historical input data [B, L, N, C]
    #         epoch: Current training epoch
    #     """
    #     try:
    #         # Generate temporal and spatial prompts
    #         temporal_prompts, spatial_prompts = self._generate_prompt(self.domain, history_data)
            
    #         # Initialize empty lists for embeddings
    #         temporal_embeddings = []
    #         spatial_embeddings = []
            
    #         # Process prompts in batches for efficiency
    #         batch_size = 128  # Increased batch size for better efficiency
            
    #         # Process temporal prompts
    #         for i in range(0, len(temporal_prompts), batch_size):
    #             batch_prompts = temporal_prompts[i:i+batch_size]
    #             with torch.no_grad():
    #                 batch_embeddings = self.llm_encoder(batch_prompts)
    #                 batch_embeddings = self.llm_proj(batch_embeddings)
    #                 temporal_embeddings.append(batch_embeddings.cpu().numpy())
            
    #         # Process spatial prompts
    #         for i in range(0, len(spatial_prompts), batch_size):
    #             batch_prompts = spatial_prompts[i:i+batch_size]
    #             with torch.no_grad():
    #                 batch_embeddings = self.llm_encoder(batch_prompts)
    #                 batch_embeddings = self.llm_proj(batch_embeddings)
    #                 spatial_embeddings.append(batch_embeddings.cpu().numpy())
            
    #         # Concatenate all embeddings
    #         if temporal_embeddings:
    #             temporal_embeddings = np.vstack(temporal_embeddings)
    #         else:
    #             temporal_embeddings = np.zeros((0, self.retrieval_dim))
            
    #         if spatial_embeddings:
    #             spatial_embeddings = np.vstack(spatial_embeddings)
    #         else:
    #             spatial_embeddings = np.zeros((0, self.retrieval_dim))
            
    #         # Update retrieval store
    #         self.retrieval_store.temporal_vectors = [v for v in temporal_embeddings]
    #         self.retrieval_store.spatial_vectors = [v for v in spatial_embeddings]
            
    #         # Update temporal and spatial values using the update_patterns method
    #         self.retrieval_store.update_patterns(history_data)
            
    #         # Rebuild indices
    #         self.retrieval_store._rebuild_indices()
            
    #         # Save documents
    #         self.retrieval_store.save_documents(
    #             self.domain, 
    #             epoch, 
    #             temporal_prompts, 
    #             spatial_prompts
    #         )
            
    #         print(f"Successfully initialized retrieval store at epoch {epoch}")
    #         print(f"Statistics: {self.retrieval_store.get_statistics()}")
            
    #     except Exception as e:
    #         print(f"Error initializing retrieval store: {e}")
    #         traceback.print_exc()
    #         # Create empty store as fallback
    #         self.retrieval_store.temporal_vectors = []
    #         self.retrieval_store.spatial_vectors = []
    #         self.retrieval_store.temporal_values = []
    #         self.retrieval_store.spatial_values = []