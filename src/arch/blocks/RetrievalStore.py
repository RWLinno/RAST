import torch
import numpy as np
import faiss
import json
import os
import glob
import traceback
import logging
from typing import List, Dict, Union, Optional, Tuple
from collections import deque
from sklearn.decomposition import PCA
import concurrent.futures
import hashlib
import time

class RetrievalStore:
    """Vector store for efficient retrieval of temporal and spatial embeddings"""
    
    def __init__(self, dim: int, doc_dir: str = "./database", max_files: int = 5,
                 num_nodes: int = None, seq_len: int = None, use_gpu: bool = True):
        """
        Args:
            dim: Dimension of vectors to store
            doc_dir: Directory for document storage
            max_files: Maximum number of files to retain per dimension
            num_nodes: Number of spatial nodes
            seq_len: Input sequence length
            use_gpu: Whether to use GPU for FAISS operations
        """
        self.dim = dim
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.use_gpu = use_gpu
        
        # GPU device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.gpu_id = 0 if torch.cuda.is_available() and use_gpu else -1
        
        # Initialize FAISS indices with GPU support
        self._init_faiss_indices()
        
        # Separate storage for temporal and spatial data
        self.temporal_vectors = []
        self.temporal_values = []
        self.spatial_vectors = []
        self.spatial_values = []
        
        # File management settings
        self.doc_dir = doc_dir
        self.max_files = max_files
        os.makedirs(doc_dir, exist_ok=True)
        
        # Status flags and logging
        self.warning_shown = False
        self.logger = self._setup_logger()
        
        # Enhanced caching with LRU mechanism
        self.cache_size = 1000  # Increased cache size
        self.temporal_cache = {}
        self.spatial_cache = {}
        self.cache_access_order = {}  # Track access order for LRU
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        
        # Runtime statistics
        self.stats = {
            "update_count": 0,
            "retrieval_count": 0,
            "avg_retrieval_time": 0,
            "cache_efficiency": 0
        }
        
        # Vector compression settings
        self.use_compression = False  # Disable compression for speed
        self.pca_dim = min(64, dim)
        self.pca_temporal = None
        self.pca_spatial = None
        
        # Pre-allocate memory for efficiency
        self.max_batch_size = 2048
        self._preallocate_memory()
        
        # Performance optimization flags
        self.skip_normalization = True  # Skip L2 normalization for speed
        self.batch_search_threshold = 512  # Threshold for batch search
        
    def _init_faiss_indices(self):
        """Initialize FAISS indices with GPU support if available"""
        try:
            if self.use_gpu and torch.cuda.is_available():
                # Initialize GPU resources
                self.faiss_res = faiss.StandardGpuResources()
                
                # Create GPU-accelerated indices
                self.index_temporal = faiss.GpuIndexFlatL2(self.faiss_res, self.dim)
                self.index_spatial = faiss.GpuIndexFlatL2(self.faiss_res, self.dim)
                
                print(f"✅ FAISS GPU indices initialized on device {self.gpu_id}")
            else:
                # Fallback to CPU indices
                self.index_temporal = faiss.IndexFlatL2(self.dim)
                self.index_spatial = faiss.IndexFlatL2(self.dim)
                print("⚠️ Using CPU FAISS indices")
                
        except Exception as e:
            print(f"Error initializing GPU FAISS: {e}, falling back to CPU")
            self.index_temporal = faiss.IndexFlatL2(self.dim)
            self.index_spatial = faiss.IndexFlatL2(self.dim)
            self.use_gpu = False
    
    def _preallocate_memory(self):
        """Pre-allocate memory for common operations"""
        # Pre-allocate arrays for search results
        self.preallocated_distances = np.zeros((self.max_batch_size, 10), dtype=np.float32)
        self.preallocated_indices = np.zeros((self.max_batch_size, 10), dtype=np.int64)
        
    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for the RetrievalStore"""
        logger = logging.getLogger('RetrievalStore')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Create a file handler
            if not os.path.exists(self.doc_dir):
                os.makedirs(self.doc_dir)
            
            log_file = os.path.join(self.doc_dir, 'retrieval_store.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # Create a formatter and add it to the handlers
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger

    def _generate_cache_key(self, query_vectors: np.ndarray, k: int, temporal: bool) -> str:
        """Generate efficient cache key using hash"""
        # Use shape and hash of first few elements for speed
        shape_str = str(query_vectors.shape)
        sample_hash = hashlib.md5(query_vectors[:min(10, len(query_vectors))].tobytes()).hexdigest()[:8]
        return f"{shape_str}_{sample_hash}_{k}_{temporal}"

    def save_documents(self, domain: str, epoch: int = None, temporal_prompts: List[str] = None, spatial_prompts: List[str] = None):
        """Save temporal and spatial documents
        
        Args:
            domain: Domain identifier
            epoch: Current epoch number
            temporal_prompts: List of temporal prompts
            spatial_prompts: List of spatial prompts
        """
        if not self.temporal_vectors or not self.spatial_vectors:
            self.logger.warning("No data to save")
            return
        
        try:
            # Convert vectors to numpy arrays
            temporal_vectors_np = np.stack([v for v in self.temporal_vectors]) if self.temporal_vectors else np.zeros((0, self.dim))
            spatial_vectors_np = np.stack([v for v in self.spatial_vectors]) if self.spatial_vectors else np.zeros((0, self.dim))
            
            # Compress vectors if enabled
            if self.use_compression:
                temporal_vectors_np = self._compress_vectors(temporal_vectors_np, is_temporal=True)
                spatial_vectors_np = self._compress_vectors(spatial_vectors_np, is_temporal=False)
            
            # 将向量转换为numpy数组以便更高效地存储
            temporal_vectors_np = np.stack([
                v if isinstance(v, np.ndarray) else np.array(v, dtype=np.float32)
                for v in temporal_vectors_np
            ]) if temporal_vectors_np.size > 0 else np.zeros((0, self.dim), dtype=np.float32)
            
            spatial_vectors_np = np.stack([
                v if isinstance(v, np.ndarray) else np.array(v, dtype=np.float32)
                for v in spatial_vectors_np
            ]) if spatial_vectors_np.size > 0 else np.zeros((0, self.dim), dtype=np.float32)
            
            # 将值序列化为JSON，便于读取
            temporal_values_json = json.dumps(self.temporal_values)
            spatial_values_json = json.dumps(self.spatial_values)
            
            # 使用npz格式存储多个数组和元数据
            store_file = os.path.join(self.doc_dir, f"{domain}_store_epoch_{epoch}.npz")
            np.savez_compressed(
                store_file,
                temporal_vectors=temporal_vectors_np,
                spatial_vectors=spatial_vectors_np,
                temporal_values=temporal_values_json,
                spatial_values=spatial_values_json,
                epoch=epoch,
                domain=domain
            )
            
            # 保存示例prompts（如果提供）
            if temporal_prompts and spatial_prompts:
                self.save_example_prompts(domain, temporal_prompts[:3], spatial_prompts[:3])
            
            # 清理旧文件，保持最新的max_files个文件
            self._cleanup_old_files(domain)
            
            self.logger.info(f"Successfully saved retrieval store data at epoch {epoch}")
            
        except Exception as e:
            self.logger.error(f"Error saving documents: {e}")
            traceback.print_exc()
    
    def _cleanup_old_files(self, domain: str):
        """清理旧的存储文件，保留最新的max_files个文件"""
        try:
            # 查找所有相关文件
            pattern = os.path.join(self.doc_dir, f"{domain}_store_epoch_*.npz")
            files = glob.glob(pattern)
            
            # 按照epoch排序
            files.sort(key=lambda x: int(x.split('_epoch_')[1].split('.')[0]), reverse=True)
            
            # 删除多余的文件
            for file_to_delete in files[self.max_files:]:
                try:
                    os.remove(file_to_delete)
                    self.logger.info(f"Deleted old store file: {file_to_delete}")
                except OSError:
                    self.logger.warning(f"Failed to delete old file: {file_to_delete}")
                    
        except Exception as e:
            self.logger.error(f"Error cleaning up old files: {e}")
    
    def load_documents(self, domain: str) -> bool:
        """Load documents from file with dimension checking"""
        try:
            # Find the latest document file
            files = glob.glob(os.path.join(self.doc_dir, f"{domain}_*.json"))
            if not files:
                return False
            
            # Sort by modification time (newest first)
            files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            latest_file = files[0]
            
            # Load the document
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            # Extract temporal and spatial prompts
            temporal_prompts = data.get('temporal_prompts', [])
            spatial_prompts = data.get('spatial_prompts', [])
            
            # Process prompts to get embeddings
            # This would typically use an LLM encoder, but for simplicity,
            # we'll just use the stored vectors if available
            
            # Check if vectors are stored in the file
            temporal_vectors = data.get('temporal_vectors', [])
            spatial_vectors = data.get('spatial_vectors', [])
            
            # Check dimensions if vectors exist
            if temporal_vectors and len(temporal_vectors) > 0:
                vector_dim = len(temporal_vectors[0])
                if vector_dim != self.dim:
                    print(f"Warning: Loaded vectors have dimension {vector_dim}, but expected {self.dim}")
                    print("Cannot use loaded vectors due to dimension mismatch")
                    return False
            
            # Update store
            if temporal_vectors and spatial_vectors:
                self.temporal_vectors = temporal_vectors
                self.spatial_vectors = spatial_vectors
                self._rebuild_indices()
                print(f"Successfully loaded {len(temporal_vectors)} temporal and {len(spatial_vectors)} spatial vectors")
                return True
            
            return False
        except Exception as e:
            print(f"Error loading documents: {e}")
            return False
    
    def _rebuild_indices(self):
        """Rebuild both temporal and spatial FAISS indices with GPU optimization"""
        try:
            # Re-initialize indices
            self._init_faiss_indices()
            
            # Add temporal vectors
            if self.temporal_vectors:
                temporal_array = np.stack([v for v in self.temporal_vectors])
                temporal_array = np.ascontiguousarray(temporal_array, dtype=np.float32)
                if temporal_array.size > 0:
                    self.index_temporal.add(temporal_array)
            
            # Add spatial vectors
            if self.spatial_vectors:
                spatial_array = np.stack([v for v in self.spatial_vectors])
                spatial_array = np.ascontiguousarray(spatial_array, dtype=np.float32)
                if spatial_array.size > 0:
                    self.index_spatial.add(spatial_array)
                    
            print(f"✅ Rebuilt indices: {len(self.temporal_vectors)} temporal, {len(self.spatial_vectors)} spatial vectors")
                
        except Exception as e:
            self.logger.error(f"Error rebuilding indices: {e}")
            traceback.print_exc()

    def update_patterns(self, data: torch.Tensor) -> Tuple[List[Dict], List[Dict]]:
        """从输入数据更新时间和空间模式，确保与批量大小无关
        
        Args:
            data: Input tensor of shape [B, T, N, D] or [T, N, D]
            
        Returns:
            Tuple of (temporal_values, spatial_values) containing metadata
        """
        # 确保数据是 4D 张量 [B, T, N, D] 或 3D 张量 [T, N, D]
        if data.dim() == 3:  # [T, N, D]
            # 添加批量维度
            data = data.unsqueeze(0)  # [1, T, N, D]
        
        B, T, N, D = data.shape
        device = data.device
        
        # 将数据在计算前移到CPU
        data = data.detach().cpu()
        
        # 不管batch_size如何，首先对批次维度进行平均，消除batch影响
        data_mean = torch.mean(data, dim=0, keepdim=True)  # [1, T, N, D]
        
        # 计算temporal patterns
        time_slice = data_mean.squeeze(0)  # [T, N, D] - 移除批量维度
        
        # 计算全局统计数据，用于提供上下文
        global_mean = torch.mean(time_slice)
        global_std = torch.std(time_slice)
        
        # 计算每个时间步的统计数据（跨节点）
        temporal_mean = torch.mean(time_slice, dim=1)   # [T, D] - 跨节点平均
        temporal_std = torch.std(time_slice, dim=1)     # [T, D]
        temporal_max, _ = torch.max(time_slice, dim=1)  # [T, D]
        temporal_min, _ = torch.min(time_slice, dim=1)  # [T, D]
        
        # 构建每个时间步的元数据
        temporal_values = []
        for t in range(T):
            # 计算每个时间步的相对重要性
            importance = torch.norm(temporal_std[t]) / torch.norm(global_std) if global_std > 0 else 1.0
            
            temporal_values.append({
                "mean": temporal_mean[t].numpy().tolist(),
                "std": temporal_std[t].numpy().tolist(),
                "max": temporal_max[t].numpy().tolist(),
                "min": temporal_min[t].numpy().tolist(),
                "timestamp": t,
                "importance": float(importance)
            })
        
        # 计算spatial patterns
        node_data = time_slice.transpose(0, 1)  # [N, T, D]
        
        # 计算每个节点的统计数据（跨时间）
        spatial_mean = torch.mean(node_data, dim=1)     # [N, D] - 跨时间平均
        spatial_std = torch.std(node_data, dim=1)       # [N, D]
        spatial_max, _ = torch.max(node_data, dim=1)    # [N, D]
        spatial_min, _ = torch.min(node_data, dim=1)    # [N, D]
        
        # 构建每个节点的元数据
        spatial_values = []
        for n in range(N):
            # 计算每个节点的相对重要性
            importance = torch.norm(spatial_std[n]) / torch.norm(global_std) if global_std > 0 else 1.0
            
            spatial_values.append({
                "mean": spatial_mean[n].numpy().tolist(),
                "std": spatial_std[n].numpy().tolist(),
                "max": spatial_max[n].numpy().tolist(),
                "min": spatial_min[n].numpy().tolist(),
                "node_id": n,
                "importance": float(importance)
            })
        
        # 更新存储的值
        self.temporal_values = temporal_values
        self.spatial_values = spatial_values
        
        # 更新统计信息
        self.stats["update_count"] += 1
        
        return temporal_values, spatial_values

    def optimize_indices(self):
        """
        Optimize the FAISS indices based on vector count
        - For small datasets: Use IndexFlatIP (exact search)
        - For larger datasets: Use IndexHNSWFlat (approximate search)
        """
        try:
            import logging
            logger = logging.getLogger("RetrievalStore")
            
            # Convert vector lists to numpy arrays
            if len(self.temporal_vectors) > 0:
                # Ensure vectors are numpy arrays of float32 type and make them contiguous
                temporal_array = np.array([v for v in self.temporal_vectors], dtype=np.float32)
                temporal_array = np.ascontiguousarray(temporal_array)
                
                # Normalize vectors - only if they're not empty
                if temporal_array.size > 0:
                    faiss.normalize_L2(temporal_array)
                
                # Choose appropriate index type based on vector count
                if len(self.temporal_vectors) < 5000:
                    self.temporal_index = faiss.IndexFlatIP(self.dim)
                else:
                    # HNSW index for larger datasets
                    self.temporal_index = faiss.IndexHNSWFlat(self.dim, 32)  # 32 neighbors per layer
                    self.temporal_index.hnsw.efConstruction = 40  # More thorough construction
                    self.temporal_index.hnsw.efSearch = 16  # Faster search
                
                # Add vectors to index
                if temporal_array.size > 0:
                    self.temporal_index.add(temporal_array)
                logger.info(f"Optimized temporal index with {len(self.temporal_vectors)} vectors")
            
            # Same for spatial vectors
            if len(self.spatial_vectors) > 0:
                # Ensure vectors are numpy arrays of float32 type and make them contiguous
                spatial_array = np.array([v for v in self.spatial_vectors], dtype=np.float32)
                spatial_array = np.ascontiguousarray(spatial_array)
                
                # Normalize vectors - only if they're not empty
                if spatial_array.size > 0:
                    faiss.normalize_L2(spatial_array)
                
                # Choose appropriate index type based on vector count
                if len(self.spatial_vectors) < 5000:
                    self.spatial_index = faiss.IndexFlatIP(self.dim)
                else:
                    # HNSW index for larger datasets
                    self.spatial_index = faiss.IndexHNSWFlat(self.dim, 32)
                    self.spatial_index.hnsw.efConstruction = 40
                    self.spatial_index.hnsw.efSearch = 16
                
                # Add vectors to index
                if spatial_array.size > 0:
                    self.spatial_index.add(spatial_array)
                logger.info(f"Optimized spatial index with {len(self.spatial_vectors)} vectors")
            
            logger.info("Successfully optimized indices")
            return True
        
        except Exception as e:
            import traceback
            logger = logging.getLogger("RetrievalStore")
            logger.error(f"Error optimizing indices: {e}")
            traceback.print_exc()
            
            # Fallback to simple indices
            self.temporal_index = faiss.IndexFlatIP(self.dim)
            self.spatial_index = faiss.IndexFlatIP(self.dim)
            
            # Add vectors if available
            if len(self.temporal_vectors) > 0:
                try:
                    temporal_array = np.array([v for v in self.temporal_vectors], dtype=np.float32)
                    self.temporal_index.add(temporal_array)
                except Exception as e2:
                    logger.error(f"Fallback error for temporal index: {e2}")
            
            if len(self.spatial_vectors) > 0:
                try:
                    spatial_array = np.array([v for v in self.spatial_vectors], dtype=np.float32)
                    self.spatial_index.add(spatial_array)
                except Exception as e2:
                    logger.error(f"Fallback error for spatial index: {e2}")
            
            return False

    def search(self, query_vectors: np.ndarray, k: int = 5, temporal: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized search for nearest k vectors with aggressive caching and GPU acceleration"""
        start_time = time.time()
        self.stats["retrieval_count"] += 1
        
        # 1. Generate cache key
        cache_key = self._generate_cache_key(query_vectors, k, temporal)
        cache = self.temporal_cache if temporal else self.spatial_cache
        
        # 2. Check cache with LRU tracking
        if cache_key in cache:
            self.cache_hit_count += 1
            self.cache_access_order[cache_key] = time.time()
            result = cache[cache_key]
            return result
        
        self.cache_miss_count += 1
        
        # 3. Get index and check if empty
        index = self.index_temporal if temporal else self.index_spatial
        vector_count = len(self.temporal_vectors) if temporal else len(self.spatial_vectors)
        
        if vector_count == 0:
            empty_result = (
                np.zeros((query_vectors.shape[0], k), dtype=np.float32),
                np.zeros((query_vectors.shape[0], k), dtype=np.int64)
            )
            return empty_result
        
        # 4. Adjust k to available vectors
        actual_k = min(k, vector_count)
        
        try:
            # 5. Optimized search execution
            if query_vectors.shape[0] > self.batch_search_threshold:
                # For large batches, process in chunks
                chunk_size = self.batch_search_threshold
                all_distances = []
                all_indices = []
                
                for i in range(0, query_vectors.shape[0], chunk_size):
                    end_idx = min(i + chunk_size, query_vectors.shape[0])
                    chunk_query = np.ascontiguousarray(query_vectors[i:end_idx], dtype=np.float32)
                    
                    # Direct search without normalization
                    chunk_distances, chunk_indices = index.search(chunk_query, actual_k)
                    all_distances.append(chunk_distances)
                    all_indices.append(chunk_indices)
                
                distances = np.vstack(all_distances)
                indices = np.vstack(all_indices)
            else:
                # Direct search for smaller batches
                query_contiguous = np.ascontiguousarray(query_vectors, dtype=np.float32)
                distances, indices = index.search(query_contiguous, actual_k)
            
            # 6. Cache result with LRU management
            result = (distances, indices)
            cache[cache_key] = result
            self.cache_access_order[cache_key] = time.time()
            
            # 7. Clean cache if too large
            if len(cache) > self.cache_size:
                self._clean_cache_lru(cache)
            
            # 8. Update timing statistics
            elapsed_time = time.time() - start_time
            self.stats["avg_retrieval_time"] = (
                self.stats["avg_retrieval_time"] * (self.stats["retrieval_count"] - 1) + elapsed_time
            ) / self.stats["retrieval_count"]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during search: {e}")
            # Return zero arrays as fallback
            empty_result = (
                np.zeros((query_vectors.shape[0], k), dtype=np.float32),
                np.zeros((query_vectors.shape[0], k), dtype=np.int64)
            )
            return empty_result

    def save_example_prompts(self, domain: str, temporal_prompts: List[str], spatial_prompts: List[str]):
        """Save example prompts for user inspection
        
        Args:
            domain: Domain identifier
            temporal_prompts: List of temporal prompts
            spatial_prompts: List of spatial prompts
        """
        try:
            # Create example prompts file
            example_file = os.path.join(self.doc_dir, f"{domain}_example_prompts.txt")
            
            with open(example_file, "w", encoding="utf-8") as f:
                # Write temporal prompts
                f.write("="*80 + "\n")
                f.write("TEMPORAL PROMPTS EXAMPLES\n")
                f.write("="*80 + "\n\n")
                
                # Save first 3 temporal prompts as examples
                for i, prompt in enumerate(temporal_prompts[:3]):
                    f.write(f"Temporal Prompt {i+1}:\n")
                    f.write("-"*40 + "\n")
                    f.write(prompt)
                    f.write("\n\n")
                
                # Write spatial prompts
                f.write("="*80 + "\n")
                f.write("SPATIAL PROMPTS EXAMPLES\n")
                f.write("="*80 + "\n\n")
                
                # Save first 3 spatial prompts as examples
                for i, prompt in enumerate(spatial_prompts[:3]):
                    f.write(f"Spatial Prompt {i+1}:\n")
                    f.write("-"*40 + "\n")
                    f.write(prompt)
                    f.write("\n\n")
            
            self.logger.info(f"Saved example prompts to {example_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving example prompts: {e}")
            traceback.print_exc()
            
    def get_statistics(self) -> Dict:
        """获取检索存储的统计信息"""
        cache_efficiency = self.cache_hit_count / (self.cache_hit_count + self.cache_miss_count) if (self.cache_hit_count + self.cache_miss_count) > 0 else 0
        
        stats = {
            "temporal_vectors": len(self.temporal_vectors),
            "spatial_vectors": len(self.spatial_vectors),
            "cache_hits": self.cache_hit_count,
            "cache_misses": self.cache_miss_count,
            "cache_hit_ratio": cache_efficiency,
            "update_count": self.stats["update_count"],
            "retrieval_count": self.stats["retrieval_count"],
            "avg_retrieval_time_ms": self.stats["avg_retrieval_time"] * 1000,
            "gpu_enabled": self.use_gpu,
            "cache_size": len(self.temporal_cache) + len(self.spatial_cache)
        }
        return stats

    def _compress_vectors(self, vectors: np.ndarray, is_temporal: bool = True):
        """Compress vectors using PCA to reduce memory footprint"""
        if not self.use_compression or vectors.shape[0] < 100:
            return vectors
        
        try:
            # Initialize or update PCA
            pca = self.pca_temporal if is_temporal else self.pca_spatial
            if pca is None or pca.n_components > vectors.shape[0] // 2:
                n_components = min(self.pca_dim, vectors.shape[0] // 2)
                pca = PCA(n_components=n_components)
                pca.fit(vectors)
                
                if is_temporal:
                    self.pca_temporal = pca
                else:
                    self.pca_spatial = pca
            
            # Transform vectors
            compressed = pca.transform(vectors)
            self.logger.info(f"Compressed vectors from {vectors.shape} to {compressed.shape}")
            return compressed
            
        except Exception as e:
            self.logger.error(f"Error compressing vectors: {e}")
            return vectors

    def preload_documents(self, domain: str):
        """Preload documents for a domain in background thread"""
        if domain in self.preload_domains:
            return
        
        self.preload_domains.add(domain)
        
        def _preload():
            try:
                pattern = os.path.join(self.doc_dir, f"{domain}_store_epoch_*.npz")
                files = glob.glob(pattern)
                
                if not files:
                    return
                
                # Sort by epoch, get latest
                files.sort(key=lambda x: int(x.split('_epoch_')[1].split('.')[0]), reverse=True)
                latest_file = files[0]
                
                # Just load file into memory but don't process yet
                with np.load(latest_file, allow_pickle=True) as data:
                    self.preloaded_data = {
                        'temporal_vectors': data['temporal_vectors'],
                        'spatial_vectors': data['spatial_vectors'],
                        'temporal_values': data['temporal_values'],
                        'spatial_values': data['spatial_values']
                    }
                
                self.logger.info(f"Preloaded data for domain {domain}")
            except Exception as e:
                self.logger.error(f"Error preloading documents: {e}")
        
        # Start preloading in background
        self.preload_future = self.executor.submit(_preload)

    def incremental_update(self, new_temporal_vectors, new_spatial_vectors, domain: str = None, epoch: int = None):
        """Optimized incremental update with GPU acceleration"""
        try:
            max_vectors = 5000  # Reduced for better performance
            
            # Add temporal vectors
            if len(new_temporal_vectors) > 0:
                # Convert to contiguous array for efficiency
                new_temp_array = np.ascontiguousarray(new_temporal_vectors, dtype=np.float32)
                
                # Add to index directly
                if hasattr(self, 'index_temporal') and self.index_temporal is not None:
                    self.index_temporal.add(new_temp_array)
                
                # Update storage with size limit
                self.temporal_vectors.extend([v for v in new_temporal_vectors])
                if len(self.temporal_vectors) > max_vectors:
                    # Keep most recent vectors
                    excess = len(self.temporal_vectors) - max_vectors
                    self.temporal_vectors = self.temporal_vectors[excess:]
                    # Rebuild index if we removed vectors
                    self._rebuild_indices()
            
            # Add spatial vectors
            if len(new_spatial_vectors) > 0:
                # Convert to contiguous array for efficiency
                new_spat_array = np.ascontiguousarray(new_spatial_vectors, dtype=np.float32)
                
                # Add to index directly
                if hasattr(self, 'index_spatial') and self.index_spatial is not None:
                    self.index_spatial.add(new_spat_array)
                
                # Update storage with size limit
                self.spatial_vectors.extend([v for v in new_spatial_vectors])
                if len(self.spatial_vectors) > max_vectors:
                    # Keep most recent vectors
                    excess = len(self.spatial_vectors) - max_vectors
                    self.spatial_vectors = self.spatial_vectors[excess:]
                    # Rebuild index if we removed vectors
                    self._rebuild_indices()
            
            # Clear caches after update
            self.temporal_cache.clear()
            self.spatial_cache.clear()
            self.cache_access_order.clear()
            
            self.stats["update_count"] += 1
            
        except Exception as e:
            self.logger.error(f"Error during incremental update: {e}")
            # Fallback to full rebuild
            self._rebuild_indices()

    def _clean_cache_lru(self, cache: dict):
        """Clean cache using LRU strategy"""
        if len(cache) <= self.cache_size:
            return
            
        # Sort by access time and remove oldest entries
        sorted_keys = sorted(self.cache_access_order.items(), key=lambda x: x[1])
        keys_to_remove = [key for key, _ in sorted_keys[:len(cache) - self.cache_size]]
        
        for key in keys_to_remove:
            cache.pop(key, None)
            self.cache_access_order.pop(key, None)

    def use_memory_mapping(self):
        """Use memory mapping to reduce memory usage for large vector sets
        
        Memory mapping stores vectors on disk and loads them into memory only when needed,
        which significantly reduces RAM usage for large datasets while maintaining fast access.
        """
        try:
            # Set threshold for when to use memory mapping
            threshold = 5000
            
            # Process temporal vectors
            if len(self.temporal_vectors) > threshold:
                # Create directory for memory mapped files
                mmap_dir = os.path.join(self.doc_dir, "mmap_files")
                os.makedirs(mmap_dir, exist_ok=True)
                
                # Create memory mapped file for temporal vectors
                temp_file = os.path.join(mmap_dir, f"temporal_vectors_{id(self)}.dat")
                
                # Stack vectors into a single array
                temporal_array = np.stack([v for v in self.temporal_vectors])
                
                # Create memory mapping in write mode
                fp = np.memmap(temp_file, dtype='float32', mode='w+', 
                              shape=temporal_array.shape)
                # Write data to disk
                fp[:] = temporal_array[:]
                fp.flush()
                
                # Replace original vector list with memory mapping in read mode
                self.temporal_vectors_mmap = np.memmap(temp_file, dtype='float32', 
                                                     mode='r', shape=temporal_array.shape)
                self.temporal_mmap_file = temp_file
                
                # Store original vector length for reference
                self.temporal_vectors_length = len(self.temporal_vectors)
                
                # Clear original vectors to free memory
                self.temporal_vectors = None
                
                self.logger.info(f"Using memory mapping for {self.temporal_vectors_length} temporal vectors")
            
            # Process spatial vectors
            if len(self.spatial_vectors) > threshold:
                # Create directory for memory mapped files (if not already created)
                mmap_dir = os.path.join(self.doc_dir, "mmap_files")
                os.makedirs(mmap_dir, exist_ok=True)
                
                # Create memory mapped file for spatial vectors
                temp_file = os.path.join(mmap_dir, f"spatial_vectors_{id(self)}.dat")
                
                # Stack vectors into a single array
                spatial_array = np.stack([v for v in self.spatial_vectors])
                
                # Create memory mapping in write mode
                fp = np.memmap(temp_file, dtype='float32', mode='w+', 
                              shape=spatial_array.shape)
                # Write data to disk
                fp[:] = spatial_array[:]
                fp.flush()
                
                # Replace original vector list with memory mapping in read mode
                self.spatial_vectors_mmap = np.memmap(temp_file, dtype='float32', 
                                                    mode='r', shape=spatial_array.shape)
                self.spatial_mmap_file = temp_file
                
                # Store original vector length for reference
                self.spatial_vectors_length = len(self.spatial_vectors)
                
                # Clear original vectors to free memory
                self.spatial_vectors = None
                
                self.logger.info(f"Using memory mapping for {self.spatial_vectors_length} spatial vectors")
            
            # Set flag indicating memory mapping is in use
            self.using_mmap = True
            
            # Update indices to use memory mapped vectors
            self._update_indices_with_mmap()
            
        except Exception as e:
            self.logger.error(f"Failed to use memory mapping: {e}")
            traceback.print_exc()
            # Ensure any partially created files are cleaned up
            self._cleanup_mmap_files()

    def _update_indices_with_mmap(self):
        """Update FAISS indices to use memory mapped vectors"""
        try:
            # Reset existing indices
            self.index_temporal = None
            self.index_spatial = None
            
            # Create index for temporal vectors
            if hasattr(self, 'temporal_vectors_mmap'):
                # Create appropriate index
                self.index_temporal = faiss.IndexFlatL2(self.dim)
                # Add memory mapped vectors to index
                self.index_temporal.add(self.temporal_vectors_mmap)
                self.logger.info(f"Updated temporal index with memory mapped vectors")
            
            # Create index for spatial vectors
            if hasattr(self, 'spatial_vectors_mmap'):
                # Create appropriate index
                self.index_spatial = faiss.IndexFlatL2(self.dim)
                # Add memory mapped vectors to index
                self.index_spatial.add(self.spatial_vectors_mmap)
                self.logger.info(f"Updated spatial index with memory mapped vectors")
            
        except Exception as e:
            self.logger.error(f"Failed to update indices with memory mapped vectors: {e}")
            traceback.print_exc()
            # Fall back to standard indices
            self._rebuild_indices()

    def _cleanup_mmap_files(self):
        """Clean up memory mapped files to prevent disk space leaks"""
        try:
            # Clean up temporal vector mapping file
            if hasattr(self, 'temporal_mmap_file') and os.path.exists(self.temporal_mmap_file):
                # First remove reference to mapping object
                if hasattr(self, 'temporal_vectors_mmap'):
                    del self.temporal_vectors_mmap
                # Then delete the file
                os.remove(self.temporal_mmap_file)
                self.logger.info(f"Deleted temporal vector mapping file: {self.temporal_mmap_file}")
            
            # Clean up spatial vector mapping file
            if hasattr(self, 'spatial_mmap_file') and os.path.exists(self.spatial_mmap_file):
                # First remove reference to mapping object
                if hasattr(self, 'spatial_vectors_mmap'):
                    del self.spatial_vectors_mmap
                # Then delete the file
                os.remove(self.spatial_mmap_file)
                self.logger.info(f"Deleted spatial vector mapping file: {self.spatial_mmap_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to clean up memory mapping files: {e}")
            traceback.print_exc()

    def __del__(self):
        """Cleanup GPU resources"""
        try:
            if hasattr(self, 'faiss_res'):
                del self.faiss_res
        except:
            pass
