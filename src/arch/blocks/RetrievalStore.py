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
    def __init__(self, dim: int, doc_dir: str = "./database", max_files: int = 5,
                 num_nodes: int = None, seq_len: int = None, use_gpu: bool = True):
        self.dim = dim
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.use_gpu = use_gpu
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.gpu_id = 0 if torch.cuda.is_available() and use_gpu else -1
        
        self._init_faiss_indices()
        
        self.temporal_vectors = []
        self.temporal_values = []
        self.spatial_vectors = []
        self.spatial_values = []
        
        self.doc_dir = doc_dir
        self.max_files = max_files
        os.makedirs(doc_dir, exist_ok=True)
        
        self.warning_shown = False
        self.logger = self._setup_logger()
        
        self.cache_size = 1000  # Increased cache size
        self.temporal_cache = {}
        self.spatial_cache = {}
        self.cache_access_order = {}
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        
        self.stats = {
            "update_count": 0,
            "retrieval_count": 0,
            "avg_retrieval_time": 0,
            "cache_efficiency": 0
        }
        
        self.use_compression = False  # Disable compression for speed
        self.pca_dim = min(64, dim)
        self.pca_temporal = None
        self.pca_spatial = None
        
        self.max_batch_size = 2048
        self._preallocate_memory()
        
        self.skip_normalization = True
        self.batch_search_threshold = 512
        
    def _init_faiss_indices(self):
        try:
            if self.use_gpu and torch.cuda.is_available():
                self.faiss_res = faiss.StandardGpuResources()
                
                self.index_temporal = faiss.GpuIndexFlatL2(self.faiss_res, self.dim)
                self.index_spatial = faiss.GpuIndexFlatL2(self.faiss_res, self.dim)
                
                print(f"✅ FAISS GPU indices initialized on device {self.gpu_id}")
            else:
                self.index_temporal = faiss.IndexFlatL2(self.dim)
                self.index_spatial = faiss.IndexFlatL2(self.dim)
                print("⚠️ Using CPU FAISS indices")
                
        except Exception as e:
            print(f"Error initializing GPU FAISS: {e}, falling back to CPU")
            self.index_temporal = faiss.IndexFlatL2(self.dim)
            self.index_spatial = faiss.IndexFlatL2(self.dim)
            self.use_gpu = False
    
    def _preallocate_memory(self):
        self.preallocated_distances = np.zeros((self.max_batch_size, 10), dtype=np.float32)
        self.preallocated_indices = np.zeros((self.max_batch_size, 10), dtype=np.int64)
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('RetrievalStore')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            if not os.path.exists(self.doc_dir):
                os.makedirs(self.doc_dir)
            
            log_file = os.path.join(self.doc_dir, 'retrieval_store.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger

    def _generate_cache_key(self, query_vectors: np.ndarray, k: int, temporal: bool) -> str:
        shape_str = str(query_vectors.shape)
        sample_hash = hashlib.md5(query_vectors[:min(10, len(query_vectors))].tobytes()).hexdigest()[:8]
        return f"{shape_str}_{sample_hash}_{k}_{temporal}"

    def save_documents(self, domain: str, epoch: int = None, temporal_prompts: List[str] = None, spatial_prompts: List[str] = None):
        if not self.temporal_vectors or not self.spatial_vectors:
            self.logger.warning("No data to save")
            return
        
        try:
            temporal_vectors_np = np.stack([v for v in self.temporal_vectors]) if self.temporal_vectors else np.zeros((0, self.dim))
            spatial_vectors_np = np.stack([v for v in self.spatial_vectors]) if self.spatial_vectors else np.zeros((0, self.dim))
            
            if self.use_compression:
                temporal_vectors_np = self._compress_vectors(temporal_vectors_np, is_temporal=True)
                spatial_vectors_np = self._compress_vectors(spatial_vectors_np, is_temporal=False)
            
            temporal_vectors_np = np.stack([
                v if isinstance(v, np.ndarray) else np.array(v, dtype=np.float32)
                for v in temporal_vectors_np
            ]) if temporal_vectors_np.size > 0 else np.zeros((0, self.dim), dtype=np.float32)
            
            spatial_vectors_np = np.stack([
                v if isinstance(v, np.ndarray) else np.array(v, dtype=np.float32)
                for v in spatial_vectors_np
            ]) if spatial_vectors_np.size > 0 else np.zeros((0, self.dim), dtype=np.float32)
            
            temporal_values_json = json.dumps(self.temporal_values)
            spatial_values_json = json.dumps(self.spatial_values)
            
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
            
            if temporal_prompts and spatial_prompts:
                self.save_example_prompts(domain, temporal_prompts[:3], spatial_prompts[:3])
            
            self._cleanup_old_files(domain)
            
            self.logger.info(f"Successfully saved retrieval store data at epoch {epoch}")
            
        except Exception as e:
            self.logger.error(f"Error saving documents: {e}")
            traceback.print_exc()
    
    def _cleanup_old_files(self, domain: str):
        try:
            pattern = os.path.join(self.doc_dir, f"{domain}_store_epoch_*.npz")
            files = glob.glob(pattern)
            
            files.sort(key=lambda x: int(x.split('_epoch_')[1].split('.')[0]), reverse=True)
            
            for file_to_delete in files[self.max_files:]:
                try:
                    os.remove(file_to_delete)
                    self.logger.info(f"Deleted old store file: {file_to_delete}")
                except OSError:
                    self.logger.warning(f"Failed to delete old file: {file_to_delete}")
                    
        except Exception as e:
            self.logger.error(f"Error cleaning up old files: {e}")
    
    def load_documents(self, domain: str) -> bool:
        try:
            files = glob.glob(os.path.join(self.doc_dir, f"{domain}_*.json"))
            if not files:
                return False
            
            files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            latest_file = files[0]
            
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            temporal_prompts = data.get('temporal_prompts', [])
            spatial_prompts = data.get('spatial_prompts', [])
            
            temporal_vectors = data.get('temporal_vectors', [])
            spatial_vectors = data.get('spatial_vectors', [])
            
            if temporal_vectors and len(temporal_vectors) > 0:
                vector_dim = len(temporal_vectors[0])
                if vector_dim != self.dim:
                    print(f"Warning: Loaded vectors have dimension {vector_dim}, but expected {self.dim}")
                    print("Cannot use loaded vectors due to dimension mismatch")
                    return False
            
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
        try:
            self._init_faiss_indices()
            
            if self.temporal_vectors:
                try:
                    temporal_array = np.array([v for v in self.temporal_vectors], dtype=np.float32)
                    if temporal_array.size > 0:
                        temporal_array = np.ascontiguousarray(temporal_array)
                        
                        if temporal_array.shape[1] != self.dim:
                            self.logger.warning(f"Temporal vector dimension mismatch: expected {self.dim}, got {temporal_array.shape[1]}")
                        else:
                            self.index_temporal.add(temporal_array)
                            self.logger.info(f"Added {len(self.temporal_vectors)} temporal vectors to index")
                except Exception as e:
                    self.logger.error(f"Error adding temporal vectors to index: {e}")
            
            if self.spatial_vectors:
                try:
                    spatial_array = np.array([v for v in self.spatial_vectors], dtype=np.float32)
                    if spatial_array.size > 0:
                        spatial_array = np.ascontiguousarray(spatial_array)
                        
                        if spatial_array.shape[1] != self.dim:
                            self.logger.warning(f"Spatial vector dimension mismatch: expected {self.dim}, got {spatial_array.shape[1]}")
                        else:
                            self.index_spatial.add(spatial_array)
                            self.logger.info(f"Added {len(self.spatial_vectors)} spatial vectors to index")
                except Exception as e:
                    self.logger.error(f"Error adding spatial vectors to index: {e}")
            
            temporal_count = self.index_temporal.ntotal if hasattr(self.index_temporal, 'ntotal') else 0
            spatial_count = self.index_spatial.ntotal if hasattr(self.index_spatial, 'ntotal') else 0
            
            print(f"✅ Rebuilt indices: {temporal_count} temporal, {spatial_count} spatial vectors")
            self.logger.info(f"Successfully rebuilt indices with {temporal_count} temporal and {spatial_count} spatial vectors")
                
        except Exception as e:
            self.logger.error(f"Error rebuilding indices: {e}")
            import traceback
            traceback.print_exc()
            
            try:
                self._init_faiss_indices()
                self.logger.info("Fallback: Created empty indices")
            except Exception as fallback_error:
                self.logger.error(f"Failed to create fallback indices: {fallback_error}")

    def update_patterns(self, data: torch.Tensor) -> Tuple[List[Dict], List[Dict]]:
        if data.dim() == 3:
            data = data.unsqueeze(0)  # [1, T, N, D]
        
        B, T, N, D = data.shape
        device = data.device
        
        data = data.detach().cpu()
        
        data_mean = torch.mean(data, dim=0, keepdim=True)  # [1, T, N, D]
        
        time_slice = data_mean.squeeze(0)
        
        global_mean = torch.mean(time_slice)
        global_std = torch.std(time_slice)
        
        temporal_mean = torch.mean(time_slice, dim=1)
        temporal_std = torch.std(time_slice, dim=1)
        temporal_max, _ = torch.max(time_slice, dim=1)
        temporal_min, _ = torch.min(time_slice, dim=1)
        
        temporal_values = []
        for t in range(T):
            importance = torch.norm(temporal_std[t]) / torch.norm(global_std) if global_std > 0 else 1.0
            
            temporal_values.append({
                "mean": temporal_mean[t].numpy().tolist(),
                "std": temporal_std[t].numpy().tolist(),
                "max": temporal_max[t].numpy().tolist(),
                "min": temporal_min[t].numpy().tolist(),
                "timestamp": t,
                "importance": float(importance)
            })
        
        node_data = time_slice.transpose(0, 1)
        
        spatial_mean = torch.mean(node_data, dim=1)
        spatial_std = torch.std(node_data, dim=1)
        spatial_max, _ = torch.max(node_data, dim=1)
        spatial_min, _ = torch.min(node_data, dim=1)
        
        spatial_values = []
        for n in range(N):
            importance = torch.norm(spatial_std[n]) / torch.norm(global_std) if global_std > 0 else 1.0
            
            spatial_values.append({
                "mean": spatial_mean[n].numpy().tolist(),
                "std": spatial_std[n].numpy().tolist(),
                "max": spatial_max[n].numpy().tolist(),
                "min": spatial_min[n].numpy().tolist(),
                "node_id": n,
                "importance": float(importance)
            })
        
        self.temporal_values = temporal_values
        self.spatial_values = spatial_values
        self.stats["update_count"] += 1
        
        return temporal_values, spatial_values

    def optimize_indices(self):
        try:
            import logging
            logger = logging.getLogger("RetrievalStore")
            
            if len(self.temporal_vectors) > 0:
                temporal_array = np.array([v for v in self.temporal_vectors], dtype=np.float32)
                temporal_array = np.ascontiguousarray(temporal_array)
                
                if temporal_array.size > 0:
                    faiss.normalize_L2(temporal_array)
                
                if len(self.temporal_vectors) < 5000:
                    self.temporal_index = faiss.IndexFlatIP(self.dim)
                else:
                    self.temporal_index = faiss.IndexHNSWFlat(self.dim, 32)  # 32 neighbors per layer
                    self.temporal_index.hnsw.efConstruction = 40  # More thorough construction
                    self.temporal_index.hnsw.efSearch = 16  # Faster search
                
                if temporal_array.size > 0:
                    self.temporal_index.add(temporal_array)
                logger.info(f"Optimized temporal index with {len(self.temporal_vectors)} vectors")
            
            if len(self.spatial_vectors) > 0:
                spatial_array = np.array([v for v in self.spatial_vectors], dtype=np.float32)
                spatial_array = np.ascontiguousarray(spatial_array)
                
                if spatial_array.size > 0:
                    faiss.normalize_L2(spatial_array)
                
                if len(self.spatial_vectors) < 5000:
                    self.spatial_index = faiss.IndexFlatIP(self.dim)
                else:
                    self.spatial_index = faiss.IndexHNSWFlat(self.dim, 32)
                    self.spatial_index.hnsw.efConstruction = 40
                    self.spatial_index.hnsw.efSearch = 16
                
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
            
            self.temporal_index = faiss.IndexFlatIP(self.dim)
            self.spatial_index = faiss.IndexFlatIP(self.dim)
            
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
        start_time = time.time()
        self.stats["retrieval_count"] += 1
        
        cache_key = self._generate_cache_key(query_vectors, k, temporal)
        cache = self.temporal_cache if temporal else self.spatial_cache
        
        if cache_key in cache:
            self.cache_hit_count += 1
            self.cache_access_order[cache_key] = time.time()
            result = cache[cache_key]
            return result
        
        self.cache_miss_count += 1
        
        index = self.index_temporal if temporal else self.index_spatial
        vector_count = len(self.temporal_vectors) if temporal else len(self.spatial_vectors)
        
        if vector_count == 0:
            empty_result = (
                np.zeros((query_vectors.shape[0], k), dtype=np.float32),
                np.zeros((query_vectors.shape[0], k), dtype=np.int64)
            )
            return empty_result
        
        try:
            if not query_vectors.flags['C_CONTIGUOUS']:
                query_vectors = np.ascontiguousarray(query_vectors)
            
            if query_vectors.dtype != np.float32:
                query_vectors = query_vectors.astype(np.float32)
            
            actual_k = min(k, vector_count)
            
            if hasattr(index, 'ntotal') and index.ntotal > 0:
                distances, indices = index.search(query_vectors, actual_k)
                
                if distances.shape[1] < k:
                    pad_width = k - distances.shape[1]
                    distances = np.pad(distances, ((0, 0), (0, pad_width)), mode='constant', constant_values=float('inf'))
                    indices = np.pad(indices, ((0, 0), (0, pad_width)), mode='constant', constant_values=-1)
                
                result = (distances, indices)
                cache[cache_key] = result
                self.cache_access_order[cache_key] = time.time()
                
                if len(cache) > self.cache_size:
                    self._clean_cache_lru(cache)
                
                search_time = time.time() - start_time
                self.stats["avg_retrieval_time"] = (self.stats["avg_retrieval_time"] * (self.stats["retrieval_count"] - 1) + search_time) / self.stats["retrieval_count"]
                self.stats["cache_efficiency"] = self.cache_hit_count / (self.cache_hit_count + self.cache_miss_count) * 100
                
                return result
            else:
                self.logger.warning(f"Index for {'temporal' if temporal else 'spatial'} search is empty")
                return (
                    np.zeros((query_vectors.shape[0], k), dtype=np.float32),
                    np.zeros((query_vectors.shape[0], k), dtype=np.int64)
                )
                
        except Exception as e:
            self.logger.error(f"Error during {'temporal' if temporal else 'spatial'} search: {e}")
            import traceback
            traceback.print_exc()
            
            return (
                np.zeros((query_vectors.shape[0], k), dtype=np.float32),
                np.zeros((query_vectors.shape[0], k), dtype=np.int64)
            )

    def save_example_prompts(self, domain: str, temporal_prompts: List[str], spatial_prompts: List[str]):
        try:
            example_file = os.path.join(self.doc_dir, f"{domain}_example_prompts.txt")
            
            with open(example_file, "w", encoding="utf-8") as f:
                f.write("="*80 + "\n")
                f.write("TEMPORAL PROMPTS EXAMPLES\n")
                f.write("="*80 + "\n\n")
                
                for i, prompt in enumerate(temporal_prompts[:3]):
                    f.write(f"Temporal Prompt {i+1}:\n")
                    f.write("-"*40 + "\n")
                    f.write(prompt)
                    f.write("\n\n")
                
                f.write("="*80 + "\n")
                f.write("SPATIAL PROMPTS EXAMPLES\n")
                f.write("="*80 + "\n\n")
                
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
        if not self.use_compression or vectors.shape[0] < 100:
            return vectors
        
        try:
            pca = self.pca_temporal if is_temporal else self.pca_spatial
            if pca is None or pca.n_components > vectors.shape[0] // 2:
                n_components = min(self.pca_dim, vectors.shape[0] // 2)
                pca = PCA(n_components=n_components)
                pca.fit(vectors)
                
                if is_temporal:
                    self.pca_temporal = pca
                else:
                    self.pca_spatial = pca
            
            compressed = pca.transform(vectors)
            self.logger.info(f"Compressed vectors from {vectors.shape} to {compressed.shape}")
            return compressed
            
        except Exception as e:
            self.logger.error(f"Error compressing vectors: {e}")
            return vectors

    def preload_documents(self, domain: str):
        if domain in self.preload_domains:
            return
        
        self.preload_domains.add(domain)
        
        def _preload():
            try:
                pattern = os.path.join(self.doc_dir, f"{domain}_store_epoch_*.npz")
                files = glob.glob(pattern)
                
                if not files:
                    return
                
                files.sort(key=lambda x: int(x.split('_epoch_')[1].split('.')[0]), reverse=True)
                latest_file = files[0]
                
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
        
        self.preload_future = self.executor.submit(_preload)

    def incremental_update(self, new_temporal_vectors, new_spatial_vectors, domain: str = None, epoch: int = None):
        try:
            max_vectors = 5000
            
            if len(new_temporal_vectors) > 0:
                new_temp_array = np.ascontiguousarray(new_temporal_vectors, dtype=np.float32)
                
                if hasattr(self, 'index_temporal') and self.index_temporal is not None:
                    self.index_temporal.add(new_temp_array)
                
                self.temporal_vectors.extend([v for v in new_temporal_vectors])
                if len(self.temporal_vectors) > max_vectors:
                    excess = len(self.temporal_vectors) - max_vectors
                    self.temporal_vectors = self.temporal_vectors[excess:]
                    self._rebuild_indices()
            
            if len(new_spatial_vectors) > 0:
                new_spat_array = np.ascontiguousarray(new_spatial_vectors, dtype=np.float32)
                
                if hasattr(self, 'index_spatial') and self.index_spatial is not None:
                    self.index_spatial.add(new_spat_array)
                
                self.spatial_vectors.extend([v for v in new_spatial_vectors])
                if len(self.spatial_vectors) > max_vectors:
                    excess = len(self.spatial_vectors) - max_vectors
                    self.spatial_vectors = self.spatial_vectors[excess:]
                    self._rebuild_indices()
            
            self.temporal_cache.clear()
            self.spatial_cache.clear()
            self.cache_access_order.clear()
            
            self.stats["update_count"] += 1
            
        except Exception as e:
            self.logger.error(f"Error during incremental update: {e}")
            self._rebuild_indices()

    def _clean_cache_lru(self, cache: dict):
        if len(cache) <= self.cache_size:
            return
            
        sorted_keys = sorted(self.cache_access_order.items(), key=lambda x: x[1])
        keys_to_remove = [key for key, _ in sorted_keys[:len(cache) - self.cache_size]]
        
        for key in keys_to_remove:
            cache.pop(key, None)
            self.cache_access_order.pop(key, None)

    def use_memory_mapping(self):
        try:
            threshold = 5000
            
            if len(self.temporal_vectors) > threshold:
                mmap_dir = os.path.join(self.doc_dir, "mmap_files")
                os.makedirs(mmap_dir, exist_ok=True)
                
                temp_file = os.path.join(mmap_dir, f"temporal_vectors_{id(self)}.dat")
                
                temporal_array = np.stack([v for v in self.temporal_vectors])
                
                fp = np.memmap(temp_file, dtype='float32', mode='w+', 
                              shape=temporal_array.shape)
                fp[:] = temporal_array[:]
                fp.flush()
                
                self.temporal_vectors_mmap = np.memmap(temp_file, dtype='float32', 
                                                     mode='r', shape=temporal_array.shape)
                self.temporal_mmap_file = temp_file
                
                self.temporal_vectors_length = len(self.temporal_vectors)
                
                self.temporal_vectors = None
                
                self.logger.info(f"Using memory mapping for {self.temporal_vectors_length} temporal vectors")
            
            if len(self.spatial_vectors) > threshold:
                mmap_dir = os.path.join(self.doc_dir, "mmap_files")
                os.makedirs(mmap_dir, exist_ok=True)
                
                temp_file = os.path.join(mmap_dir, f"spatial_vectors_{id(self)}.dat")
                
                spatial_array = np.stack([v for v in self.spatial_vectors])
                
                fp = np.memmap(temp_file, dtype='float32', mode='w+', 
                              shape=spatial_array.shape)
                fp[:] = spatial_array[:]
                fp.flush()
                
                self.spatial_vectors_mmap = np.memmap(temp_file, dtype='float32', 
                                                    mode='r', shape=spatial_array.shape)
                self.spatial_mmap_file = temp_file
                
                self.spatial_vectors_length = len(self.spatial_vectors)
                
                self.spatial_vectors = None
                
                self.logger.info(f"Using memory mapping for {self.spatial_vectors_length} spatial vectors")
            
            self.using_mmap = True
            
            self._update_indices_with_mmap()
            
        except Exception as e:
            self.logger.error(f"Failed to use memory mapping: {e}")
            traceback.print_exc()
            self._cleanup_mmap_files()

    def _update_indices_with_mmap(self):
        try:
            self.index_temporal = None
            self.index_spatial = None
            
            if hasattr(self, 'temporal_vectors_mmap'):
                self.index_temporal = faiss.IndexFlatL2(self.dim)
                self.index_temporal.add(self.temporal_vectors_mmap)
                self.logger.info(f"Updated temporal index with memory mapped vectors")
            
            if hasattr(self, 'spatial_vectors_mmap'):
                self.index_spatial = faiss.IndexFlatL2(self.dim)
                self.index_spatial.add(self.spatial_vectors_mmap)
                self.logger.info(f"Updated spatial index with memory mapped vectors")
            
        except Exception as e:
            self.logger.error(f"Failed to update indices with memory mapped vectors: {e}")
            traceback.print_exc()
            self._rebuild_indices()

    def _cleanup_mmap_files(self):
        try:
            if hasattr(self, 'temporal_mmap_file') and os.path.exists(self.temporal_mmap_file):
                if hasattr(self, 'temporal_vectors_mmap'):
                    del self.temporal_vectors_mmap
                os.remove(self.temporal_mmap_file)
                self.logger.info(f"Deleted temporal vector mapping file: {self.temporal_mmap_file}")
            
            if hasattr(self, 'spatial_mmap_file') and os.path.exists(self.spatial_mmap_file):
                if hasattr(self, 'spatial_vectors_mmap'):
                    del self.spatial_vectors_mmap
                os.remove(self.spatial_mmap_file)
                self.logger.info(f"Deleted spatial vector mapping file: {self.spatial_mmap_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to clean up memory mapping files: {e}")
            traceback.print_exc()

    def __del__(self):
        try:
            if hasattr(self, 'faiss_res'):
                del self.faiss_res
        except:
            pass
