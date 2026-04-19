"""
Vector Store for RAG System.
Uses FAISS for efficient similarity search over embeddings.
"""

import faiss
import numpy as np
import pickle
import json
import logging
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS-based vector store for managing embeddings and metadata.
    """
    
    def __init__(self, embedding_dim: int):
        """
        Initialize vector store.
        
        Args:
            embedding_dim: Dimensionality of embeddings
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.metadata = []
        self.size = 0
        logger.info(f"Initialized FAISS index with dimension {embedding_dim}")
    
    def add_embeddings(self, embeddings: np.ndarray, 
                      metadata: List[Dict[str, Any]]) -> None:
        """
        Add embeddings and metadata to store.
        
        Args:
            embeddings: numpy array of shape (n, embedding_dim)
            metadata: List of metadata dictionaries
        """
        if embeddings.shape[0] != len(metadata):
            raise ValueError("Embeddings and metadata lengths don't match")
        
        # Ensure embeddings are float32
        embeddings = embeddings.astype(np.float32)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store metadata
        self.metadata.extend(metadata)
        self.size += len(embeddings)
        
        logger.info(f"Added {len(embeddings)} embeddings. Total: {self.size}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5
              ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for most similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (metadata, distance) tuples, sorted by similarity
        """
        if self.size == 0:
            logger.warning("Vector store is empty")
            return []
        
        # Prepare query
        query = query_embedding.astype(np.float32).reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query, min(k, self.size))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.metadata):
                # Convert L2 distance to similarity (lower distance = higher similarity)
                similarity = 1.0 / (1.0 + dist)
                results.append((self.metadata[idx], similarity))
        
        logger.info(f"Search returned {len(results)} results")
        return results
    
    def search_by_similarity(self, query_embedding: np.ndarray, 
                            threshold: float = 0.5, k: int = 5
                            ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search with similarity threshold filtering.
        
        Args:
            query_embedding: Query embedding vector
            threshold: Minimum similarity score (0-1)
            k: Maximum number of results
            
        Returns:
            List of (metadata, similarity) tuples above threshold
        """
        results = self.search(query_embedding, k=k)
        
        # Filter by threshold
        filtered = [(meta, sim) for meta, sim in results if sim >= threshold]
        
        return filtered
    
    def get_all_metadata(self) -> List[Dict[str, Any]]:
        """Get all stored metadata."""
        return self.metadata.copy()
    
    def clear(self) -> None:
        """Clear all entries from store."""
        self.index.reset()
        self.metadata = []
        self.size = 0
        logger.info("Vector store cleared")
    
    def save(self, index_path: str, metadata_path: str) -> None:
        """
        Save vector store to disk.
        
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata
        """
        try:
            # Create directories if needed
            Path(index_path).parent.mkdir(parents=True, exist_ok=True)
            Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save index
            faiss.write_index(self.index, index_path)
            
            # Save metadata
            with open(metadata_path, "wb") as f:
                pickle.dump(self.metadata, f)
            
            logger.info(f"Vector store saved: {index_path}, {metadata_path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            raise
    
    @staticmethod
    def load(index_path: str, metadata_path: str) -> "VectorStore":
        """
        Load vector store from disk.
        
        Args:
            index_path: Path to FAISS index
            metadata_path: Path to metadata
            
        Returns:
            Loaded VectorStore instance
        """
        try:
            # Load index
            index = faiss.read_index(index_path)
            
            # Load metadata
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
            
            # Create store
            store = VectorStore(index.d)
            store.index = index
            store.metadata = metadata
            store.size = len(metadata)
            
            logger.info(f"Vector store loaded from disk")
            return store
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise
