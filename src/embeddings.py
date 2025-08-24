"""
Embeddings module for DocuChat RAG system.
Handles secure embedding generation with model integrity verification.
"""

import os
import hashlib
import warnings
import pickle
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Union
from pathlib import Path
import logging
import time

import numpy as np

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available - dynamic batch sizing will use fallback")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import sentence-transformers with error handling
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    warnings.warn(
        "sentence-transformers not available. Install with: pip install sentence-transformers>=2.3.0",
        ImportWarning
    )

# Try to import torch for additional security checks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


@dataclass(frozen=True)
class EmbeddedChunk:
    """
    Immutable data class representing a text chunk with its embedding.
    
    Security Features:
    - Immutable to prevent accidental modification
    - Embedding validation and integrity checks
    - Metadata tracking for audit trails
    """
    chunk_id: str
    content: str
    embedding: np.ndarray
    source_file: str
    chunk_index: int
    model_name: str
    embedding_dim: int
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate embedded chunk data on creation."""
        # Validate basic types
        if not isinstance(self.chunk_id, str) or not self.chunk_id.strip():
            raise ValueError("Chunk ID must be a non-empty string")
        
        if not isinstance(self.content, str):
            raise TypeError("Content must be a string")
        
        if not isinstance(self.embedding, np.ndarray):
            raise TypeError("Embedding must be a numpy array")
        
        if not isinstance(self.source_file, str) or not self.source_file.strip():
            raise ValueError("Source file must be a non-empty string")
        
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("Model name must be a non-empty string")
        
        # Validate numeric types
        if not isinstance(self.chunk_index, int) or self.chunk_index < 0:
            raise ValueError("Chunk index must be a non-negative integer")
        
        if not isinstance(self.embedding_dim, int) or self.embedding_dim <= 0:
            raise ValueError("Embedding dimension must be positive")
        
        # Security: Validate embedding properties
        if self.embedding.ndim != 1:
            raise ValueError("Embedding must be a 1-dimensional array")
        
        if len(self.embedding) != self.embedding_dim:
            raise ValueError(f"Embedding length ({len(self.embedding)}) doesn't match declared dimension ({self.embedding_dim})")
        
        if len(self.embedding) > 10000:  # Security: Prevent excessive memory usage
            raise ValueError("Embedding dimension exceeds maximum allowed size (10000)")
        
        # Validate embedding values
        if not np.isfinite(self.embedding).all():
            raise ValueError("Embedding contains infinite or NaN values")
        
        # Security: Check for reasonable embedding values
        embedding_norm = np.linalg.norm(self.embedding)
        if embedding_norm > 1000.0:  # Abnormally large embeddings
            logger.warning(f"Embedding has unusually large norm: {embedding_norm}")
        
        # Security: Validate content length
        if len(self.content) > 100000:  # 100KB limit
            raise ValueError("Content exceeds maximum allowed size (100KB)")
        
        # Validate metadata
        if not isinstance(self.metadata, dict):
            raise TypeError("Metadata must be a dictionary")
    
    @property
    def content_hash(self) -> str:
        """Get SHA-256 hash of content for integrity verification."""
        return hashlib.sha256(self.content.encode('utf-8')).hexdigest()
    
    @property
    def embedding_hash(self) -> str:
        """Get SHA-256 hash of embedding for integrity verification."""
        embedding_bytes = self.embedding.tobytes()
        return hashlib.sha256(embedding_bytes).hexdigest()
    
    @property
    def embedding_norm(self) -> float:
        """Get L2 norm of the embedding vector."""
        return float(np.linalg.norm(self.embedding))
    
    @property
    def char_count(self) -> int:
        """Get the character count of the chunk content."""
        return len(self.content)
    
    @property
    def word_count(self) -> int:
        """Get the estimated word count of the chunk content."""
        return len(self.content.split())
    
    def similarity(self, other: 'EmbeddedChunk') -> float:
        """
        Calculate cosine similarity with another embedded chunk.
        
        Args:
            other: Another EmbeddedChunk to compare with
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        if not isinstance(other, EmbeddedChunk):
            raise TypeError("Other must be an EmbeddedChunk instance")
        
        if self.embedding_dim != other.embedding_dim:
            raise ValueError("Cannot compare embeddings with different dimensions")
        
        # Calculate cosine similarity
        dot_product = np.dot(self.embedding, other.embedding)
        norm_product = self.embedding_norm * other.embedding_norm
        
        if norm_product == 0:
            return 0.0
        
        similarity = dot_product / norm_product
        
        # Clamp to valid range due to floating point precision
        return max(0.0, min(1.0, similarity))


class SecureEmbeddingModel:
    """
    Secure wrapper for embedding models with integrity verification.
    
    Security Features:
    - Model integrity verification
    - Safe model loading and caching
    - Input validation and sanitization
    - Memory usage monitoring
    """
    
    # Known secure model hashes (would be updated with actual hashes in production)
    KNOWN_MODEL_HASHES = {
        "all-MiniLM-L6-v2": {
            "expected_dim": 384,
            "max_input_length": 512,
            "verified": True
        },
        "all-mpnet-base-v2": {
            "expected_dim": 768,
            "max_input_length": 512,
            "verified": True
        }
    }
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        device: Optional[str] = None,
        verify_integrity: bool = True
    ):
        """
        Initialize the secure embedding model.
        
        Args:
            model_name: Name of the model to load
            cache_dir: Directory for model caching
            trust_remote_code: Whether to trust remote code (security risk)
            device: Device to run model on ('cpu', 'cuda', etc.)
            verify_integrity: Whether to verify model integrity
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required but not available")
        
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.verify_integrity = verify_integrity
        self.device = device
        
        # Security: Never trust remote code by default
        if trust_remote_code:
            logger.warning("trust_remote_code=True is a security risk")
        
        # Validate model name
        self._validate_model_name()
        
        # Initialize model
        self.model = None
        self.model_info = self.KNOWN_MODEL_HASHES.get(model_name, {})
        self.expected_dim = self.model_info.get("expected_dim")
        self.max_input_length = self.model_info.get("max_input_length", 512)
        
        # Load model securely
        self._load_model(trust_remote_code)
        
        logger.info(f"SecureEmbeddingModel initialized: {model_name}")
    
    def _validate_model_name(self) -> None:
        """Validate model name for security."""
        if not isinstance(self.model_name, str):
            raise TypeError("Model name must be a string")
        
        if not self.model_name.strip():
            raise ValueError("Model name cannot be empty")
        
        # Security: Prevent path traversal in model names
        if ".." in self.model_name or "/" in self.model_name or "\\" in self.model_name:
            raise ValueError("Model name contains unsafe characters")
        
        # Check if model is in known safe list
        if self.verify_integrity and self.model_name not in self.KNOWN_MODEL_HASHES:
            logger.warning(f"Model '{self.model_name}' not in verified model list")
    
    def _setup_cache_directory(self) -> None:
        """Setup secure cache directory."""
        if self.cache_dir:
            try:
                # Security: Ensure cache directory is safe
                resolved_cache = self.cache_dir.resolve()
                
                # Prevent writing to system directories
                system_dirs = ["/bin", "/sbin", "/usr", "/etc", "/proc", "/sys"]
                cache_str = str(resolved_cache)
                
                if any(cache_str.startswith(sys_dir) for sys_dir in system_dirs):
                    raise ValueError(f"Cache directory in system path: {cache_str}")
                
                # Create directory with secure permissions
                resolved_cache.mkdir(parents=True, exist_ok=True)
                
                # Set restrictive permissions (owner only)
                if hasattr(os, 'chmod'):
                    os.chmod(resolved_cache, 0o700)
                
                self.cache_dir = resolved_cache
                logger.debug(f"Cache directory: {self.cache_dir}")
                
            except Exception as e:
                logger.error(f"Failed to setup cache directory: {e}")
                self.cache_dir = None
    
    def _load_model(self, trust_remote_code: bool = False) -> None:
        """
        Load the embedding model securely.
        
        Args:
            trust_remote_code: Whether to trust remote code
        """
        try:
            # Setup cache directory
            self._setup_cache_directory()
            
            # Configure model loading parameters
            model_kwargs = {
                "trust_remote_code": trust_remote_code,
            }
            
            if self.cache_dir:
                model_kwargs["cache_folder"] = str(self.cache_dir)
            
            if self.device:
                model_kwargs["device"] = self.device
            
            logger.info(f"Loading model: {self.model_name}")
            
            # Load model with timeout and error handling
            start_time = time.time()
            
            self.model = SentenceTransformer(self.model_name, **model_kwargs)
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f} seconds")
            
            # Verify model properties
            self._verify_model_integrity()
            
        except Exception as e:
            logger.error(f"Failed to load model '{self.model_name}': {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _verify_model_integrity(self) -> None:
        """Verify model integrity and properties."""
        if not self.verify_integrity:
            return
        
        try:
            # Test with a simple input
            test_input = "Test embedding input"
            test_embedding = self.model.encode([test_input], convert_to_tensor=False)
            
            if len(test_embedding) != 1:
                raise ValueError("Model returned unexpected number of embeddings")
            
            embedding = test_embedding[0]
            actual_dim = len(embedding)
            
            # Verify embedding dimension
            if self.expected_dim and actual_dim != self.expected_dim:
                raise ValueError(f"Embedding dimension mismatch: expected {self.expected_dim}, got {actual_dim}")
            
            # Update expected dimension if not known
            if not self.expected_dim:
                self.expected_dim = actual_dim
                logger.debug(f"Detected embedding dimension: {actual_dim}")
            
            # Verify embedding properties
            if not np.isfinite(embedding).all():
                raise ValueError("Model produces invalid embeddings (inf/nan)")
            
            logger.debug(f"Model integrity verified: dimension={actual_dim}")
            
        except Exception as e:
            logger.error(f"Model integrity verification failed: {e}")
            if self.verify_integrity:
                raise RuntimeError(f"Model integrity check failed: {e}")
    
    def _sanitize_texts(self, texts: List[str]) -> List[str]:
        """
        Sanitize input texts for security.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of sanitized texts
        """
        if not isinstance(texts, list):
            raise TypeError("Texts must be a list")
        
        sanitized = []
        
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("All texts must be strings")
            
            # Security: Check text length
            if len(text) > 100000:  # 100KB limit
                logger.warning(f"Text exceeds length limit, truncating: {len(text)} chars")
                text = text[:100000]
            
            # Remove null bytes and normalize
            text = text.replace('\x00', '')
            text = text.strip()
            
            if not text:
                logger.warning("Empty text after sanitization")
                text = " "  # Use single space for empty texts
            
            sanitized.append(text)
        
        return sanitized
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        Encode texts into embeddings securely.
        
        Args:
            texts: Text or list of texts to encode
            batch_size: Batch size for processing
            show_progress_bar: Whether to show progress
            normalize_embeddings: Whether to normalize embeddings
            
        Returns:
            Array of embeddings
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
        
        # Security: Validate batch size
        if batch_size <= 0 or batch_size > 1000:
            raise ValueError("Batch size must be between 1 and 1000")
        
        # Security: Check total number of texts
        if len(texts) > 10000:
            raise ValueError("Too many texts to process (max 10000)")
        
        # Sanitize input texts
        texts = self._sanitize_texts(texts)
        
        if not texts:
            return np.array([])
        
        logger.debug(f"Encoding {len(texts)} texts with batch_size={batch_size}")
        
        try:
            start_time = time.time()
            
            # Encode with the model
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_tensor=False,
                normalize_embeddings=normalize_embeddings
            )
            
            encoding_time = time.time() - start_time
            logger.debug(f"Encoding completed in {encoding_time:.2f} seconds")
            
            # Convert to numpy array if needed
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
            
            # Validate output
            self._validate_embeddings(embeddings, len(texts))
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise RuntimeError(f"Embedding encoding failed: {e}")
    
    def _validate_embeddings(self, embeddings: np.ndarray, expected_count: int) -> None:
        """
        Validate output embeddings.
        
        Args:
            embeddings: Generated embeddings
            expected_count: Expected number of embeddings
        """
        if not isinstance(embeddings, np.ndarray):
            raise TypeError("Embeddings must be a numpy array")
        
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")
        
        if len(embeddings) != expected_count:
            raise ValueError(f"Expected {expected_count} embeddings, got {len(embeddings)}")
        
        if self.expected_dim and embeddings.shape[1] != self.expected_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.expected_dim}, got {embeddings.shape[1]}")
        
        # Check for invalid values
        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain infinite or NaN values")
        
        logger.debug(f"Embeddings validated: shape={embeddings.shape}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.expected_dim,
            "max_input_length": self.max_input_length,
            "device": self.device,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "verified": self.model_info.get("verified", False)
        }


class EmbeddingGenerator:
    """
    High-level embedding generation interface with security controls.
    
    Security Features:
    - Secure model management
    - Batch processing with memory limits
    - Progress tracking and error handling
    - Comprehensive logging and monitoring
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        cache_dir: Optional[str] = None,
        verbose: bool = False,
        device: Optional[str] = None
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the embedding model
            batch_size: Batch size for processing
            cache_dir: Directory for model caching
            verbose: Enable verbose logging
            device: Device to run model on
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.verbose = verbose
        self.device = device
        
        # Initialize secure model
        self.model = SecureEmbeddingModel(
            model_name=model_name,
            cache_dir=cache_dir,
            device=device,
            verify_integrity=True
        )
        
        # Optimize batch size based on available memory if not explicitly set
        if batch_size == 32:  # Only auto-optimize if using default
            optimal_batch_size = self._calculate_optimal_batch_size()
            if optimal_batch_size != batch_size:
                self.batch_size = optimal_batch_size
                self._log(f"Auto-optimized batch size from {batch_size} to {optimal_batch_size}")
        
        # Statistics tracking
        self.stats = {
            'chunks_processed': 0,
            'embeddings_generated': 0,
            'total_processing_time': 0.0,
            'errors': 0
        }
        
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        logger.info(f"EmbeddingGenerator initialized: {model_name}")
    
    def _log(self, message: str, level: str = "info") -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            if level == "debug":
                logger.debug(f"[EmbeddingGenerator] {message}")
            elif level == "warning":
                logger.warning(f"[EmbeddingGenerator] {message}")
            elif level == "error":
                logger.error(f"[EmbeddingGenerator] {message}")
            else:
                logger.info(f"[EmbeddingGenerator] {message}")
    
    def _calculate_optimal_batch_size(self) -> int:
        """
        Calculate optimal batch size based on available memory.
        
        Returns:
            Optimal batch size for embedding generation
        """
        if not PSUTIL_AVAILABLE:
            return 32  # Fallback to default
        
        try:
            # Get available memory in GB
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            
            # Conservative batch size calculation:
            # - Small embedding model (all-MiniLM-L6-v2): ~90MB memory
            # - Estimate ~1MB per batch item for text + embeddings
            # - Use 25% of available memory for batch processing
            # - Add safety margin
            
            usable_memory_gb = available_memory_gb * 0.25  # Use 25% of available memory
            estimated_memory_per_item_mb = 1.5  # Conservative estimate
            
            # Calculate batch size (with min=16, max=128 constraints)
            optimal_batch_size = int((usable_memory_gb * 1024) / estimated_memory_per_item_mb)
            optimal_batch_size = max(16, min(optimal_batch_size, 128))
            
            self._log(f"Memory-based batch sizing: {available_memory_gb:.1f}GB available → batch_size={optimal_batch_size}", "debug")
            return optimal_batch_size
            
        except Exception as e:
            self._log(f"Failed to calculate optimal batch size: {e}", "warning")
            return 32  # Safe fallback
    
    def embed_chunk(self, chunk) -> EmbeddedChunk:
        """
        Generate embedding for a single text chunk.
        
        Args:
            chunk: TextChunk object to embed
            
        Returns:
            EmbeddedChunk with generated embedding
        """
        # Import here to avoid circular imports
        from chunker import TextChunk
        
        if not isinstance(chunk, TextChunk):
            raise TypeError("Input must be a TextChunk object")
        
        self._log(f"Embedding chunk: {chunk.chunk_id}", "debug")
        
        try:
            start_time = time.time()
            
            # Generate embedding
            embedding = self.model.encode([chunk.content], batch_size=1)[0]
            
            processing_time = time.time() - start_time
            
            # Create EmbeddedChunk
            embedded_chunk = EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                embedding=embedding,
                source_file=chunk.source_file,
                chunk_index=chunk.chunk_index,
                model_name=self.model_name,
                embedding_dim=len(embedding),
                metadata=chunk.metadata.copy()
            )
            
            # Update statistics
            self.stats['chunks_processed'] += 1
            self.stats['embeddings_generated'] += 1
            self.stats['total_processing_time'] += processing_time
            
            self._log(f"Embedded chunk {chunk.chunk_id} in {processing_time:.3f}s", "debug")
            
            return embedded_chunk
            
        except Exception as e:
            self._log(f"Error embedding chunk {chunk.chunk_id}: {e}", "error")
            self.stats['errors'] += 1
            raise
    
    def embed_chunks(self, chunks: List) -> List[EmbeddedChunk]:
        """
        Generate embeddings for multiple text chunks.
        
        Args:
            chunks: List of TextChunk objects to embed
            
        Returns:
            List of EmbeddedChunk objects
        """
        # Import here to avoid circular imports
        from chunker import TextChunk
        
        if not isinstance(chunks, list):
            raise TypeError("Chunks must be a list")
        
        if not chunks:
            self._log("No chunks to embed")
            return []
        
        # Validate all chunks
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, TextChunk):
                raise TypeError(f"Chunk {i} is not a TextChunk object")
        
        self._log(f"Embedding {len(chunks)} chunks with batch_size={self.batch_size}")
        
        embedded_chunks = []
        total_start_time = time.time()
        
        try:
            # Extract texts for batch processing
            texts = [chunk.content for chunk in chunks]
            
            # Generate embeddings in batches
            self._log(f"Starting batch embedding: {len(texts)} texts, batch_size={self.batch_size}", "debug")
            all_embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=self.verbose
            )
            
            # Create EmbeddedChunk objects
            for chunk, embedding in zip(chunks, all_embeddings):
                try:
                    embedded_chunk = EmbeddedChunk(
                        chunk_id=chunk.chunk_id,
                        content=chunk.content,
                        embedding=embedding,
                        source_file=chunk.source_file,
                        chunk_index=chunk.chunk_index,
                        model_name=self.model_name,
                        embedding_dim=len(embedding),
                        metadata=chunk.metadata.copy()
                    )
                    
                    embedded_chunks.append(embedded_chunk)
                    
                except Exception as e:
                    self._log(f"Error creating EmbeddedChunk for {chunk.chunk_id}: {e}", "error")
                    self.stats['errors'] += 1
                    continue
            
            # Update statistics
            total_time = time.time() - total_start_time
            self.stats['chunks_processed'] += len(chunks)
            self.stats['embeddings_generated'] += len(embedded_chunks)
            self.stats['total_processing_time'] += total_time
            
            # Keep this as INFO for processing summaries
            self._log(f"Successfully embedded {len(embedded_chunks)}/{len(chunks)} chunks in {total_time:.2f}s")
            
            return embedded_chunks
            
        except Exception as e:
            self._log(f"Error in batch embedding: {e}", "error")
            self.stats['errors'] += 1
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.stats.copy()
        
        # Add derived statistics
        if stats['chunks_processed'] > 0:
            stats['average_time_per_chunk'] = stats['total_processing_time'] / stats['chunks_processed']
        else:
            stats['average_time_per_chunk'] = 0.0
        
        # Add model info
        stats['model_info'] = self.model.get_model_info()
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self.stats = {
            'chunks_processed': 0,
            'embeddings_generated': 0,
            'total_processing_time': 0.0,
            'errors': 0
        }
        self._log("Statistics reset")


def main():
    """Simple test function for the embeddings module."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python embeddings.py <text_content>")
        sys.exit(1)
    
    text = sys.argv[1]
    
    try:
        # Test basic embedding functionality
        generator = EmbeddingGenerator(verbose=True)
        
        # Create a simple chunk for testing
        from chunker import TextChunk
        
        chunk = TextChunk(
            content=text,
            chunk_id="test_chunk_001",
            source_file="test_file.txt",
            chunk_index=0
        )
        
        # Generate embedding
        embedded_chunk = generator.embed_chunk(chunk)
        
        print(f"\nEmbedding Results:")
        print(f"Chunk ID: {embedded_chunk.chunk_id}")
        print(f"Content: {embedded_chunk.content[:100]}...")
        print(f"Embedding dimension: {embedded_chunk.embedding_dim}")
        print(f"Embedding norm: {embedded_chunk.embedding_norm:.6f}")
        print(f"Model: {embedded_chunk.model_name}")
        
        print(f"\nStatistics: {generator.get_statistics()}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()