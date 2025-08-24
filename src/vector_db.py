"""
Vector database module for DocuChat RAG system.
Handles ChromaDB integration with security and database integrity checks.
"""

import os
import json
import hashlib
import shutil
import warnings
from typing import List, Dict, Optional, Tuple, Any, Union
from pathlib import Path
import logging
import time
import uuid

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import ChromaDB with error handling
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None
    Settings = None
    embedding_functions = None
    CHROMADB_AVAILABLE = False
    warnings.warn(
        "chromadb not available. Install with: pip install chromadb>=0.4.18",
        ImportWarning
    )


class VectorDatabaseError(Exception):
    """Custom exception for vector database operations."""
    pass


class SecurityViolationError(VectorDatabaseError):
    """Exception raised when security violations are detected."""
    pass


class ChromaDBVectorStore:
    """
    Secure ChromaDB vector store with integrity checks and audit logging.
    
    Security Features:
    - Path validation and sanitization
    - Database integrity verification
    - Access control and audit logging
    - Safe collection management
    - Data validation and sanitization
    """
    
    DEFAULT_COLLECTION_NAME = "docuchat_embeddings"
    
    def __init__(
        self,
        persist_directory: str = "./chroma",
        collection_name: str = DEFAULT_COLLECTION_NAME,
        verbose: bool = False,
        verify_integrity: bool = True,
        max_collection_size: int = 1_000_000  # Max 1M documents
    ):
        """
        Initialize the ChromaDB vector store.
        
        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the collection
            verbose: Enable verbose logging
            verify_integrity: Whether to verify database integrity
            max_collection_size: Maximum number of documents in collection
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb is required but not available")
        
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.verbose = verbose
        self.verify_integrity = verify_integrity
        self.max_collection_size = max_collection_size
        
        # Validate inputs
        self._validate_collection_name()
        self._validate_persist_directory()
        
        # Initialize database
        self.client = None
        self.collection = None
        self._setup_database()
        
        # Statistics tracking
        self.stats = {
            'documents_added': 0,
            'documents_updated': 0,
            'documents_deleted': 0,
            'queries_performed': 0,
            'total_operation_time': 0.0,
            'errors': 0
        }
        
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        logger.info(f"ChromaDBVectorStore initialized: {collection_name}")
    
    def _log(self, message: str, level: str = "info") -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            if level == "debug":
                logger.debug(f"[ChromaDBVectorStore] {message}")
            elif level == "warning":
                logger.warning(f"[ChromaDBVectorStore] {message}")
            elif level == "error":
                logger.error(f"[ChromaDBVectorStore] {message}")
            else:
                logger.info(f"[ChromaDBVectorStore] {message}")
    
    def _validate_collection_name(self) -> None:
        """Validate collection name for security."""
        if not isinstance(self.collection_name, str):
            raise TypeError("Collection name must be a string")
        
        if not self.collection_name.strip():
            raise ValueError("Collection name cannot be empty")
        
        # Security: Prevent SQL injection and path traversal
        if any(char in self.collection_name for char in ['/', '\\', '..', ';', "'", '"', '<', '>']):
            raise SecurityViolationError("Collection name contains unsafe characters")
        
        # Limit length
        if len(self.collection_name) > 100:
            raise ValueError("Collection name too long (max 100 characters)")
        
        # Ensure valid identifier
        if not self.collection_name.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Collection name must be alphanumeric with _ or - only")
    
    def _validate_persist_directory(self) -> None:
        """Validate and setup persist directory for security."""
        try:
            # Security: Resolve path to prevent traversal attacks
            resolved_path = self.persist_directory.resolve()
            
            # Prevent writing to system directories
            system_dirs = ["/bin", "/sbin", "/usr", "/etc", "/proc", "/sys", "/root"]
            path_str = str(resolved_path)
            
            if any(path_str.startswith(sys_dir) for sys_dir in system_dirs):
                raise SecurityViolationError(f"Persist directory in system path: {path_str}")
            
            # Update path to resolved version
            self.persist_directory = resolved_path
            
            self._log(f"Persist directory: {self.persist_directory}")
            
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Cannot resolve persist directory: {e}")
    
    def _setup_database(self) -> None:
        """Setup ChromaDB client and collection securely."""
        try:
            # Ensure persist directory exists with secure permissions
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            
            # Set restrictive permissions (owner only)
            if hasattr(os, 'chmod'):
                os.chmod(self.persist_directory, 0o700)
            
            # Configure ChromaDB settings
            settings = Settings(
                persist_directory=str(self.persist_directory),
                anonymized_telemetry=False,  # Disable telemetry for privacy
                allow_reset=True
            )
            
            # Initialize client
            self._log("Initializing ChromaDB client")
            self.client = chromadb.PersistentClient(settings=settings)
            
            # Get or create collection
            self._setup_collection()
            
            # Verify database integrity if enabled
            if self.verify_integrity:
                self._verify_database_integrity()
            
            self._log("Database setup completed", "debug")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            raise VectorDatabaseError(f"Failed to setup database: {e}")
    
    def _setup_collection(self) -> None:
        """Setup the embeddings collection."""
        try:
            # Check if collection exists
            existing_collections = [col.name for col in self.client.list_collections()]
            
            if self.collection_name in existing_collections:
                self._log(f"Loading existing collection: {self.collection_name}")
                self.collection = self.client.get_collection(self.collection_name)
            else:
                self._log(f"Creating new collection: {self.collection_name}")
                
                # Create collection with metadata
                metadata = {
                    "created_at": str(int(time.time())),
                    "docuchat_version": "2.0",
                    "security_verified": True
                }
                
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata=metadata
                )
            
            # Get collection info
            collection_count = self.collection.count()
            self._log(f"Collection loaded: {collection_count} documents", "debug")
            
            # Security: Check collection size
            if collection_count > self.max_collection_size:
                raise SecurityViolationError(f"Collection exceeds maximum size: {collection_count}")
            
        except Exception as e:
            logger.error(f"Collection setup failed: {e}")
            raise VectorDatabaseError(f"Failed to setup collection: {e}")
    
    def _verify_database_integrity(self) -> None:
        """Verify database integrity and detect corruption."""
        try:
            self._log("Verifying database integrity")
            
            # Basic health check
            if self.collection is None:
                raise VectorDatabaseError("Collection not initialized")
            
            # Check if we can perform basic operations
            collection_count = self.collection.count()
            
            if collection_count > 0:
                # Sample a few documents to verify data integrity
                sample_size = min(10, collection_count)
                
                # Get sample documents
                sample_results = self.collection.get(limit=sample_size)
                
                if not sample_results or 'ids' not in sample_results:
                    raise VectorDatabaseError("Cannot retrieve sample documents")
                
                # Verify sample data integrity
                for i, doc_id in enumerate(sample_results['ids']):
                    if not doc_id or not isinstance(doc_id, str):
                        raise VectorDatabaseError(f"Invalid document ID at index {i}")
                
                self._log(f"Integrity check passed: sampled {sample_size} documents", "debug")
            
            self._log("Database integrity verified", "debug")
            
        except Exception as e:
            logger.error(f"Database integrity check failed: {e}")
            if self.verify_integrity:
                raise VectorDatabaseError(f"Database integrity verification failed: {e}")
    
    def _sanitize_document_id(self, doc_id: str) -> str:
        """
        Sanitize document ID for security.
        
        Args:
            doc_id: Original document ID
            
        Returns:
            Sanitized document ID
        """
        if not isinstance(doc_id, str):
            raise TypeError("Document ID must be a string")
        
        if not doc_id.strip():
            raise ValueError("Document ID cannot be empty")
        
        # Remove potentially dangerous characters
        sanitized = doc_id.replace('\x00', '').replace('\n', '').replace('\r', '')
        
        # Limit length
        if len(sanitized) > 200:
            # Create hash-based ID for long IDs
            hash_suffix = hashlib.sha256(sanitized.encode()).hexdigest()[:16]
            sanitized = sanitized[:180] + "_" + hash_suffix
        
        return sanitized
    
    def _validate_embedding(self, embedding: np.ndarray) -> None:
        """
        Validate embedding data for security.
        
        Args:
            embedding: Embedding array to validate
        """
        if not isinstance(embedding, np.ndarray):
            raise TypeError("Embedding must be a numpy array")
        
        if embedding.ndim != 1:
            raise ValueError("Embedding must be 1-dimensional")
        
        if len(embedding) == 0:
            raise ValueError("Embedding cannot be empty")
        
        if len(embedding) > 10000:
            raise SecurityViolationError("Embedding dimension too large (max 10000)")
        
        # Check for invalid values
        if not np.isfinite(embedding).all():
            raise ValueError("Embedding contains infinite or NaN values")
        
        # Check for reasonable magnitude
        embedding_norm = np.linalg.norm(embedding)
        if embedding_norm > 1000.0:
            logger.warning(f"Embedding has large norm: {embedding_norm}")
    
    def add_embedded_chunk(self, embedded_chunk) -> bool:
        """
        Add a single embedded chunk to the vector store.
        
        Args:
            embedded_chunk: EmbeddedChunk object to add
            
        Returns:
            True if successful, False otherwise
        """
        # Import here to avoid circular imports
        from embeddings import EmbeddedChunk
        
        if not isinstance(embedded_chunk, EmbeddedChunk):
            raise TypeError("Input must be an EmbeddedChunk object")
        
        return self.add_embedded_chunks([embedded_chunk])
    
    def add_embedded_chunks(self, embedded_chunks: List) -> bool:
        """
        Add multiple embedded chunks to the vector store.
        
        Args:
            embedded_chunks: List of EmbeddedChunk objects to add
            
        Returns:
            True if successful, False otherwise
        """
        # Import here to avoid circular imports
        from embeddings import EmbeddedChunk
        
        if not isinstance(embedded_chunks, list):
            raise TypeError("Embedded chunks must be a list")
        
        if not embedded_chunks:
            self._log("No chunks to add")
            return True
        
        # Validate all chunks
        for i, chunk in enumerate(embedded_chunks):
            if not isinstance(chunk, EmbeddedChunk):
                raise TypeError(f"Chunk {i} is not an EmbeddedChunk object")
        
        # Security: Check collection size limit
        current_count = self.collection.count()
        if current_count + len(embedded_chunks) > self.max_collection_size:
            raise SecurityViolationError(
                f"Adding {len(embedded_chunks)} chunks would exceed maximum collection size"
            )
        
        self._log(f"Adding {len(embedded_chunks)} chunks to vector store")
        
        try:
            start_time = time.time()
            
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            metadatas = []
            documents = []
            
            for chunk in embedded_chunks:
                # Sanitize and validate data
                chunk_id = self._sanitize_document_id(chunk.chunk_id)
                self._validate_embedding(chunk.embedding)
                
                # Prepare metadata (ensure JSON serializable)
                metadata = {
                    "source_file": chunk.source_file,
                    "chunk_index": chunk.chunk_index,
                    "model_name": chunk.model_name,
                    "embedding_dim": chunk.embedding_dim,
                    "created_at": chunk.created_at,
                    "content_hash": chunk.content_hash,
                    "embedding_hash": chunk.embedding_hash,
                    "char_count": chunk.char_count,
                    "word_count": chunk.word_count
                }
                
                # Add custom metadata
                for key, value in chunk.metadata.items():
                    # Ensure value is JSON serializable
                    if isinstance(value, (str, int, float, bool, type(None))):
                        metadata[f"custom_{key}"] = value
                
                ids.append(chunk_id)
                embeddings.append(chunk.embedding.tolist())
                metadatas.append(metadata)
                documents.append(chunk.content)
            
            # Add to collection
            self._log(f"Adding batch to ChromaDB: {len(ids)} chunks", "debug")
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            operation_time = time.time() - start_time
            
            # Update statistics
            self.stats['documents_added'] += len(embedded_chunks)
            self.stats['total_operation_time'] += operation_time
            
            # Keep this as INFO for processing summaries
            self._log(f"Successfully added {len(embedded_chunks)} chunks in {operation_time:.2f}s")
            
            return True
            
        except Exception as e:
            self._log(f"Error adding chunks: {e}", "error")
            self.stats['errors'] += 1
            raise VectorDatabaseError(f"Failed to add chunks: {e}")
    
    def query_similar(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query for similar documents using embedding.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of similar documents with metadata
        """
        # Validate inputs
        self._validate_embedding(query_embedding)
        
        if k <= 0 or k > 1000:
            raise ValueError("k must be between 1 and 1000")
        
        # Security: Validate filter metadata
        if filter_metadata:
            if not isinstance(filter_metadata, dict):
                raise TypeError("Filter metadata must be a dictionary")
            
            # Limit filter complexity
            if len(filter_metadata) > 10:
                raise SecurityViolationError("Too many filter conditions (max 10)")
        
        self._log(f"Querying for {k} similar documents", "debug")
        
        try:
            start_time = time.time()
            
            # Perform query
            query_params = {
                "query_embeddings": [query_embedding.tolist()],
                "n_results": k
            }
            
            if filter_metadata:
                query_params["where"] = filter_metadata
            
            results = self.collection.query(**query_params)
            
            query_time = time.time() - start_time
            
            # Process results
            similar_docs = []
            
            if results and 'ids' in results and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    doc_data = {
                        'id': results['ids'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else None,
                        'content': results['documents'][0][i] if 'documents' in results else None,
                        'metadata': results['metadatas'][0][i] if 'metadatas' in results else {}
                    }
                    similar_docs.append(doc_data)
            
            # Update statistics
            self.stats['queries_performed'] += 1
            self.stats['total_operation_time'] += query_time
            
            self._log(f"Query returned {len(similar_docs)} results in {query_time:.3f}s", "debug")
            
            return similar_docs
            
        except Exception as e:
            self._log(f"Error querying vector store: {e}", "error")
            self.stats['errors'] += 1
            raise VectorDatabaseError(f"Query failed: {e}")
    
    def delete_by_source_file(self, source_file: str) -> int:
        """
        Delete all chunks from a specific source file.
        
        Args:
            source_file: Source file path to delete chunks for
            
        Returns:
            Number of chunks deleted
        """
        if not isinstance(source_file, str):
            raise TypeError("Source file must be a string")
        
        if not source_file.strip():
            raise ValueError("Source file cannot be empty")
        
        self._log(f"Deleting chunks from source file: {source_file}")
        
        try:
            start_time = time.time()
            
            # Find documents from this source file
            results = self.collection.get(
                where={"source_file": source_file}
            )
            
            if not results or not results['ids']:
                self._log("No documents found for source file")
                return 0
            
            # Delete found documents
            delete_ids = results['ids']
            self.collection.delete(ids=delete_ids)
            
            operation_time = time.time() - start_time
            deleted_count = len(delete_ids)
            
            # Update statistics
            self.stats['documents_deleted'] += deleted_count
            self.stats['total_operation_time'] += operation_time
            
            # Keep this as INFO for processing summaries
            self._log(f"Deleted {deleted_count} chunks in {operation_time:.2f}s")
            
            return deleted_count
            
        except Exception as e:
            self._log(f"Error deleting chunks: {e}", "error")
            self.stats['errors'] += 1
            raise VectorDatabaseError(f"Failed to delete chunks: {e}")
    
    def rebuild_collection(self) -> bool:
        """
        Rebuild the entire collection (delete and recreate).
        
        Returns:
            True if successful
        """
        self._log("Rebuilding collection")
        
        try:
            start_time = time.time()
            
            # Delete existing collection
            if self.collection:
                try:
                    self.client.delete_collection(self.collection_name)
                    self._log("Deleted existing collection")
                except Exception as e:
                    self._log(f"Warning: Could not delete existing collection: {e}", "warning")
            
            # Recreate collection
            self._setup_collection()
            
            operation_time = time.time() - start_time
            
            self._log(f"Collection rebuilt in {operation_time:.2f}s")
            
            return True
            
        except Exception as e:
            self._log(f"Error rebuilding collection: {e}", "error")
            self.stats['errors'] += 1
            raise VectorDatabaseError(f"Failed to rebuild collection: {e}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            if not self.collection:
                return {"error": "Collection not initialized"}
            
            count = self.collection.count()
            
            # Get sample of documents for statistics
            sample_results = self.collection.get(limit=min(100, count))
            
            info = {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": str(self.persist_directory),
                "max_collection_size": self.max_collection_size,
                "sample_documents": len(sample_results['ids']) if sample_results else 0
            }
            
            # Add source file statistics
            if sample_results and 'metadatas' in sample_results:
                source_files = set()
                for metadata in sample_results['metadatas']:
                    if metadata and 'source_file' in metadata:
                        source_files.add(metadata['source_file'])
                info["unique_source_files_sample"] = len(source_files)
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.stats.copy()
        
        # Add collection info
        stats["collection_info"] = self.get_collection_info()
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self.stats = {
            'documents_added': 0,
            'documents_updated': 0,
            'documents_deleted': 0,
            'queries_performed': 0,
            'total_operation_time': 0.0,
            'errors': 0
        }
        self._log("Statistics reset")
    
    def clear_all_data(self) -> bool:
        """
        Clear all data from the database by removing the entire persist directory.
        This completely wipes the database and creates a fresh one.
        
        Returns:
            True if successful
        """
        self._log("Clearing all database data")
        
        try:
            start_time = time.time()
            
            # Close current connections
            self.client = None
            self.collection = None
            
            # Remove entire persist directory with retry logic
            if self.persist_directory.exists():
                try:
                    # Try to remove normally first
                    shutil.rmtree(self.persist_directory)
                    self._log(f"Removed persist directory: {self.persist_directory}")
                except Exception as e:
                    self._log(f"First removal attempt failed: {e}, trying force removal", "warning")
                    
                    # Force removal - set permissions and try again
                    try:
                        import stat
                        
                        # Walk through all files and set permissions
                        for root, dirs, files in os.walk(self.persist_directory):
                            for d in dirs:
                                os.chmod(os.path.join(root, d), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                            for f in files:
                                file_path = os.path.join(root, f)
                                os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)
                        
                        # Now try to remove again
                        shutil.rmtree(self.persist_directory)
                        self._log(f"Force removed persist directory: {self.persist_directory}")
                    except Exception as e2:
                        # If all else fails, just recreate the database without removing the directory
                        self._log(f"Could not remove directory: {e2}, clearing collections instead", "warning")
                        
                        # Reinitialize database and clear collections
                        self._setup_database()
                        
                        # Try to delete all collections
                        try:
                            collections = self.client.list_collections()
                            for collection in collections:
                                self.client.delete_collection(collection.name)
                                self._log(f"Deleted collection: {collection.name}")
                        except Exception as e3:
                            self._log(f"Error deleting collections: {e3}", "warning")
                        
                        # Create fresh collection
                        self._setup_collection()
                        return True
            
            # Recreate database
            self._setup_database()
            
            operation_time = time.time() - start_time
            
            # Reset statistics
            self.stats = {
                'documents_added': 0,
                'documents_updated': 0,
                'documents_deleted': 0,
                'queries_performed': 0,
                'total_operation_time': 0.0,
                'errors': 0
            }
            
            self._log(f"All database data cleared and recreated in {operation_time:.2f}s")
            
            return True
            
        except Exception as e:
            self._log(f"Error clearing database data: {e}", "error")
            self.stats['errors'] += 1
            raise VectorDatabaseError(f"Failed to clear database data: {e}")

    def close(self) -> None:
        """Close database connections and cleanup."""
        try:
            if self.client:
                # ChromaDB doesn't have explicit close method
                self.client = None
                self.collection = None
                self._log("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database: {e}")


class VectorDatabase:
    """
    High-level vector database interface with security controls.
    
    Security Features:
    - Comprehensive input validation
    - Safe database operations
    - Audit logging and monitoring
    - Error handling and recovery
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma",
        collection_name: str = "docuchat_embeddings",
        verbose: bool = False,
        rebuild: bool = False
    ):
        """
        Initialize the vector database.
        
        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the collection
            verbose: Enable verbose logging
            rebuild: Whether to rebuild the database
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.verbose = verbose
        
        # Initialize vector store
        self.vector_store = ChromaDBVectorStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
            verbose=verbose,
            verify_integrity=True
        )
        
        # Rebuild if requested
        if rebuild:
            self.rebuild()
        
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        logger.info(f"VectorDatabase initialized")
    
    def _log(self, message: str, level: str = "info") -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            if level == "debug":
                logger.debug(f"[VectorDatabase] {message}")
            elif level == "warning":
                logger.warning(f"[VectorDatabase] {message}")
            elif level == "error":
                logger.error(f"[VectorDatabase] {message}")
            else:
                logger.info(f"[VectorDatabase] {message}")
    
    def add_chunks(self, embedded_chunks: List) -> bool:
        """
        Add embedded chunks to the database.
        
        Args:
            embedded_chunks: List of EmbeddedChunk objects
            
        Returns:
            True if successful
        """
        self._log(f"Adding {len(embedded_chunks)} chunks to database")
        return self.vector_store.add_embedded_chunks(embedded_chunks)
    
    def search_similar(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            source_filter: Optional source file filter
            
        Returns:
            List of similar documents
        """
        filter_metadata = None
        if source_filter:
            filter_metadata = {"source_file": source_filter}
        
        return self.vector_store.query_similar(
            query_embedding=query_embedding,
            k=k,
            filter_metadata=filter_metadata
        )
    
    def remove_source_file(self, source_file: str) -> int:
        """
        Remove all chunks from a source file.
        
        Args:
            source_file: Source file to remove
            
        Returns:
            Number of chunks removed
        """
        self._log(f"Removing chunks from source file: {source_file}")
        return self.vector_store.delete_by_source_file(source_file)
    
    def rebuild(self) -> bool:
        """
        Rebuild the entire database.
        
        Returns:
            True if successful
        """
        self._log("Rebuilding vector database")
        return self.vector_store.rebuild_collection()
    
    def get_info(self) -> Dict[str, Any]:
        """Get database information."""
        return self.vector_store.get_collection_info()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.vector_store.get_statistics()
    
    def clear_database(self) -> bool:
        """
        Clear all database data by removing all entries from chroma folder.
        
        Returns:
            True if successful
        """
        self._log("Clearing all database data")
        return self.vector_store.clear_all_data()

    def close(self) -> None:
        """Close database connections."""
        self.vector_store.close()


def main():
    """Simple test function for the vector database module."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python vector_db.py <test_mode>")
        print("Test modes: info, rebuild")
        sys.exit(1)
    
    test_mode = sys.argv[1]
    
    try:
        # Create test database
        db = VectorDatabase(verbose=True)
        
        if test_mode == "info":
            # Get database info
            info = db.get_info()
            print(f"\nDatabase Info:")
            for key, value in info.items():
                print(f"  {key}: {value}")
            
            stats = db.get_statistics()
            print(f"\nStatistics:")
            for key, value in stats.items():
                if key != "collection_info":
                    print(f"  {key}: {value}")
        
        elif test_mode == "rebuild":
            # Rebuild database
            success = db.rebuild()
            print(f"Rebuild successful: {success}")
        
        else:
            print(f"Unknown test mode: {test_mode}")
            sys.exit(1)
        
        # Close database
        db.close()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()