"""
RAG Pipeline module for DocuChat RAG system.
Orchestrates the complete Retrieval-Augmented Generation workflow.
"""

import time
import logging
import hashlib
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from functools import lru_cache
import numpy as np

# Import existing components
from embeddings import EmbeddingGenerator, EmbeddedChunk
from vector_db import VectorDatabase
from llm_client import LLMClient, LLMResponse
from simple_timer import get_timer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """
    Data class representing the complete result of a RAG query.
    
    Contains the generated answer along with source attribution
    and processing metadata for audit trails.
    """
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    processing_time: float
    model_used: str
    embedding_model: str
    top_k_used: int
    context_length: int
    tokens_per_second: Optional[float]
    metadata: Dict[str, Any]
    
    @property
    def has_sources(self) -> bool:
        """Check if result has source documents."""
        return len(self.sources) > 0
    
    @property
    def source_files(self) -> List[str]:
        """Get list of unique source files."""
        files = set()
        for source in self.sources:
            if 'metadata' in source and 'source_file' in source['metadata']:
                files.add(source['metadata']['source_file'])
        return sorted(list(files))
    
    @property
    def confidence_score(self) -> float:
        """Calculate confidence score based on source relevance."""
        if not self.sources:
            return 0.0
        
        # Use average distance as inverse confidence (lower distance = higher confidence)
        distances = [s.get('distance', 1.0) for s in self.sources]
        avg_distance = sum(distances) / len(distances)
        
        # Convert distance to confidence (closer = more confident)
        confidence = max(0.0, min(1.0, 1.0 - avg_distance))
        return confidence


class RAGPipeline:
    """
    Complete RAG pipeline orchestrator.
    
    Integrates all DocuChat components to provide end-to-end
    Retrieval-Augmented Generation functionality with security
    controls and comprehensive error handling.
    """
    
    DEFAULT_TOP_K = 10
    DEFAULT_PROMPT_TEMPLATE = """Based on the following information, answer the question directly and conversationally:

{context}

Question: {question}
Answer:"""

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        vector_database: VectorDatabase,
        llm_client: LLMClient,
        verbose: bool = False,
        prompt_template: Optional[str] = None
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedding_generator: Initialized embedding generator
            vector_database: Initialized vector database
            llm_client: Initialized LLM client
            verbose: Enable verbose logging
            prompt_template: Custom prompt template for RAG
        """
        self.embedding_generator = embedding_generator
        self.vector_database = vector_database
        self.llm_client = llm_client
        self.verbose = verbose
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        
        # Validate components
        self._validate_components()
        
        # Query embedding cache (LRU with max 1000 entries)
        self._embedding_cache = {}
        self._cache_max_size = 1000
        self._cache_access_order = []
        
        # Statistics tracking
        self.stats = {
            'queries_processed': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_processing_time': 0.0,
            'total_context_length': 0,
            'total_sources_retrieved': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': []
        }
        
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        logger.info("RAGPipeline initialized successfully")
    
    def _log(self, message: str, level: str = "info") -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            if level == "debug":
                logger.debug(f"[RAGPipeline] {message}")
            elif level == "warning":
                logger.warning(f"[RAGPipeline] {message}")
            elif level == "error":
                logger.error(f"[RAGPipeline] {message}")
            else:
                logger.info(f"[RAGPipeline] {message}")
    
    def _validate_components(self) -> None:
        """Validate that all required components are properly initialized."""
        if not isinstance(self.embedding_generator, EmbeddingGenerator):
            raise TypeError("embedding_generator must be an EmbeddingGenerator instance")
        
        if not isinstance(self.vector_database, VectorDatabase):
            raise TypeError("vector_database must be a VectorDatabase instance")
        
        if not isinstance(self.llm_client, LLMClient):
            raise TypeError("llm_client must be an LLMClient instance")
        
        # Test component connectivity
        try:
            # Test LLM connection
            if not self.llm_client.is_available():
                raise RuntimeError("LLM client is not available")
            
            # Test vector database
            db_info = self.vector_database.get_info()
            if 'error' in db_info:
                raise RuntimeError(f"Vector database error: {db_info['error']}")
            
            self._log("All components validated successfully")
            
        except Exception as e:
            logger.error(f"Component validation failed: {e}")
            raise RuntimeError(f"Component validation failed: {e}")
    
    def _sanitize_question(self, question: str) -> str:
        """
        Sanitize user question for security.
        
        Args:
            question: Raw user question
            
        Returns:
            Sanitized question
        """
        if not isinstance(question, str):
            raise TypeError("Question must be a string")
        
        if not question.strip():
            raise ValueError("Question cannot be empty")
        
        # Security: Check question length
        if len(question) > 10000:  # 10KB limit
            self._log(f"Question exceeds length limit, truncating: {len(question)} chars", "warning")
            question = question[:10000]
        
        # Remove null bytes and normalize
        question = question.replace('\x00', '')
        question = question.strip()
        
        if not question:
            raise ValueError("Question is empty after sanitization")
        
        return question
    
    def _cache_embedding(self, question_hash: str, embedding: np.ndarray) -> None:
        """
        Cache question embedding with LRU eviction policy.
        
        Args:
            question_hash: MD5 hash of the question
            embedding: Question embedding vector
        """
        # Remove least recently used item if cache is full
        if len(self._embedding_cache) >= self._cache_max_size:
            lru_key = self._cache_access_order.pop(0)
            del self._embedding_cache[lru_key]
            self._log(f"Evicted LRU cache entry: {lru_key}", "debug")
        
        # Add new entry
        self._embedding_cache[question_hash] = embedding
        self._cache_access_order.append(question_hash)
        self._log(f"Cached embedding for question hash: {question_hash}", "debug")
    
    def _embed_question(self, question: str) -> np.ndarray:
        """
        Convert question to embedding vector with LRU cache.
        
        Args:
            question: User question
            
        Returns:
            Question embedding vector
        """
        self._log(f"Embedding question: {question[:100]}...", "debug")
        
        # Generate cache key from question hash
        question_hash = hashlib.md5(question.encode('utf-8')).hexdigest()
        
        # Check cache first
        if question_hash in self._embedding_cache:
            self._log("Using cached embedding", "debug")
            # Update LRU access order
            self._cache_access_order.remove(question_hash)
            self._cache_access_order.append(question_hash)
            return self._embedding_cache[question_hash]
        
        try:
            start_time = time.time()
            
            # Generate embedding using the same model as documents
            embedding = self.embedding_generator.model.encode([question])[0]
            
            embedding_time = time.time() - start_time
            self._log(f"Question embedded in {embedding_time:.3f}s (dim={len(embedding)})", "debug")
            
            # Cache the embedding with LRU eviction
            self._cache_embedding(question_hash, embedding)
            
            return embedding
            
        except Exception as e:
            self._log(f"Question embedding failed: {e}", "error")
            raise RuntimeError(f"Failed to embed question: {e}")
    
    def _retrieve_context(
        self,
        question_embedding: np.ndarray,
        top_k: int = DEFAULT_TOP_K
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context documents from vector database.
        
        Args:
            question_embedding: Embedded question vector
            top_k: Number of top documents to retrieve
            
        Returns:
            List of relevant documents with metadata
        """
        self._log(f"Retrieving top-{top_k} relevant documents", "debug")
        
        try:
            start_time = time.time()
            
            # Search for similar documents
            similar_docs = self.vector_database.search_similar(
                query_embedding=question_embedding,
                k=top_k
            )
            
            retrieval_time = time.time() - start_time
            self._log(f"Retrieved {len(similar_docs)} documents in {retrieval_time:.3f}s (requested top-{top_k})", "debug")
            
            return similar_docs
            
        except Exception as e:
            self._log(f"Context retrieval failed: {e}", "error")
            raise RuntimeError(f"Failed to retrieve context: {e}")
    
    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant information available."
        
        context_parts = []
        
        for doc in documents:
            content = doc.get('content', '')
            
            # Truncate very long content
            if len(content) > 2000:
                content = content[:2000] + "..."
            
            # Just add the content without document labels
            if content.strip():
                context_parts.append(content.strip())
        
        # Join with double line breaks for natural separation
        context = "\n\n".join(context_parts)
        
        # Security: Limit total context length
        if len(context) > 20000:  # 20KB limit
            self._log(f"Context exceeds length limit, truncating: {len(context)} chars", "warning")
            context = context[:20000] + "\n\n[Information truncated due to length...]"
        
        return context
    
    def _construct_prompt(self, question: str, context: str) -> str:
        """
        Construct the final prompt for the LLM.
        
        Args:
            question: User question
            context: Formatted context documents
            
        Returns:
            Complete prompt for LLM
        """
        try:
            prompt = self.prompt_template.format(
                context=context,
                question=question
            )
            
            self._log(f"Constructed prompt: {len(prompt)} characters", "debug")
            
            return prompt
            
        except Exception as e:
            self._log(f"Prompt construction failed: {e}", "error")
            raise RuntimeError(f"Failed to construct prompt: {e}")
    
    def _generate_answer(self, prompt: str) -> LLMResponse:
        """
        Generate answer using the LLM.
        
        Args:
            prompt: Complete prompt for LLM
            
        Returns:
            LLM response object
        """
        self._log("Generating answer with LLM", "debug")
        
        try:
            start_time = time.time()
            
            # Generate response
            response = self.llm_client.generate(prompt)
            
            generation_time = time.time() - start_time
            tokens_info = f", {response.eval_count} tokens" if response.eval_count else ""
            self._log(f"Answer generated in {generation_time:.2f}s{tokens_info}", "debug")
            
            return response
            
        except Exception as e:
            self._log(f"Answer generation failed: {e}", "error")
            raise RuntimeError(f"Failed to generate answer: {e}")
    
    def answer_question(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        include_sources: bool = True
    ) -> RAGResult:
        """
        Answer a question using the complete RAG pipeline.
        
        Args:
            question: User question to answer
            top_k: Number of top documents to retrieve for context
            include_sources: Whether to include source information
            
        Returns:
            RAGResult with answer and metadata
            
        Raises:
            ValueError: If question is invalid
            RuntimeError: If pipeline processing fails
        """
        # Validate inputs
        question = self._sanitize_question(question)
        
        if top_k <= 0 or top_k > 100:
            raise ValueError("top_k must be between 1 and 100")
        
        self._log(f"Processing question: {question[:100]}...", "debug")
        
        pipeline_start_time = time.time()
        
        try:
            # Update statistics
            self.stats['queries_processed'] += 1
            
            # Get performance timer
            timer = get_timer()
            
            # Step 1: Embed the question
            with timer.time_block("question_embedding"):
                question_embedding = self._embed_question(question)
            
            # Step 2: Retrieve relevant context
            with timer.time_block("vector_search"):
                retrieved_docs = self._retrieve_context(question_embedding, top_k)
            
            # Step 3: Format context
            with timer.time_block("context_preparation"):
                context = self._format_context(retrieved_docs)
            
            # Step 4: Construct prompt
            prompt = self._construct_prompt(question, context)
            
            # Step 5: Generate answer
            with timer.time_block("llm_generation"):
                llm_response = self._generate_answer(prompt)
            
            # Calculate total processing time
            total_time = time.time() - pipeline_start_time
            
            # Prepare sources for result
            sources = []
            if include_sources:
                sources = retrieved_docs
            
            # Create result object with performance timings
            timing_results = timer.get_results()
            result = RAGResult(
                question=question,
                answer=llm_response.response,
                sources=sources,
                processing_time=total_time,
                model_used=llm_response.model,
                embedding_model=self.embedding_generator.model_name,
                top_k_used=top_k,
                context_length=len(context),
                tokens_per_second=llm_response.tokens_per_second,
                metadata={
                    'llm_response_metadata': llm_response.metadata,
                    'prompt_length': len(prompt),
                    'documents_retrieved': len(retrieved_docs),
                    'generation_time': llm_response.metadata.get('generation_time', 0.0),
                    'performance_timings': timing_results.get_breakdown()
                }
            )
            
            # Update statistics
            self.stats['successful_queries'] += 1
            self.stats['total_processing_time'] += total_time
            self.stats['total_context_length'] += len(context)
            self.stats['total_sources_retrieved'] += len(retrieved_docs)
            
            self._log(f"Question answered successfully in {total_time:.2f}s", "debug")
            
            return result
            
        except Exception as e:
            # Update error statistics
            self.stats['failed_queries'] += 1
            self.stats['errors'].append(str(e))
            
            self._log(f"Question processing failed: {e}", "error")
            raise RuntimeError(f"RAG pipeline failed: {e}")
    
    def test_pipeline(self, test_question: str = "What is this document collection about?") -> bool:
        """
        Test the complete RAG pipeline with a simple question.
        
        Args:
            test_question: Question to test with
            
        Returns:
            True if test successful
        """
        try:
            self._log(f"Testing pipeline with question: {test_question}")
            
            result = self.answer_question(test_question, top_k=3)
            
            if result.answer and len(result.answer.strip()) > 0:
                self._log("Pipeline test successful")
                return True
            else:
                self._log("Pipeline test failed: empty answer", "warning")
                return False
                
        except Exception as e:
            self._log(f"Pipeline test failed: {e}", "error")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline processing statistics."""
        stats = self.stats.copy()
        
        # Add derived statistics
        if stats['queries_processed'] > 0:
            stats['success_rate'] = stats['successful_queries'] / stats['queries_processed']
            
            if stats['successful_queries'] > 0:
                stats['average_processing_time'] = stats['total_processing_time'] / stats['successful_queries']
                stats['average_context_length'] = stats['total_context_length'] / stats['successful_queries']
                stats['average_sources_per_query'] = stats['total_sources_retrieved'] / stats['successful_queries']
            else:
                stats['average_processing_time'] = 0.0
                stats['average_context_length'] = 0
                stats['average_sources_per_query'] = 0
        else:
            stats['success_rate'] = 0.0
            stats['average_processing_time'] = 0.0
            stats['average_context_length'] = 0
            stats['average_sources_per_query'] = 0
        
        # Add component info
        stats['component_info'] = {
            'embedding_model': self.embedding_generator.model_name,
            'llm_model': self.llm_client.model,
            'vector_db_info': self.vector_database.get_info()
        }
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset pipeline statistics."""
        self.stats = {
            'queries_processed': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_processing_time': 0.0,
            'total_context_length': 0,
            'total_sources_retrieved': 0,
            'errors': []
        }
        self._log("Pipeline statistics reset")
    
    def close(self) -> None:
        """Close all pipeline components."""
        try:
            self.vector_database.close()
            self.llm_client.close()
            self._log("Pipeline closed successfully")
        except Exception as e:
            logger.error(f"Error closing pipeline: {e}")


def main():
    """Test function for the RAG pipeline module."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python rag_pipeline.py <question>")
        print("Example: python rag_pipeline.py 'What are the main topics in the documents?'")
        sys.exit(1)
    
    question = sys.argv[1]
    
    try:
        # Initialize components (would normally be done by main application)
        from embeddings import EmbeddingGenerator
        from vector_db import VectorDatabase
        from llm_client import LLMClient
        
        print("Initializing RAG pipeline components...")
        
        embedding_generator = EmbeddingGenerator(verbose=True)
        vector_database = VectorDatabase(verbose=True)
        llm_client = LLMClient(verbose=True)
        
        # Create pipeline
        pipeline = RAGPipeline(
            embedding_generator=embedding_generator,
            vector_database=vector_database,
            llm_client=llm_client,
            verbose=True
        )
        
        # Test pipeline
        print(f"\nTesting pipeline with question: '{question}'")
        result = pipeline.answer_question(question, top_k=5)
        
        print(f"\n{'='*60}")
        print("RAG PIPELINE RESULT")
        print(f"{'='*60}")
        print(f"Question: {result.question}")
        print(f"\nAnswer:\n{result.answer}")
        print(f"\nProcessing time: {result.processing_time:.2f} seconds")
        print(f"Sources found: {len(result.sources)}")
        print(f"Confidence: {result.confidence_score:.2f}")
        
        if result.sources:
            print(f"\nSource files: {', '.join(result.source_files)}")
        
        # Show statistics
        stats = pipeline.get_statistics()
        print(f"\nPipeline Statistics:")
        for key, value in stats.items():
            if key not in ['component_info', 'errors']:
                print(f"  {key}: {value}")
        
        # Close pipeline
        pipeline.close()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()