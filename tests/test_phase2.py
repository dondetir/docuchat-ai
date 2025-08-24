#!/usr/bin/env python3
"""
Phase 2 End-to-End Validation Script
Comprehensive QA validation for DocuChat RAG System Phase 2

This script validates the complete Phase 2 pipeline:
- Document chunking quality
- Embedding generation accuracy  
- Vector database integrity
- Retrieval functionality
- Performance characteristics
"""

import sys
import os
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass

# Add src directory to Python path
test_dir = Path(__file__).parent
src_path = test_dir.parent / "src"
sys.path.insert(0, str(src_path))

# Import Phase 2 components
try:
    from chunker import DocumentChunker, TextChunk
    from embeddings import EmbeddingGenerator, EmbeddedChunk
    from vector_db import VectorDatabase
    from document_loader import DocumentLoader
    PHASE2_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL: Phase 2 components not available: {e}")
    PHASE2_AVAILABLE = False


@dataclass
class TestResult:
    """Test result container."""
    test_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = None
    execution_time: float = 0.0


class Phase2Validator:
    """Comprehensive Phase 2 validation suite."""
    
    def __init__(self, test_data_dir: str = "../test_data", verbose: bool = True):
        self.test_data_dir = Path(test_data_dir)
        self.verbose = verbose
        self.results: List[TestResult] = []
        self.db = None
        self.embedder = None
        self.chunker = None
        
    def log(self, message: str):
        """Log message if verbose mode enabled."""
        if self.verbose:
            print(message)
    
    def run_test(self, test_func, test_name: str) -> TestResult:
        """Run a single test and capture results."""
        self.log(f"\n🧪 Running: {test_name}")
        start_time = time.time()
        
        try:
            passed, message, details = test_func()
            execution_time = time.time() - start_time
            
            result = TestResult(
                test_name=test_name,
                passed=passed,
                message=message,
                details=details or {},
                execution_time=execution_time
            )
            
            status = "✅ PASS" if passed else "❌ FAIL"
            self.log(f"{status}: {message} ({execution_time:.2f}s)")
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult(
                test_name=test_name,
                passed=False,
                message=f"Test failed with exception: {str(e)}",
                details={"exception": str(e), "traceback": traceback.format_exc()},
                execution_time=execution_time
            )
            self.log(f"❌ FAIL: Exception in {test_name}: {e}")
            
        self.results.append(result)
        return result
    
    def test_phase2_dependencies(self) -> Tuple[bool, str, Dict]:
        """Test 1: Verify Phase 2 dependencies are available."""
        if not PHASE2_AVAILABLE:
            return False, "Phase 2 components not importable", {}
        
        # Test individual component imports
        missing_components = []
        try:
            from chunker import DocumentChunker
        except ImportError:
            missing_components.append("DocumentChunker")
            
        try:
            from embeddings import EmbeddingGenerator
        except ImportError:
            missing_components.append("EmbeddingGenerator")
            
        try:
            from vector_db import VectorDatabase
        except ImportError:
            missing_components.append("VectorDatabase")
        
        if missing_components:
            return False, f"Missing components: {missing_components}", {"missing": missing_components}
        
        return True, "All Phase 2 components available", {"components": ["DocumentChunker", "EmbeddingGenerator", "VectorDatabase"]}
    
    def test_component_initialization(self) -> Tuple[bool, str, Dict]:
        """Test 2: Verify all components initialize correctly."""
        details = {}
        
        try:
            # Initialize chunker
            self.chunker = DocumentChunker(chunk_size=500, chunk_overlap=100, verbose=False)
            details["chunker"] = "✅ Initialized"
            
            # Initialize embedding generator
            self.embedder = EmbeddingGenerator(verbose=False)
            details["embedder"] = "✅ Initialized"
            
            # Initialize vector database (using actual API)
            self.db = VectorDatabase(persist_directory="./test_chroma_db", verbose=False, rebuild=True)
            details["vector_db"] = "✅ Initialized"
            
        except Exception as e:
            return False, f"Component initialization failed: {e}", details
        
        return True, "All components initialized successfully", details
    
    def test_document_loading(self) -> Tuple[bool, str, Dict]:
        """Test 3: Verify document loading works correctly."""
        if not self.test_data_dir.exists():
            return False, f"Test data directory not found: {self.test_data_dir}", {}
        
        loader = DocumentLoader(verbose=False)
        documents = list(loader.load_documents_from_directory(str(self.test_data_dir)))
        
        if len(documents) == 0:
            return False, "No documents loaded from test directory", {"document_count": 0}
        
        # Verify document content
        details = {
            "documents_loaded": len(documents),
            "total_characters": sum(len(content) for _, content in documents),
            "files": [Path(filename).name for filename, _ in documents]
        }
        
        # Check for expected test files
        expected_files = {"bonsai_art.txt", "silk_road_history.txt", "stargazing_guide.md", 
                         "stoicism_philosophy.txt", "unusual_fruits.txt", "cryptography_basics.txt"}
        loaded_files = {Path(filename).name for filename, _ in documents}
        missing_files = expected_files - loaded_files
        
        if missing_files:
            return False, f"Missing expected test files: {missing_files}", details
        
        return True, f"Successfully loaded {len(documents)} documents", details
    
    def test_text_chunking(self) -> Tuple[bool, str, Dict]:
        """Test 4: Verify text chunking produces valid chunks."""
        test_text = "This is a test document. " * 50  # 1250 chars
        test_file = "chunking_test.txt"
        
        chunks = list(self.chunker.chunk_document(test_text, test_file))
        
        if len(chunks) == 0:
            return False, "No chunks generated from test text", {}
        
        details = {
            "chunks_created": len(chunks),
            "source_text_length": len(test_text),
            "chunk_sizes": [len(chunk.content) for chunk in chunks],
            "total_chunk_chars": sum(len(chunk.content) for chunk in chunks)
        }
        
        # Validate chunk properties
        failures = []
        
        for i, chunk in enumerate(chunks):
            # Check chunk has required fields
            if not hasattr(chunk, 'content') or not chunk.content:
                failures.append(f"Chunk {i}: missing or empty content")
            if not hasattr(chunk, 'chunk_id') or not chunk.chunk_id:
                failures.append(f"Chunk {i}: missing chunk_id")
            if chunk.chunk_index != i:
                failures.append(f"Chunk {i}: incorrect chunk_index {chunk.chunk_index}")
            # Note: source_file is preserved as input (could be full path), so just check it exists
            if not hasattr(chunk, 'source_file') or not chunk.source_file:
                failures.append(f"Chunk {i}: missing source_file")
            
            # Check chunk size constraints
            if len(chunk.content) > 600:  # Allow some tolerance
                failures.append(f"Chunk {i}: too large ({len(chunk.content)} chars)")
        
        # Check overlap between consecutive chunks
        if len(chunks) > 1:
            overlap_found = False
            for i in range(len(chunks) - 1):
                current_end = chunks[i].content[-50:]  # Last 50 chars
                next_start = chunks[i + 1].content[:50]  # First 50 chars
                
                # Simple overlap detection
                for j in range(10, min(50, len(current_end))):
                    if current_end[-j:] in next_start:
                        overlap_found = True
                        break
                if overlap_found:
                    break
            
            if not overlap_found:
                failures.append("No overlap detected between consecutive chunks")
        
        if failures:
            return False, f"Chunking validation failed: {failures[0]}", {**details, "failures": failures}
        
        return True, f"Generated {len(chunks)} valid chunks", details
    
    def test_embedding_generation(self) -> Tuple[bool, str, Dict]:
        """Test 5: Verify embedding generation produces valid embeddings."""
        # Create test chunks
        test_chunks = [
            TextChunk(
                content=f"Test content for embedding {i}",
                chunk_id=f"test_chunk_{i}",
                source_file="embedding_test.txt",
                chunk_index=i
            ) for i in range(3)
        ]
        
        embedded_chunks = self.embedder.embed_chunks(test_chunks)
        
        if len(embedded_chunks) != len(test_chunks):
            return False, f"Expected {len(test_chunks)} embeddings, got {len(embedded_chunks)}", {}
        
        details = {
            "chunks_embedded": len(embedded_chunks),
            "embedding_dimensions": [],
            "embedding_norms": [],
            "content_preserved": True
        }
        
        failures = []
        
        for i, embedded_chunk in enumerate(embedded_chunks):
            # Check embedding properties
            if not hasattr(embedded_chunk, 'embedding'):
                failures.append(f"Chunk {i}: missing embedding")
                continue
                
            embedding = embedded_chunk.embedding
            if not isinstance(embedding, np.ndarray):
                failures.append(f"Chunk {i}: embedding not numpy array")
                continue
            
            if embedding.ndim != 1:
                failures.append(f"Chunk {i}: embedding not 1-dimensional")
                continue
            
            if not np.all(np.isfinite(embedding)):
                failures.append(f"Chunk {i}: embedding contains non-finite values")
                continue
            
            details["embedding_dimensions"].append(len(embedding))
            details["embedding_norms"].append(float(np.linalg.norm(embedding)))
            
            # Check content preservation
            if embedded_chunk.content != test_chunks[i].content:
                details["content_preserved"] = False
                failures.append(f"Chunk {i}: content not preserved")
            
            # Check metadata properties
            if not hasattr(embedded_chunk, 'char_count'):
                failures.append(f"Chunk {i}: missing char_count property")
            elif embedded_chunk.char_count != len(embedded_chunk.content):
                failures.append(f"Chunk {i}: incorrect char_count")
                
            if not hasattr(embedded_chunk, 'word_count'):
                failures.append(f"Chunk {i}: missing word_count property")
        
        # Check embedding consistency
        if details["embedding_dimensions"]:
            unique_dims = set(details["embedding_dimensions"])
            if len(unique_dims) > 1:
                failures.append(f"Inconsistent embedding dimensions: {unique_dims}")
            
            expected_dim = 384  # all-MiniLM-L6-v2 dimension
            if list(unique_dims)[0] != expected_dim:
                failures.append(f"Unexpected embedding dimension: {list(unique_dims)[0]}, expected {expected_dim}")
        
        if failures:
            return False, f"Embedding validation failed: {failures[0]}", {**details, "failures": failures}
        
        return True, f"Generated {len(embedded_chunks)} valid embeddings", details
    
    def test_vector_database_storage(self) -> Tuple[bool, str, Dict]:
        """Test 6: Verify vector database can store and retrieve embeddings."""
        # Create test embedded chunks
        test_chunks = [
            TextChunk(
                content=f"Database test content {i}",
                chunk_id=f"db_test_{i}",
                source_file=f"db_test_{i}.txt",
                chunk_index=0
            ) for i in range(3)
        ]
        
        embedded_chunks = self.embedder.embed_chunks(test_chunks)
        
        # Store in database
        success = self.db.add_chunks(embedded_chunks)
        
        if not success:
            return False, "Failed to store chunks in database", {}
        
        # Verify storage
        stats = self.db.get_info()
        
        details = {
            "chunks_stored": stats.get("document_count", 0),
            "storage_success": success,
            "database_stats": stats
        }
        
        # Try to retrieve chunks
        try:
            collection = self.db.vector_store.client.get_collection("docuchat_embeddings")
            stored_results = collection.get(include=["documents", "metadatas", "embeddings"])
            
            details["retrieved_count"] = len(stored_results.get("documents", []))
            details["metadata_keys"] = list(stored_results.get("metadatas", [{}])[0].keys()) if stored_results.get("metadatas") else []
            
        except Exception as e:
            return False, f"Failed to retrieve stored chunks: {e}", details
        
        # Validate required metadata fields
        if stored_results.get("metadatas"):
            required_fields = {"source_file", "chunk_index", "char_count", "word_count", "content_hash"}
            actual_fields = set(stored_results["metadatas"][0].keys())
            missing_fields = required_fields - actual_fields
            
            if missing_fields:
                return False, f"Missing required metadata fields: {missing_fields}", details
        
        return True, f"Successfully stored and retrieved {len(embedded_chunks)} chunks", details
    
    def test_retrieval_functionality(self) -> Tuple[bool, str, Dict]:
        """Test 7: Verify retrieval returns relevant results."""
        # Load real documents from test_data directory
        loader = DocumentLoader(verbose=False)
        real_documents = list(loader.load_documents_from_directory(str(self.test_data_dir)))
        test_documents = [(content, Path(filename).name) for filename, content in real_documents]
        
        if len(test_documents) < 5:
            return False, f"Insufficient test documents loaded: {len(test_documents)}", {}
        
        # Create test chunks using DocumentChunker to ensure proper format
        test_chunks = []
        for i, (content, filename) in enumerate(test_documents):
            # Use the actual chunker to create properly formatted chunks
            chunks = list(self.chunker.chunk_document(content, filename))
            if chunks:
                test_chunks.extend(chunks)
        
        embedded_chunks = self.embedder.embed_chunks(test_chunks)
        self.db.add_chunks(embedded_chunks)
        
        # Test retrieval with relevant queries
        test_queries = [
            ("Japanese bonsai art ornamental dwarfed trees containers horticulture", "bonsai_art.txt"),
            ("ancient silk road trade routes East West cultural economic interactions", "silk_road_history.txt"),
            ("stargazing astronomy constellations Ursa Major Orion dark sky guide", "stargazing_guide.md"),
            ("stoic philosophy Hellenistic ethics Zeno Citium eudaimonia flourishing", "stoicism_philosophy.txt"),
            ("unusual exotic fruits durian rambutan kiwano jabuticaba Buddha hand", "unusual_fruits.txt"),
            ("cryptography encryption decryption secure communication ciphertext keys", "cryptography_basics.txt")
        ]
        
        details = {
            "queries_tested": len(test_queries),
            "correct_retrievals": 0,
            "retrieval_results": []
        }
        
        failures = []
        
        for query_text, expected_file in test_queries:
            # Create query embedding using chunker
            query_chunks = list(self.chunker.chunk_document(query_text, "query.txt"))
            if not query_chunks:
                failures.append(f"Query '{query_text}' failed to generate chunks")
                continue
            query_embedded = self.embedder.embed_chunk(query_chunks[0])
            
            # Search database using the actual VectorDatabase API
            try:
                results = self.db.search_similar(
                    query_embedding=query_embedded.embedding,
                    k=1
                )
                
                if results:
                    top_result = results[0]
                    top_result_file = Path(top_result["metadata"]["source_file"]).name
                    distance = top_result.get("distance", 0.0)
                    
                    result_info = {
                        "query": query_text,
                        "expected": expected_file,
                        "actual": top_result_file,
                        "distance": distance,
                        "correct": top_result_file == expected_file
                    }
                    
                    details["retrieval_results"].append(result_info)
                    
                    if result_info["correct"]:
                        details["correct_retrievals"] += 1
                    else:
                        failures.append(f"Query '{query_text}' returned {top_result_file}, expected {expected_file}")
                
                else:
                    failures.append(f"Query '{query_text}' returned no results")
                    
            except Exception as e:
                failures.append(f"Query '{query_text}' failed with error: {e}")
        
        accuracy = details["correct_retrievals"] / details["queries_tested"] if details["queries_tested"] > 0 else 0
        details["accuracy"] = accuracy
        
        if accuracy < 0.5:  # At least 50% accuracy required
            return False, f"Low retrieval accuracy: {accuracy:.1%}", details
        
        if failures:
            return False, f"Retrieval issues: {failures[0]}", details
        
        return True, f"Retrieval accuracy: {accuracy:.1%} ({details['correct_retrievals']}/{details['queries_tested']})", details
    
    def test_end_to_end_pipeline(self) -> Tuple[bool, str, Dict]:
        """Test 8: Verify complete end-to-end pipeline."""
        if not self.test_data_dir.exists():
            return False, "Test data directory not found", {}
        
        # Initialize fresh database (rebuild via constructor)
        self.db = VectorDatabase(persist_directory="./test_chroma_db", verbose=False, rebuild=True)
        
        # Run complete pipeline
        loader = DocumentLoader(verbose=False)
        documents = list(loader.load_documents_from_directory(str(self.test_data_dir)))
        
        if not documents:
            return False, "No documents loaded for end-to-end test", {}
        
        pipeline_stats = {
            "documents_loaded": len(documents),
            "total_chunks": 0,
            "total_embeddings": 0,
            "stored_chunks": 0,
            "processing_time": 0
        }
        
        start_time = time.time()
        
        try:
            for filename, content in documents:
                # Chunk document
                chunks = list(self.chunker.chunk_document(content, filename))
                pipeline_stats["total_chunks"] += len(chunks)
                
                if chunks:
                    # Generate embeddings
                    embedded_chunks = self.embedder.embed_chunks(chunks)
                    pipeline_stats["total_embeddings"] += len(embedded_chunks)
                    
                    # Store in database
                    if self.db.add_chunks(embedded_chunks):
                        pipeline_stats["stored_chunks"] += len(embedded_chunks)
            
            pipeline_stats["processing_time"] = time.time() - start_time
            
            # Verify final database state
            final_stats = self.db.get_info()
            pipeline_stats["final_db_count"] = final_stats.get("document_count", 0)
            
            # Check consistency
            if pipeline_stats["stored_chunks"] != pipeline_stats["final_db_count"]:
                return False, f"Storage inconsistency: stored {pipeline_stats['stored_chunks']}, database has {pipeline_stats['final_db_count']}", pipeline_stats
            
            if pipeline_stats["total_chunks"] == 0:
                return False, "No chunks generated in pipeline", pipeline_stats
            
            if pipeline_stats["total_embeddings"] == 0:
                return False, "No embeddings generated in pipeline", pipeline_stats
                
        except Exception as e:
            pipeline_stats["processing_time"] = time.time() - start_time
            return False, f"Pipeline failed: {e}", pipeline_stats
        
        return True, f"Pipeline processed {pipeline_stats['documents_loaded']} docs → {pipeline_stats['total_chunks']} chunks → {pipeline_stats['stored_chunks']} stored", pipeline_stats
    
    def test_performance_characteristics(self) -> Tuple[bool, str, Dict]:
        """Test 9: Verify performance meets acceptable thresholds."""
        # Test with controlled content
        test_content = "This is a performance test sentence. " * 100  # ~3700 chars
        test_file = "performance_test.txt"
        
        performance_metrics = {}
        
        # Test chunking performance
        start_time = time.time()
        chunks = list(self.chunker.chunk_document(test_content, test_file))
        chunking_time = time.time() - start_time
        
        performance_metrics["chunking"] = {
            "time": chunking_time,
            "chars_per_second": len(test_content) / chunking_time if chunking_time > 0 else float('inf'),
            "chunks_generated": len(chunks)
        }
        
        # Test embedding performance
        start_time = time.time()
        embedded_chunks = self.embedder.embed_chunks(chunks)
        embedding_time = time.time() - start_time
        
        performance_metrics["embedding"] = {
            "time": embedding_time,
            "chunks_per_second": len(chunks) / embedding_time if embedding_time > 0 else float('inf'),
            "embeddings_generated": len(embedded_chunks)
        }
        
        # Test storage performance
        start_time = time.time()
        storage_success = self.db.add_chunks(embedded_chunks)
        storage_time = time.time() - start_time
        
        performance_metrics["storage"] = {
            "time": storage_time,
            "chunks_per_second": len(embedded_chunks) / storage_time if storage_time > 0 else float('inf'),
            "storage_success": storage_success
        }
        
        # Performance thresholds
        failures = []
        
        if performance_metrics["chunking"]["chars_per_second"] < 1000:  # At least 1000 chars/sec
            failures.append(f"Slow chunking: {performance_metrics['chunking']['chars_per_second']:.1f} chars/sec")
        
        if performance_metrics["embedding"]["chunks_per_second"] < 1:  # At least 1 chunk/sec
            failures.append(f"Slow embedding: {performance_metrics['embedding']['chunks_per_second']:.1f} chunks/sec")
        
        if performance_metrics["storage"]["chunks_per_second"] < 5:  # At least 5 chunks/sec
            failures.append(f"Slow storage: {performance_metrics['storage']['chunks_per_second']:.1f} chunks/sec")
        
        total_time = chunking_time + embedding_time + storage_time
        performance_metrics["total_pipeline_time"] = total_time
        
        if total_time > 10.0:  # Total pipeline should complete within 10 seconds
            failures.append(f"Slow total pipeline: {total_time:.2f} seconds")
        
        if failures:
            return False, f"Performance issues: {failures[0]}", performance_metrics
        
        return True, f"Performance acceptable (total: {total_time:.2f}s)", performance_metrics
    
    def cleanup(self):
        """Clean up test resources."""
        try:
            if self.db:
                # Close database connections
                self.db.close()
                # Clean up test database
                test_db_path = Path("./test_chroma_db")
                if test_db_path.exists():
                    import shutil
                    shutil.rmtree(test_db_path, ignore_errors=True)
        except Exception as e:
            self.log(f"⚠️  Cleanup warning: {e}")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        self.log("🚀 Starting Phase 2 Comprehensive Validation")
        self.log("=" * 60)
        
        # Define test suite
        tests = [
            (self.test_phase2_dependencies, "Phase 2 Dependencies"),
            (self.test_component_initialization, "Component Initialization"),
            (self.test_document_loading, "Document Loading"),
            (self.test_text_chunking, "Text Chunking"),
            (self.test_embedding_generation, "Embedding Generation"),
            (self.test_vector_database_storage, "Vector Database Storage"),
            (self.test_retrieval_functionality, "Retrieval Functionality"),
            (self.test_end_to_end_pipeline, "End-to-End Pipeline"),
            (self.test_performance_characteristics, "Performance Characteristics")
        ]
        
        # Run all tests
        for test_func, test_name in tests:
            self.run_test(test_func, test_name)
        
        # Generate summary
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        summary = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": success_rate,
            "overall_status": "PASS" if success_rate >= 80 else "FAIL",  # 80% pass rate required
            "results": self.results
        }
        
        # Print detailed summary
        self.log("\n" + "=" * 60)
        self.log("📊 PHASE 2 VALIDATION SUMMARY")
        self.log("=" * 60)
        
        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            self.log(f"{status} {result.test_name}: {result.message}")
            if not result.passed and result.details.get("failures"):
                for failure in result.details["failures"][:3]:  # Show first 3 failures
                    self.log(f"    └─ {failure}")
        
        self.log(f"\n📈 Overall Results:")
        self.log(f"   Tests Run: {total_tests}")
        self.log(f"   Passed: {passed_tests}")
        self.log(f"   Failed: {failed_tests}")
        self.log(f"   Success Rate: {success_rate:.1f}%")
        
        overall_status = "✅ OVERALL PASS" if success_rate >= 80 else "❌ OVERALL FAIL"
        self.log(f"\n🎯 {overall_status}")
        
        if success_rate >= 80:
            self.log("✅ Phase 2 is ready for Phase 3!")
        else:
            self.log("❌ Phase 2 requires fixes before proceeding to Phase 3")
            self.log("\n🔧 Failed Tests:")
            for result in self.results:
                if not result.passed:
                    self.log(f"   • {result.test_name}: {result.message}")
        
        # Cleanup
        self.cleanup()
        
        return summary


def main():
    """Main entry point for validation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 2 Comprehensive Validation")
    parser.add_argument("--test-data", default="test_data", help="Test data directory")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    
    args = parser.parse_args()
    
    # Run validation
    validator = Phase2Validator(
        test_data_dir=args.test_data,
        verbose=not args.quiet
    )
    
    summary = validator.run_all_tests()
    
    # Output results
    if args.json:
        import json
        # Convert results to JSON-serializable format
        json_summary = {
            "total_tests": summary["total_tests"],
            "passed": summary["passed"],
            "failed": summary["failed"],
            "success_rate": summary["success_rate"],
            "overall_status": summary["overall_status"],
            "test_results": [
                {
                    "test_name": r.test_name,
                    "passed": r.passed,
                    "message": r.message,
                    "execution_time": r.execution_time
                }
                for r in summary["results"]
            ]
        }
        print(json.dumps(json_summary, indent=2))
    
    # Return appropriate exit code
    exit_code = 0 if summary["success_rate"] >= 80 else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()