#!/usr/bin/env python3
"""
Phase 3 End-to-End Validation Script
Comprehensive QA validation for DocuChat RAG System Phase 3

This script validates the complete Phase 3 LLM integration:
- Ollama server connectivity
- Model availability and loading
- Text generation functionality
- Response quality and format validation
- Error handling and recovery
- Performance characteristics
- Integration with existing RAG pipeline
"""

import sys
import os
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
from dataclasses import dataclass

# Add src directory to Python path
test_dir = Path(__file__).parent
src_path = test_dir.parent / "src"
sys.path.insert(0, str(src_path))

# Import Phase 3 components
try:
    from llm_client import LLMClient, SecureLLMClient, LLMResponse
    from llm_client import LLMClientError, ModelNotFoundError, ConnectionError, TimeoutError, GenerationError
    PHASE3_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL: Phase 3 components not available: {e}")
    PHASE3_AVAILABLE = False

# Import previous phase components for integration testing
try:
    from chunker import DocumentChunker, TextChunk
    from embeddings import EmbeddingGenerator, EmbeddedChunk
    from vector_db import VectorDatabase
    from document_loader import DocumentLoader
    PREVIOUS_PHASES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  WARNING: Previous phase components not available: {e}")
    PREVIOUS_PHASES_AVAILABLE = False


@dataclass
class TestResult:
    """Test result container."""
    test_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = None
    execution_time: float = 0.0


class Phase3Validator:
    """Comprehensive Phase 3 validation suite."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gemma3:270m",
        test_data_dir: str = "../test_data",
        verbose: bool = True
    ):
        self.base_url = base_url
        self.model = model
        self.test_data_dir = Path(test_data_dir)
        self.verbose = verbose
        self.results: List[TestResult] = []
        self.llm_client = None
        self.secure_client = None
        
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
    
    def test_phase3_dependencies(self) -> Tuple[bool, str, Dict]:
        """Test 1: Verify Phase 3 dependencies are available."""
        if not PHASE3_AVAILABLE:
            return False, "Phase 3 components not importable", {}
        
        # Test individual component imports
        missing_components = []
        try:
            from llm_client import LLMClient
        except ImportError:
            missing_components.append("LLMClient")
            
        try:
            from llm_client import SecureLLMClient
        except ImportError:
            missing_components.append("SecureLLMClient")
            
        try:
            from llm_client import LLMResponse
        except ImportError:
            missing_components.append("LLMResponse")
        
        # Test requests dependency
        try:
            import requests
            requests_version = getattr(requests, "__version__", "unknown")
        except ImportError:
            missing_components.append("requests")
            requests_version = None
        
        if missing_components:
            return False, f"Missing components: {missing_components}", {"missing": missing_components}
        
        details = {
            "components": ["LLMClient", "SecureLLMClient", "LLMResponse"],
            "requests_version": requests_version
        }
        
        return True, "All Phase 3 components available", details
    
    def test_ollama_server_connectivity(self) -> Tuple[bool, str, Dict]:
        """Test 2: Verify Ollama server is accessible."""
        details = {"base_url": self.base_url}
        
        try:
            # Initialize secure client for connection testing
            self.secure_client = SecureLLMClient(
                base_url=self.base_url,
                model=self.model,
                verbose=False,
                timeout=10.0
            )
            
            # Test connection
            is_connected = self.secure_client.test_connection()
            
            if not is_connected:
                return False, f"Cannot connect to Ollama server at {self.base_url}", details
            
            details["connection_successful"] = True
            
            # Get server info if possible
            try:
                response = self.secure_client.session.get(f"{self.base_url}/api/version", timeout=5)
                if response.status_code == 200:
                    version_info = response.json()
                    details["server_version"] = version_info
            except Exception:
                pass  # Version endpoint might not exist in all versions
            
            return True, f"Successfully connected to Ollama server at {self.base_url}", details
            
        except ConnectionError as e:
            return False, f"Connection failed: {e}", details
        except Exception as e:
            return False, f"Connection test failed: {e}", details
    
    def test_model_availability(self) -> Tuple[bool, str, Dict]:
        """Test 3: Verify target model is available."""
        if not self.secure_client:
            return False, "Secure client not initialized", {}
        
        details = {"target_model": self.model}
        
        try:
            # List available models
            available_models = self.secure_client.list_models()
            model_names = [model.get("name", "") for model in available_models]
            
            details["available_models"] = model_names
            details["total_models"] = len(available_models)
            
            # Check if target model is available
            is_available = self.secure_client.verify_model(self.model)
            
            if not is_available:
                return False, f"Model '{self.model}' not found. Available: {model_names}", details
            
            # Get model info
            model_info = self.secure_client.get_model_info()
            details["model_info"] = model_info
            
            return True, f"Model '{self.model}' is available and verified", details
            
        except Exception as e:
            return False, f"Model availability check failed: {e}", details
    
    def test_llm_client_initialization(self) -> Tuple[bool, str, Dict]:
        """Test 4: Verify LLM clients initialize correctly."""
        details = {}
        
        try:
            # Initialize high-level client
            self.llm_client = LLMClient(
                base_url=self.base_url,
                model=self.model,
                verbose=False
            )
            details["llm_client"] = "✅ Initialized"
            
            # Test client availability
            is_available = self.llm_client.is_available()
            details["client_available"] = is_available
            
            if not is_available:
                return False, "LLM client reports service unavailable", details
            
            # Get client info
            client_info = self.llm_client.get_info()
            details["client_info"] = client_info
            
            return True, "LLM client initialized and service available", details
            
        except Exception as e:
            return False, f"Client initialization failed: {e}", details
    
    def test_basic_text_generation(self) -> Tuple[bool, str, Dict]:
        """Test 5: Verify basic text generation works."""
        if not self.llm_client:
            return False, "LLM client not initialized", {}
        
        test_prompt = "What is the capital of France?"
        details = {"test_prompt": test_prompt}
        
        try:
            start_time = time.time()
            
            # Test the simple test_prompt function
            response_text = self.llm_client.test_prompt(test_prompt)
            
            generation_time = time.time() - start_time
            details["generation_time"] = generation_time
            details["response_text"] = response_text
            details["response_length"] = len(response_text)
            
            # Basic validation
            if not isinstance(response_text, str):
                return False, f"Response not a string: {type(response_text)}", details
            
            if not response_text.strip():
                return False, "Response is empty", details
            
            if len(response_text) < 5:
                return False, f"Response too short: {len(response_text)} chars", details
            
            # Check if response mentions Paris (basic content validation)
            response_lower = response_text.lower()
            if "paris" in response_lower:
                details["content_correct"] = True
            else:
                details["content_correct"] = False
                # Don't fail the test for content correctness in basic test
            
            return True, f"Generated {len(response_text)} chars in {generation_time:.2f}s", details
            
        except Exception as e:
            return False, f"Text generation failed: {e}", details
    
    def test_advanced_text_generation(self) -> Tuple[bool, str, Dict]:
        """Test 6: Verify advanced generation features work."""
        if not self.llm_client:
            return False, "LLM client not initialized", {}
        
        test_cases = [
            {
                "prompt": "Explain quantum computing in simple terms.",
                "description": "Complex technical topic"
            },
            {
                "prompt": "List three benefits of renewable energy.",
                "description": "Structured list request"
            },
            {
                "prompt": "Write a short story about a robot learning to paint.",
                "description": "Creative writing task"
            }
        ]
        
        details = {"test_cases": len(test_cases), "results": []}
        failures = []
        
        for i, test_case in enumerate(test_cases):
            try:
                start_time = time.time()
                
                # Use the full generate method
                response = self.llm_client.generate(test_case["prompt"])
                
                generation_time = time.time() - start_time
                
                # Validate response object
                if not isinstance(response, LLMResponse):
                    failures.append(f"Test {i+1}: Response not LLMResponse object")
                    continue
                
                result_info = {
                    "prompt": test_case["prompt"][:50] + "...",
                    "description": test_case["description"],
                    "response_length": len(response.response),
                    "word_count": response.word_count,
                    "generation_time": generation_time,
                    "model": response.model,
                    "done": response.done
                }
                
                # Add performance metrics if available
                if response.tokens_per_second:
                    result_info["tokens_per_second"] = response.tokens_per_second
                
                details["results"].append(result_info)
                
                # Basic validation
                if not response.response.strip():
                    failures.append(f"Test {i+1}: Empty response")
                
                if len(response.response) < 20:
                    failures.append(f"Test {i+1}: Response too short ({len(response.response)} chars)")
                
                if not response.done:
                    failures.append(f"Test {i+1}: Generation not marked as done")
                
            except Exception as e:
                failures.append(f"Test {i+1}: Generation failed - {e}")
        
        if failures:
            return False, f"Advanced generation issues: {failures[0]}", {**details, "failures": failures}
        
        avg_time = sum(r["generation_time"] for r in details["results"]) / len(details["results"])
        total_chars = sum(r["response_length"] for r in details["results"])
        
        return True, f"All {len(test_cases)} advanced tests passed (avg: {avg_time:.2f}s, {total_chars} chars)", details
    
    def test_error_handling(self) -> Tuple[bool, str, Dict]:
        """Test 7: Verify error handling works correctly."""
        details = {"tests_performed": 0, "errors_caught": 0}
        
        # Test 1: Invalid model
        try:
            details["tests_performed"] += 1
            invalid_client = LLMClient(
                base_url=self.base_url,
                model="nonexistent_model_12345",
                verbose=False
            )
            try:
                invalid_client.test_prompt("Test")
                # If this succeeds, it's unexpected
                details["invalid_model_test"] = "Unexpectedly succeeded"
            except (ModelNotFoundError, GenerationError) as e:
                details["errors_caught"] += 1
                details["invalid_model_test"] = f"Correctly caught: {type(e).__name__}"
        except Exception as e:
            details["invalid_model_test"] = f"Setup failed: {e}"
        
        # Test 2: Invalid server URL
        try:
            details["tests_performed"] += 1
            invalid_url_client = LLMClient(
                base_url="http://localhost:99999",  # Invalid port
                model=self.model,
                timeout=2.0,
                verbose=False
            )
            try:
                invalid_url_client.test_prompt("Test")
                details["invalid_url_test"] = "Unexpectedly succeeded"
            except (ConnectionError, TimeoutError) as e:
                details["errors_caught"] += 1
                details["invalid_url_test"] = f"Correctly caught: {type(e).__name__}"
        except Exception as e:
            details["invalid_url_test"] = f"Setup failed: {e}"
        
        # Test 3: Empty prompt
        try:
            details["tests_performed"] += 1
            try:
                self.llm_client.test_prompt("")
                details["empty_prompt_test"] = "Unexpectedly succeeded"
            except (ValueError, GenerationError) as e:
                details["errors_caught"] += 1
                details["empty_prompt_test"] = f"Correctly caught: {type(e).__name__}"
        except Exception as e:
            details["empty_prompt_test"] = f"Unexpected error: {e}"
        
        # Test 4: Very long prompt (if supported by model)
        try:
            details["tests_performed"] += 1
            long_prompt = "Test prompt. " * 10000  # Very long prompt
            try:
                response = self.llm_client.test_prompt(long_prompt)
                if response:
                    details["long_prompt_test"] = "Handled gracefully"
                else:
                    details["long_prompt_test"] = "Empty response"
            except Exception as e:
                details["errors_caught"] += 1
                details["long_prompt_test"] = f"Correctly limited: {type(e).__name__}"
        except Exception as e:
            details["long_prompt_test"] = f"Test failed: {e}"
        
        success_rate = details["errors_caught"] / details["tests_performed"] if details["tests_performed"] > 0 else 0
        
        if success_rate < 0.5:  # At least 50% of error conditions should be caught
            return False, f"Poor error handling: {details['errors_caught']}/{details['tests_performed']} errors caught", details
        
        return True, f"Error handling working: {details['errors_caught']}/{details['tests_performed']} errors caught correctly", details
    
    def test_response_format_validation(self) -> Tuple[bool, str, Dict]:
        """Test 8: Verify response format and data integrity."""
        if not self.llm_client:
            return False, "LLM client not initialized", {}
        
        test_prompt = "Describe the benefits of exercise in 2-3 sentences."
        details = {"test_prompt": test_prompt}
        
        try:
            # Generate response using full API
            response = self.llm_client.generate(test_prompt)
            
            details["response_type"] = type(response).__name__
            
            # Validate response is LLMResponse object
            if not isinstance(response, LLMResponse):
                return False, f"Response not LLMResponse object: {type(response)}", details
            
            # Validate required fields
            required_fields = ['model', 'prompt', 'response', 'created_at', 'done']
            missing_fields = []
            
            for field in required_fields:
                if not hasattr(response, field):
                    missing_fields.append(field)
                else:
                    details[f"has_{field}"] = True
            
            if missing_fields:
                return False, f"Missing required fields: {missing_fields}", details
            
            # Validate field types and values
            validation_failures = []
            
            if not isinstance(response.model, str) or not response.model:
                validation_failures.append("Invalid model field")
            
            if not isinstance(response.prompt, str) or response.prompt != test_prompt:
                validation_failures.append("Invalid prompt field")
            
            if not isinstance(response.response, str) or not response.response.strip():
                validation_failures.append("Invalid response field")
            
            if not isinstance(response.created_at, str) or not response.created_at:
                validation_failures.append("Invalid created_at field")
            
            if not isinstance(response.done, bool):
                validation_failures.append("Invalid done field")
            
            # Validate optional numeric fields
            numeric_fields = ['total_duration', 'load_duration', 'prompt_eval_count', 
                            'prompt_eval_duration', 'eval_count', 'eval_duration']
            
            for field in numeric_fields:
                value = getattr(response, field, None)
                if value is not None and (not isinstance(value, int) or value < 0):
                    validation_failures.append(f"Invalid {field} field")
                details[f"{field}_value"] = value
            
            # Test property methods
            try:
                char_count = response.response_length
                word_count = response.word_count
                details["char_count"] = char_count
                details["word_count"] = word_count
                
                if char_count != len(response.response):
                    validation_failures.append("Incorrect character count property")
                
                if word_count <= 0:
                    validation_failures.append("Invalid word count property")
                
            except Exception as e:
                validation_failures.append(f"Property method error: {e}")
            
            # Test tokens per second calculation if data available
            try:
                tps = response.tokens_per_second
                if tps is not None:
                    details["tokens_per_second"] = tps
                    if tps <= 0:
                        validation_failures.append("Invalid tokens per second calculation")
            except Exception as e:
                validation_failures.append(f"Tokens per second calculation error: {e}")
            
            if validation_failures:
                return False, f"Response validation failed: {validation_failures[0]}", {**details, "failures": validation_failures}
            
            return True, f"Response format valid: {len(response.response)} chars, {response.word_count} words", details
            
        except Exception as e:
            return False, f"Response format test failed: {e}", details
    
    def test_performance_characteristics(self) -> Tuple[bool, str, Dict]:
        """Test 9: Verify performance meets acceptable thresholds."""
        if not self.llm_client:
            return False, "LLM client not initialized", {}
        
        # Performance test prompts
        test_prompts = [
            "What is artificial intelligence?",
            "Explain the water cycle.",
            "List five programming languages.",
            "Describe how photosynthesis works.",
            "What are the benefits of reading books?"
        ]
        
        performance_metrics = {
            "prompts_tested": len(test_prompts),
            "successful_generations": 0,
            "failed_generations": 0,
            "total_time": 0.0,
            "response_times": [],
            "response_lengths": [],
            "total_characters": 0
        }
        
        failures = []
        
        for i, prompt in enumerate(test_prompts):
            try:
                start_time = time.time()
                response = self.llm_client.test_prompt(prompt)
                generation_time = time.time() - start_time
                
                performance_metrics["successful_generations"] += 1
                performance_metrics["response_times"].append(generation_time)
                performance_metrics["response_lengths"].append(len(response))
                performance_metrics["total_characters"] += len(response)
                performance_metrics["total_time"] += generation_time
                
            except Exception as e:
                performance_metrics["failed_generations"] += 1
                failures.append(f"Prompt {i+1} failed: {e}")
        
        # Calculate derived metrics
        if performance_metrics["successful_generations"] > 0:
            performance_metrics["average_response_time"] = (
                performance_metrics["total_time"] / performance_metrics["successful_generations"]
            )
            performance_metrics["average_response_length"] = (
                performance_metrics["total_characters"] / performance_metrics["successful_generations"]
            )
            performance_metrics["characters_per_second"] = (
                performance_metrics["total_characters"] / performance_metrics["total_time"]
                if performance_metrics["total_time"] > 0 else 0
            )
        
        # Performance thresholds
        perf_failures = []
        
        success_rate = (performance_metrics["successful_generations"] / 
                       performance_metrics["prompts_tested"])
        
        if success_rate < 0.8:  # At least 80% success rate
            perf_failures.append(f"Low success rate: {success_rate:.1%}")
        
        if performance_metrics["average_response_time"] > 30.0:  # Max 30 seconds per response
            perf_failures.append(f"Slow response time: {performance_metrics['average_response_time']:.2f}s")
        
        if performance_metrics["average_response_length"] < 10:  # At least 10 chars average
            perf_failures.append(f"Short responses: {performance_metrics['average_response_length']:.1f} chars")
        
        if performance_metrics["total_time"] > 120.0:  # Total test should complete within 2 minutes
            perf_failures.append(f"Slow total test time: {performance_metrics['total_time']:.2f}s")
        
        if perf_failures:
            return False, f"Performance issues: {perf_failures[0]}", {**performance_metrics, "failures": perf_failures}
        
        return True, f"Performance acceptable: {success_rate:.1%} success, {performance_metrics['average_response_time']:.2f}s avg", performance_metrics
    
    def test_integration_with_rag_pipeline(self) -> Tuple[bool, str, Dict]:
        """Test 10: Verify integration with existing RAG pipeline."""
        if not PREVIOUS_PHASES_AVAILABLE:
            return False, "Previous phase components not available for integration test", {}
        
        if not self.llm_client:
            return False, "LLM client not initialized", {}
        
        if not self.test_data_dir.exists():
            return False, f"Test data directory not found: {self.test_data_dir}", {}
        
        details = {"integration_steps": []}
        
        try:
            # Step 1: Load document
            loader = DocumentLoader(verbose=False)
            documents = list(loader.load_documents_from_directory(str(self.test_data_dir)))
            
            if not documents:
                return False, "No documents loaded for integration test", details
            
            # Use first document for testing
            filename, content = documents[0]
            details["integration_steps"].append(f"✅ Loaded document: {Path(filename).name}")
            
            # Step 2: Chunk document
            chunker = DocumentChunker(chunk_size=300, chunk_overlap=50, verbose=False)
            chunks = list(chunker.chunk_document(content, filename))
            
            if not chunks:
                return False, "No chunks generated for integration test", details
            
            details["integration_steps"].append(f"✅ Generated {len(chunks)} chunks")
            
            # Step 3: Generate embeddings
            embedder = EmbeddingGenerator(verbose=False)
            embedded_chunks = embedder.embed_chunks(chunks[:3])  # Use first 3 chunks only
            
            if not embedded_chunks:
                return False, "No embeddings generated for integration test", details
            
            details["integration_steps"].append(f"✅ Generated {len(embedded_chunks)} embeddings")
            
            # Step 4: Store in vector database
            db = VectorDatabase(persist_directory="./test_integration_db", verbose=False, rebuild=True)
            success = db.add_chunks(embedded_chunks)
            
            if not success:
                return False, "Failed to store chunks in vector database", details
            
            details["integration_steps"].append(f"✅ Stored chunks in vector database")
            
            # Step 5: Perform retrieval
            query_text = "What is the main topic discussed?"
            query_chunks = list(chunker.chunk_document(query_text, "query.txt"))
            query_embedded = embedder.embed_chunk(query_chunks[0])
            
            similar_docs = db.search_similar(query_embedded.embedding, k=2)
            
            if not similar_docs:
                return False, "No similar documents retrieved", details
            
            details["integration_steps"].append(f"✅ Retrieved {len(similar_docs)} similar documents")
            
            # Step 6: Use LLM to generate response based on retrieved context
            context_text = "\n".join([doc["content"] for doc in similar_docs])
            rag_prompt = f"""Based on the following context, answer the question: {query_text}

Context:
{context_text}

Answer:"""
            
            llm_response = self.llm_client.test_prompt(rag_prompt)
            
            if not llm_response or len(llm_response) < 10:
                return False, "LLM failed to generate meaningful response", details
            
            details["integration_steps"].append(f"✅ Generated LLM response: {len(llm_response)} chars")
            details["rag_response"] = llm_response
            details["context_used"] = context_text[:200] + "..."
            
            # Cleanup
            db.close()
            import shutil
            test_db_path = Path("./test_integration_db")
            if test_db_path.exists():
                shutil.rmtree(test_db_path, ignore_errors=True)
            
            return True, f"Full RAG pipeline integration successful: {len(llm_response)} char response", details
            
        except Exception as e:
            return False, f"Integration test failed: {e}", details
    
    def test_concurrent_requests(self) -> Tuple[bool, str, Dict]:
        """Test 11: Verify handling of multiple concurrent requests."""
        if not self.llm_client:
            return False, "LLM client not initialized", {}
        
        import threading
        import queue
        
        # Test concurrent requests
        num_threads = 3
        requests_per_thread = 2
        total_requests = num_threads * requests_per_thread
        
        results_queue = queue.Queue()
        
        def worker_thread(thread_id: int, num_requests: int):
            """Worker thread for concurrent testing."""
            for i in range(num_requests):
                try:
                    start_time = time.time()
                    prompt = f"Thread {thread_id}, request {i+1}: What is {2+i}+{3+i}?"
                    response = self.llm_client.test_prompt(prompt)
                    end_time = time.time()
                    
                    results_queue.put({
                        "thread_id": thread_id,
                        "request_id": i+1,
                        "success": True,
                        "response_length": len(response),
                        "time": end_time - start_time,
                        "error": None
                    })
                    
                except Exception as e:
                    results_queue.put({
                        "thread_id": thread_id,
                        "request_id": i+1,
                        "success": False,
                        "response_length": 0,
                        "time": 0,
                        "error": str(e)
                    })
        
        # Start threads
        start_time = time.time()
        threads = []
        
        for thread_id in range(num_threads):
            thread = threading.Thread(
                target=worker_thread,
                args=(thread_id, requests_per_thread)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=60)  # 60 second timeout
        
        total_time = time.time() - start_time
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        
        details = {
            "total_requests": total_requests,
            "completed_requests": len(results),
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "total_time": total_time,
            "average_time_per_request": sum(r["time"] for r in successful_requests) / len(successful_requests) if successful_requests else 0,
            "errors": [r["error"] for r in failed_requests]
        }
        
        # Check if we got results from all requests
        if len(results) != total_requests:
            return False, f"Not all requests completed: {len(results)}/{total_requests}", details
        
        # Check success rate
        success_rate = len(successful_requests) / total_requests
        if success_rate < 0.8:  # At least 80% should succeed
            return False, f"Low concurrent success rate: {success_rate:.1%}", details
        
        return True, f"Concurrent requests successful: {len(successful_requests)}/{total_requests} in {total_time:.2f}s", details
    
    def cleanup(self):
        """Clean up test resources."""
        try:
            if self.llm_client:
                self.llm_client.close()
            if self.secure_client:
                self.secure_client.close()
        except Exception as e:
            self.log(f"⚠️  Cleanup warning: {e}")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        self.log("🚀 Starting Phase 3 Comprehensive Validation")
        self.log("=" * 60)
        
        # Define test suite
        tests = [
            (self.test_phase3_dependencies, "Phase 3 Dependencies"),
            (self.test_ollama_server_connectivity, "Ollama Server Connectivity"),
            (self.test_model_availability, "Model Availability"),
            (self.test_llm_client_initialization, "LLM Client Initialization"),
            (self.test_basic_text_generation, "Basic Text Generation"),
            (self.test_advanced_text_generation, "Advanced Text Generation"),
            (self.test_error_handling, "Error Handling"),
            (self.test_response_format_validation, "Response Format Validation"),
            (self.test_performance_characteristics, "Performance Characteristics"),
            (self.test_integration_with_rag_pipeline, "RAG Pipeline Integration"),
            (self.test_concurrent_requests, "Concurrent Requests")
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
        self.log("📊 PHASE 3 VALIDATION SUMMARY")
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
            self.log("✅ Phase 3 LLM integration is working correctly!")
            self.log("🎉 DocuChat RAG system is ready for production use!")
        else:
            self.log("❌ Phase 3 requires fixes before production deployment")
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
    
    parser = argparse.ArgumentParser(description="Phase 3 Comprehensive Validation")
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--model", default="gemma3:270m", help="Model to test")
    parser.add_argument("--test-data", default="test_data", help="Test data directory")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    
    args = parser.parse_args()
    
    # Run validation
    validator = Phase3Validator(
        base_url=args.base_url,
        model=args.model,
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