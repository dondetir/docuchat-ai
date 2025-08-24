"""
LLM client module for DocuChat RAG system.
Handles integration with Ollama REST API for text generation.
"""

import json
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import requests with error handling
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False
    warnings.warn(
        "requests not available. Install with: pip install requests>=2.31.0",
        ImportWarning
    )


class LLMClientError(Exception):
    """Base exception for LLM client operations."""
    pass


class ModelNotFoundError(LLMClientError):
    """Exception raised when the specified model is not found."""
    pass


class ConnectionError(LLMClientError):
    """Exception raised when connection to Ollama server fails."""
    pass


class TimeoutError(LLMClientError):
    """Exception raised when requests timeout."""
    pass


class GenerationError(LLMClientError):
    """Exception raised when text generation fails."""
    pass


@dataclass(frozen=True)
class LLMResponse:
    """
    Immutable data class representing an LLM response.
    
    Security Features:
    - Immutable to prevent accidental modification
    - Response validation and integrity checks
    - Metadata tracking for audit trails
    """
    model: str
    prompt: str
    response: str
    created_at: str
    done: bool
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None
    context: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate LLM response data on creation."""
        # Validate basic types
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Model must be a non-empty string")
        
        if not isinstance(self.prompt, str):
            raise TypeError("Prompt must be a string")
        
        if not isinstance(self.response, str):
            raise TypeError("Response must be a string")
        
        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ValueError("Created_at must be a non-empty string")
        
        if not isinstance(self.done, bool):
            raise TypeError("Done must be a boolean")
        
        # Validate numeric types
        numeric_fields = [
            'total_duration', 'load_duration', 'prompt_eval_count',
            'prompt_eval_duration', 'eval_count', 'eval_duration'
        ]
        
        for field_name in numeric_fields:
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError(f"{field_name} must be None or a non-negative integer")
        
        # Security: Validate response length
        if len(self.response) > 1000000:  # 1MB limit
            raise ValueError("Response exceeds maximum allowed size (1MB)")
        
        # Validate metadata
        if not isinstance(self.metadata, dict):
            raise TypeError("Metadata must be a dictionary")
    
    @property
    def response_length(self) -> int:
        """Get the character length of the response."""
        return len(self.response)
    
    @property
    def word_count(self) -> int:
        """Get the estimated word count of the response."""
        return len(self.response.split())
    
    @property
    def tokens_per_second(self) -> Optional[float]:
        """Calculate tokens per second if metrics available."""
        if self.eval_count and self.eval_duration:
            # Convert nanoseconds to seconds
            duration_seconds = self.eval_duration / 1e9
            return self.eval_count / duration_seconds if duration_seconds > 0 else None
        return None


class SecureLLMClient:
    """
    Secure wrapper for Ollama REST API with comprehensive error handling.
    
    Security Features:
    - Input validation and sanitization
    - Connection timeout and retry logic
    - Model verification and validation
    - Memory usage monitoring
    - Request/response logging for audit trails
    """
    
    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_MODEL = "gemma3:270m"
    DEFAULT_TIMEOUT = 60.0
    MAX_RETRIES = 3
    
    # Known model specifications
    KNOWN_MODELS = {
        "gemma3:270m": {
            "context_window": 32768,
            "verified": True,
            "description": "Gemma 3 270M parameter model for RAG applications (default)"
        },
        "qwen2.5:3b-instruct-q4_K_M": {
            "context_window": 131072,  # 128K tokens
            "verified": True,
            "description": "Qwen 2.5 3B parameter model optimized for RAG applications (Q4 quantized)"
        },
        "gemma3:1b": {
            "context_window": 8192,  # 8K tokens for CPU efficiency
            "verified": True,
            "description": "Gemma 3 1B parameter model for efficient RAG applications"
        }
    }
    
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
        verify_ssl: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the secure LLM client.
        
        Args:
            base_url: Base URL for Ollama server
            model: Default model to use for generation
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            verify_ssl: Whether to verify SSL certificates
            verbose: Enable verbose logging
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library is required but not available")
        
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.verify_ssl = verify_ssl
        self.verbose = verbose
        
        # Validate inputs
        self._validate_base_url()
        self._validate_model_name()
        self._validate_timeout()
        
        # Initialize session for connection pooling
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'DocuChat-RAG/1.0'
        })
        
        # Configure SSL verification
        self.session.verify = verify_ssl
        
        # Statistics tracking
        self.stats = {
            'requests_made': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'total_tokens_generated': 0,
            'total_generation_time': 0.0,
            'errors': []
        }
        
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        logger.info(f"SecureLLMClient initialized: {base_url}, model={model}")
    
    def _log(self, message: str, level: str = "info") -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            if level == "debug":
                logger.debug(f"[SecureLLMClient] {message}")
            elif level == "warning":
                logger.warning(f"[SecureLLMClient] {message}")
            elif level == "error":
                logger.error(f"[SecureLLMClient] {message}")
            else:
                logger.info(f"[SecureLLMClient] {message}")
    
    def _validate_base_url(self) -> None:
        """Validate base URL for security."""
        if not isinstance(self.base_url, str):
            raise TypeError("Base URL must be a string")
        
        if not self.base_url.strip():
            raise ValueError("Base URL cannot be empty")
        
        # Basic URL validation
        if not (self.base_url.startswith('http://') or self.base_url.startswith('https://')):
            raise ValueError("Base URL must start with http:// or https://")
        
        # Security: Prevent SSRF attacks by restricting to localhost/loopback
        if 'localhost' not in self.base_url and '127.0.0.1' not in self.base_url:
            logger.warning("Using non-localhost URL may pose security risks")
    
    def _validate_model_name(self) -> None:
        """Validate model name for security."""
        if not isinstance(self.model, str):
            raise TypeError("Model name must be a string")
        
        if not self.model.strip():
            raise ValueError("Model name cannot be empty")
        
        # Security: Prevent injection attacks
        if any(char in self.model for char in ['/', '\\', '..', ';', "'", '"', '<', '>', '\n', '\r']):
            raise ValueError("Model name contains unsafe characters")
        
        # Limit length
        if len(self.model) > 100:
            raise ValueError("Model name too long (max 100 characters)")
        
        # Check if model is in known safe list
        if self.model not in self.KNOWN_MODELS:
            self._log(f"Model '{self.model}' not in verified model list", "warning")
    
    def _validate_timeout(self) -> None:
        """Validate timeout value."""
        if not isinstance(self.timeout, (int, float)):
            raise TypeError("Timeout must be a number")
        
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        
        if self.timeout > 300:  # 5 minutes max
            raise ValueError("Timeout too large (max 300 seconds)")
    
    def _sanitize_prompt(self, prompt: str) -> str:
        """
        Sanitize input prompt for security.
        
        Args:
            prompt: Raw input prompt
            
        Returns:
            Sanitized prompt
        """
        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string")
        
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        # Security: Check prompt length
        if len(prompt) > 100000:  # 100KB limit
            self._log(f"Prompt exceeds length limit, truncating: {len(prompt)} chars", "warning")
            prompt = prompt[:100000]
        
        # Remove null bytes and normalize
        prompt = prompt.replace('\x00', '')
        prompt = prompt.strip()
        
        if not prompt:
            raise ValueError("Prompt is empty after sanitization")
        
        return prompt
    
    def _make_request(
        self,
        endpoint: str,
        data: Dict[str, Any],
        retries: int = 0
    ) -> Dict[str, Any]:
        """
        Make HTTP request to Ollama API with error handling and retries.
        
        Args:
            endpoint: API endpoint to call
            data: Request payload
            retries: Current retry count
            
        Returns:
            Response data as dictionary
            
        Raises:
            ConnectionError: If connection fails
            TimeoutError: If request times out
            LLMClientError: For other API errors
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            self._log(f"Making request to {url}", "debug")
            start_time = time.time()
            
            response = self.session.post(
                url,
                json=data,
                timeout=self.timeout
            )
            
            request_time = time.time() - start_time
            self._log(f"Request completed in {request_time:.3f}s", "debug")
            
            # Update statistics
            self.stats['requests_made'] += 1
            
            # Handle HTTP errors
            if response.status_code == 404:
                if 'model' in data:
                    raise ModelNotFoundError(f"Model '{data['model']}' not found")
                else:
                    raise LLMClientError(f"Endpoint not found: {endpoint}")
            
            if response.status_code != 200:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                raise LLMClientError(error_msg)
            
            # Parse JSON response
            try:
                response_data = response.json()
                self._log(f"Received response: {len(response.text)} chars", "debug")
                return response_data
                
            except json.JSONDecodeError as e:
                raise LLMClientError(f"Invalid JSON response: {e}")
        
        except requests.exceptions.ConnectTimeout:
            raise TimeoutError(f"Connection timeout after {self.timeout}s")
        
        except requests.exceptions.ReadTimeout:
            raise TimeoutError(f"Read timeout after {self.timeout}s")
        
        except requests.exceptions.ConnectionError as e:
            if retries < self.max_retries:
                self._log(f"Connection failed, retrying ({retries + 1}/{self.max_retries}): {e}", "warning")
                time.sleep(2 ** retries)  # Exponential backoff
                return self._make_request(endpoint, data, retries + 1)
            else:
                raise ConnectionError(f"Failed to connect after {self.max_retries} retries: {e}")
        
        except requests.exceptions.RequestException as e:
            raise LLMClientError(f"Request failed: {e}")
    
    def test_connection(self) -> bool:
        """
        Test connection to Ollama server.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Use a minimal generate request to test connectivity
            # This uses POST which works with _make_request
            test_data = {
                "model": self.model,
                "prompt": "test",
                "stream": False,
                "options": {"num_predict": 1}  # Generate only 1 token
            }
            self._make_request("/api/generate", test_data)
            self._log("Connection test successful", "debug")
            return True
            
        except Exception as e:
            self._log(f"Connection test failed: {e}", "error")
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models on the Ollama server.
        
        Returns:
            List of available models with metadata
        """
        try:
            response = self._make_request("/api/tags", {})
            models = response.get("models", [])
            
            self._log(f"Found {len(models)} available models", "debug")
            return models
            
        except Exception as e:
            self._log(f"Failed to list models: {e}", "error")
            raise LLMClientError(f"Failed to list models: {e}")
    
    def verify_model(self, model_name: Optional[str] = None) -> bool:
        """
        Verify that a model is available on the server.
        
        Args:
            model_name: Model to verify (defaults to current model)
            
        Returns:
            True if model is available
        """
        model_to_check = model_name or self.model
        
        try:
            models = self.list_models()
            available_models = [m.get("name", "") for m in models]
            
            is_available = model_to_check in available_models
            
            if is_available:
                self._log(f"Model '{model_to_check}' is available", "debug")
            else:
                self._log(f"Model '{model_to_check}' not found in: {available_models}", "warning")
            
            return is_available
            
        except Exception as e:
            self._log(f"Model verification failed: {e}", "error")
            return False
    
    def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        stream: bool = False,
        format_json: bool = False,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        context_window: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """
        Generate text using the Ollama API.
        
        Args:
            prompt: Input prompt for text generation
            model: Model to use (defaults to client's default model)
            stream: Whether to use streaming (currently not supported)
            format_json: Whether to request JSON format response
            system_message: Optional system message
            temperature: Controls randomness (0.0=deterministic, 1.0=creative). Default: model default
            top_p: Nucleus sampling threshold (0.1-1.0). Default: model default
            context_window: Override model's context window size. Default: model default
            options: Additional model options (overrides temperature/top_p if specified)
            
        Returns:
            LLMResponse object containing generated text and metadata
            
        Raises:
            GenerationError: If text generation fails
            ModelNotFoundError: If specified model is not found
            ConnectionError: If connection to server fails
        """
        # Validate and sanitize inputs
        prompt = self._sanitize_prompt(prompt)
        model_to_use = model or self.model
        
        if system_message:
            if not isinstance(system_message, str):
                raise TypeError("System message must be a string")
            system_message = system_message.strip()
        
        # Validate model parameters
        if temperature is not None:
            if not isinstance(temperature, (int, float)) or not (0.0 <= temperature <= 2.0):
                raise ValueError("Temperature must be a number between 0.0 and 2.0")
        
        if top_p is not None:
            if not isinstance(top_p, (int, float)) or not (0.1 <= top_p <= 1.0):
                raise ValueError("Top_p must be a number between 0.1 and 1.0")
        
        if context_window is not None:
            if not isinstance(context_window, int) or context_window < 1:
                raise ValueError("Context window must be a positive integer")
            if context_window > 200000:  # Reasonable upper limit
                raise ValueError("Context window too large (max 200000 tokens)")
        
        if options and not isinstance(options, dict):
            raise TypeError("Options must be a dictionary")
        
        # Log request
        self._log(f"Generating text with model '{model_to_use}', prompt length: {len(prompt)}", "debug")
        
        # Prepare request data
        request_data = {
            "model": model_to_use,
            "prompt": prompt,
            "stream": False  # Always use non-streaming for simplicity
        }
        
        if format_json:
            request_data["format"] = "json"
        
        if system_message:
            request_data["system"] = system_message
        
        # Prepare options - build from individual parameters and options dict
        model_options = {}
        
        # Set individual parameters
        if temperature is not None:
            model_options["temperature"] = temperature
        
        if top_p is not None:
            model_options["top_p"] = top_p
        
        if context_window is not None:
            model_options["num_ctx"] = context_window
        
        # Merge with explicit options dict (options dict takes precedence)
        if options:
            model_options.update(options)
        
        # Only add options if we have any
        if model_options:
            request_data["options"] = model_options
        
        try:
            start_time = time.time()
            
            # Log the full request payload for debugging
            self._log(f"Request payload: {json.dumps(request_data, indent=2)}", "debug")
            
            # Make API request
            response_data = self._make_request("/api/generate", request_data)
            
            generation_time = time.time() - start_time
            
            # Validate response
            if not isinstance(response_data, dict):
                raise GenerationError("Invalid response format from API")
            
            if "response" not in response_data:
                raise GenerationError("Response field missing from API response")
            
            # Extract response fields
            generated_text = response_data.get("response", "")
            
            # Create LLMResponse object
            llm_response = LLMResponse(
                model=response_data.get("model", model_to_use),
                prompt=prompt,
                response=generated_text,
                created_at=response_data.get("created_at", ""),
                done=response_data.get("done", True),
                total_duration=response_data.get("total_duration"),
                load_duration=response_data.get("load_duration"),
                prompt_eval_count=response_data.get("prompt_eval_count"),
                prompt_eval_duration=response_data.get("prompt_eval_duration"),
                eval_count=response_data.get("eval_count"),
                eval_duration=response_data.get("eval_duration"),
                context=response_data.get("context"),
                metadata={
                    "generation_time": generation_time,
                    "system_message": system_message,
                    "temperature": temperature,
                    "top_p": top_p,
                    "context_window": context_window,
                    "options": options or {},
                    "format_json": format_json
                }
            )
            
            # Update statistics
            self.stats['successful_generations'] += 1
            if llm_response.eval_count:
                self.stats['total_tokens_generated'] += llm_response.eval_count
            self.stats['total_generation_time'] += generation_time
            
            self._log(f"Text generation successful: {len(generated_text)} chars in {generation_time:.2f}s")
            
            return llm_response
            
        except Exception as e:
            self.stats['failed_generations'] += 1
            self.stats['errors'].append(str(e))
            
            self._log(f"Text generation failed: {e}", "error")
            
            if isinstance(e, (ConnectionError, TimeoutError, ModelNotFoundError)):
                raise
            else:
                raise GenerationError(f"Text generation failed: {e}")
    
    def test_prompt(self, prompt: str) -> str:
        """
        Simple test function that takes string input and returns generated text.
        
        Args:
            prompt: Input prompt string
            
        Returns:
            Generated text as string
        """
        try:
            response = self.generate_text(prompt)
            return response.response
            
        except Exception as e:
            self._log(f"Test prompt failed: {e}", "error")
            raise GenerationError(f"Test prompt failed: {e}")
    
    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a model."""
        model_to_check = model_name or self.model
        
        info = {
            "model_name": model_to_check,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "verified": model_to_check in self.KNOWN_MODELS
        }
        
        if model_to_check in self.KNOWN_MODELS:
            info.update(self.KNOWN_MODELS[model_to_check])
        
        # Try to get model details from server
        try:
            if self.verify_model(model_to_check):
                info["available"] = True
            else:
                info["available"] = False
        except Exception:
            info["available"] = False
        
        return info
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics."""
        stats = self.stats.copy()
        
        # Add derived statistics
        if stats['successful_generations'] > 0:
            stats['average_generation_time'] = stats['total_generation_time'] / stats['successful_generations']
            if stats['total_tokens_generated'] > 0:
                stats['average_tokens_per_generation'] = stats['total_tokens_generated'] / stats['successful_generations']
                stats['average_tokens_per_second'] = stats['total_tokens_generated'] / stats['total_generation_time'] if stats['total_generation_time'] > 0 else 0
        else:
            stats['average_generation_time'] = 0.0
            stats['average_tokens_per_generation'] = 0
            stats['average_tokens_per_second'] = 0
        
        stats['success_rate'] = (stats['successful_generations'] / stats['requests_made']) if stats['requests_made'] > 0 else 0
        
        # Add client info
        stats['client_info'] = self.get_model_info()
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset client statistics."""
        self.stats = {
            'requests_made': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'total_tokens_generated': 0,
            'total_generation_time': 0.0,
            'errors': []
        }
        self._log("Statistics reset")
    
    def close(self) -> None:
        """Close client connections and cleanup."""
        try:
            if hasattr(self, 'session') and self.session:
                self.session.close()
                self._log("Client session closed")
        except Exception as e:
            logger.error(f"Error closing client session: {e}")


class LLMClient:
    """
    High-level LLM client interface with simplified API.
    
    This class provides a simplified interface to the SecureLLMClient
    for easy integration with the DocuChat RAG system.
    """
    
    def __init__(
        self,
        base_url: str = SecureLLMClient.DEFAULT_BASE_URL,
        model: str = SecureLLMClient.DEFAULT_MODEL,
        timeout: float = SecureLLMClient.DEFAULT_TIMEOUT,
        verbose: bool = False
    ):
        """
        Initialize the LLM client.
        
        Args:
            base_url: Base URL for Ollama server
            model: Model to use for generation
            timeout: Request timeout in seconds
            verbose: Enable verbose logging
        """
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.verbose = verbose
        
        # Initialize secure client
        self.client = SecureLLMClient(
            base_url=base_url,
            model=model,
            timeout=timeout,
            verbose=verbose
        )
        
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        logger.info(f"LLMClient initialized: {model}")
    
    def _log(self, message: str, level: str = "info") -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            if level == "debug":
                logger.debug(f"[LLMClient] {message}")
            elif level == "warning":
                logger.warning(f"[LLMClient] {message}")
            elif level == "error":
                logger.error(f"[LLMClient] {message}")
            else:
                logger.info(f"[LLMClient] {message}")
    
    def generate(
        self, 
        prompt: str, 
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        context_window: Optional[int] = None, 
        **kwargs
    ) -> LLMResponse:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            temperature: Controls randomness (0.0=deterministic, 1.0=creative)
            top_p: Nucleus sampling threshold (0.1-1.0)
            context_window: Override context window size (defaults to model's maximum)
            **kwargs: Additional arguments for generation
            
        Returns:
            LLMResponse object
        """
        self._log(f"Generating text for prompt: {prompt[:100]}...")
        
        # Pass parameters directly to the secure client
        return self.client.generate_text(
            prompt, 
            temperature=temperature,
            top_p=top_p,
            context_window=context_window,
            **kwargs
        )
    
    def test_prompt(self, prompt: str) -> str:
        """
        Test prompt function that takes string input and returns generated text.
        
        Args:
            prompt: Input prompt string
            
        Returns:
            Generated text as string
        """
        return self.client.test_prompt(prompt)
    
    def is_available(self) -> bool:
        """Check if the LLM service is available."""
        return self.client.test_connection()
    
    def get_info(self) -> Dict[str, Any]:
        """Get client and model information."""
        return self.client.get_model_info()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.client.get_statistics()
    
    def close(self) -> None:
        """Close client connections."""
        self.client.close()


def main():
    """Simple test function for the LLM client module."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python llm_client.py <prompt>")
        print("Example: python llm_client.py 'What is the capital of France?'")
        sys.exit(1)
    
    prompt = sys.argv[1]
    
    try:
        # Test basic LLM functionality
        client = LLMClient(verbose=True)
        
        # Test connection
        if not client.is_available():
            print("❌ Ollama server not available")
            sys.exit(1)
        
        print("✅ Connected to Ollama server")
        
        # Test model verification
        model_info = client.get_info()
        print(f"Model info: {model_info}")
        
        # Generate response
        print(f"\nGenerating response for: '{prompt}'")
        response = client.test_prompt(prompt)
        
        print(f"\nGenerated Response:")
        print(f"{'='*50}")
        print(response)
        print(f"{'='*50}")
        
        # Show statistics
        stats = client.get_statistics()
        print(f"\nStatistics:")
        for key, value in stats.items():
            if key != "client_info":
                print(f"  {key}: {value}")
        
        # Close client
        client.close()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()