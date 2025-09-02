"""
CLI module for DocuChat RAG system.
Handles command line argument parsing, validation, and interactive chat interface.
"""

import argparse
import os
import sys
import time
import signal
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Native conversation memory implementation
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Model-specific optimal chunk sizes (with safety margins)
MODEL_CHUNK_SIZES = {
    "BAAI/bge-m3": 6000,                           # 75% of 8K context window
    "all-MiniLM-L6-v2": 200,                       # 80% of 256 optimal length
    "all-mpnet-base-v2": 300,                      # 80% of 384 optimal length
}

DEFAULT_CHUNK_SIZE = 1000  # Fallback for unknown models


def check_llm_model_available(model_name: str) -> bool:
    """
    Check if an LLM model is available in Ollama.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        True if model is available, False otherwise
    """
    import subprocess
    
    try:
        # Run 'ollama list' to get available models
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            # Check if model name appears in the output
            available_models = result.stdout.lower()
            return model_name.lower() in available_models
        else:
            logger.warning(f"Failed to check Ollama models: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.warning("Timeout checking Ollama models")
        return False
    except FileNotFoundError:
        logger.warning("Ollama not found - ensure Ollama is installed and in PATH")
        return False
    except Exception as e:
        logger.warning(f"Error checking Ollama models: {e}")
        return False


def get_available_llm_models() -> list:
    """
    Get list of available LLM models from Ollama.
    
    Returns:
        List of available model names
    """
    import subprocess
    
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            models = []
            
            # Skip header line and parse model names
            for line in lines[1:]:
                if line.strip():
                    # Extract model name (first column)
                    parts = line.split()
                    if parts:
                        models.append(parts[0])
            
            return models
        else:
            return []
            
    except Exception:
        return []


def get_optimal_chunk_size(embedding_model: str) -> int:
    """
    Get optimal chunk size for a given embedding model.
    
    Args:
        embedding_model: Name of the embedding model
        
    Returns:
        Optimal chunk size for the model
    """
    return MODEL_CHUNK_SIZES.get(embedding_model, DEFAULT_CHUNK_SIZE)


def format_chunk_size_info(embedding_model: str, chunk_size: int, is_auto: bool) -> str:
    """
    Format chunk size information for user display.
    
    Args:
        embedding_model: Name of the embedding model
        chunk_size: Actual chunk size being used
        is_auto: Whether chunk size was auto-selected
        
    Returns:
        Formatted information string
    """
    optimal = get_optimal_chunk_size(embedding_model)
    
    if is_auto:
        return f"📊 Auto-selected chunk size: {chunk_size} chars (optimal for {embedding_model})"
    elif chunk_size == optimal:
        return f"📊 Using chunk size: {chunk_size} chars (matches optimal for {embedding_model})"
    else:
        efficiency = "efficient" if abs(chunk_size - optimal) / optimal < 0.3 else "sub-optimal"
        return f"📊 Using chunk size: {chunk_size} chars (user override, optimal: {optimal}, {efficiency})"


class CLIArgs:
    """Container for parsed CLI arguments."""
    
    def __init__(self, directory: Optional[str], verbose: bool = False, chunk_size: int = 1000, 
                 top_k: int = 10, rebuild: bool = False, show_sources: bool = False,
                 chat: bool = False, max_context: int = 20000, embedding_model: str = 'BAAI/bge-m3',
                 llm: str = 'gemma3:1b', chunk_size_auto: bool = False, timeout: float = 1.0,
                 content_aware: bool = False, llm_classification: bool = False):
        self.directory = directory
        self.verbose = verbose
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.rebuild = rebuild
        self.show_sources = show_sources
        self.chat = chat
        self.max_context = max_context
        self.embedding_model = embedding_model
        self.llm = llm
        self.chunk_size_auto = chunk_size_auto  # Track if chunk size was auto-selected
        self.timeout = timeout  # Timeout in minutes
        self.content_aware = content_aware  # Enable content-aware chunking
        self.llm_classification = llm_classification  # Enable LLM-based query classification vs fast keyword matching
    
    def get_chunk_size_info(self) -> str:
        """
        Get formatted chunk size information for display.
        
        Returns:
            Formatted chunk size information string
        """
        return format_chunk_size_info(self.embedding_model, self.chunk_size, self.chunk_size_auto)
    
    def get_timeout_seconds(self) -> float:
        """
        Get timeout in seconds for the LLM client.
        
        Returns:
            Timeout in seconds (converts from minutes)
        """
        return self.timeout * 60.0


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog='docuchat',
        description='DocuChat RAG System - Complete Pipeline with Interactive Chat',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  docuchat /path/to/documents
  docuchat /path/to/documents --verbose --chunk-size 500
  docuchat /path/to/documents --rebuild --show-sources
  docuchat /path/to/documents --embedding-model all-mpnet-base-v2
  docuchat /path/to/documents --chat
  docuchat --chat
  docuchat --chat --top-k 5 --show-sources --embedding-model all-mpnet-base-v2
  docuchat /path/to/documents --timeout 2.5  # 2.5 minute timeout
  docuchat --chat --timeout 3.0  # 3 minute timeout for chat

Embedding Model Performance:
  BAAI/bge-m3        8K context, excellent quality (auto chunk: 6000 chars)
  all-MiniLM-L6-v2   Fast processing (auto chunk: 200 chars)
  all-mpnet-base-v2  Better quality (auto chunk: 300 chars)

Dynamic Chunk Sizing:
  Chunk size is automatically optimized for each embedding model.
  Use --chunk-size to override with a custom value.

Timeout Configuration:
  Default: 1.0 minute (60 seconds)
  Maximum: 5.0 minutes (300 seconds)
  Adjust based on your hardware performance and model size.
        """
    )
    
    # Optional positional argument
    parser.add_argument(
        'directory',
        type=str,
        nargs='?',
        help='Directory path containing documents to process (optional)'
    )
    
    # Optional flags
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output with progress information'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=None,  # Will be auto-selected based on embedding model
        metavar='SIZE',
        help='Size of text chunks for processing (auto-selected per model if not specified)'
    )
    
    parser.add_argument(
        '--content-aware',
        action='store_true',
        help='Enable content-aware chunking for better semantic boundaries'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        metavar='K',
        help='Number of top results to return for RAG context (default: 5)'
    )
    
    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Force rebuild of the document index'
    )
    
    parser.add_argument(
        '--show-sources',
        action='store_true',
        help='Show source documents in chat responses'
    )
    
    parser.add_argument(
        '--chat',
        action='store_true',
        help='Start interactive chat mode (use alone for chat-only mode)'
    )
    
    parser.add_argument(
        '--max-context',
        type=int,
        default=20000,
        metavar='CHARS',
        help='Maximum context length for RAG prompts (default: 20000)'
    )
    
    parser.add_argument(
        '--embedding-model',
        type=str,
        choices=['BAAI/bge-m3', 'all-MiniLM-L6-v2', 'all-mpnet-base-v2'],
        default='BAAI/bge-m3',
        metavar='MODEL',
        help='Embedding model: BAAI/bge-m3 (8K context, 1024d) or all-MiniLM-L6-v2 (fast, 384d) or all-mpnet-base-v2 (quality, 768d)'
    )
    
    parser.add_argument(
        '--llm',
        type=str,
        default='gemma3:1b',
        metavar='MODEL',
        help='LLM model name (must be installed in Ollama). Examples: gemma3:270m, gemma3:1b, qwen2.5:3b-instruct-q4_K_M'
    )
    
    parser.add_argument(
        '--timeout',
        type=float,
        default=1.0,
        metavar='MINUTES',
        help='LLM request timeout in minutes (max 5.0, default: 1.0 minute = 60 seconds)'
    )
    
    # Query classification method selection
    # Allows users to choose between fast keyword-based classification (default) 
    # and slower but more accurate LLM-based semantic analysis
    parser.add_argument(
        '--llm-classification',
        action='store_true',
        help='Use LLM-based query classification for higher accuracy (slower, more resource-intensive). Default: fast keyword-based classification'
    )
    
    return parser


def validate_directory(directory_path: str) -> Path:
    """
    Validate that the provided directory path exists and is accessible.
    
    Args:
        directory_path: String path to validate
        
    Returns:
        Path object if valid
        
    Raises:
        argparse.ArgumentTypeError: If path is invalid
    """
    path = Path(directory_path)
    
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Directory does not exist: {directory_path}")
    
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"Path is not a directory: {directory_path}")
    
    # Check if directory is readable by trying to access it
    try:
        os.access(path, os.R_OK)
    except (OSError, PermissionError):
        raise argparse.ArgumentTypeError(f"Directory is not readable: {directory_path}")
    
    return path


def parse_args(args: Optional[list] = None) -> CLIArgs:
    """
    Parse command line arguments and validate them.
    
    Args:
        args: Optional list of arguments (for testing)
        
    Returns:
        CLIArgs object with parsed arguments
        
    Raises:
        SystemExit: If arguments are invalid
    """
    parser = create_parser()
    
    try:
        parsed_args = parser.parse_args(args)
        
        # Validate that we have either directory or chat mode
        if not parsed_args.directory and not parsed_args.chat:
            parser.error("Must specify either a directory to process or --chat mode")
        
        # Smart validation for directory with chat context awareness
        validated_dir = None
        if parsed_args.directory:
            # Check if user likely meant to send a chat message, not specify directory
            if (parsed_args.chat and 
                len(parsed_args.directory) < 50 and  # Short strings are likely messages
                '/' not in parsed_args.directory and '\\' not in parsed_args.directory and  # No path separators
                not Path(parsed_args.directory).exists()):  # Doesn't exist as directory
                
                parser.error(f"It looks like you're trying to send '{parsed_args.directory}' as a chat message.\n"
                           f"For chat mode, use: python docuchat.py --chat\n"
                           f"Then type your message when the interactive chat starts.")
            
            # Normal directory validation
            validated_dir = validate_directory(parsed_args.directory)
        
        # Handle dynamic chunk sizing
        chunk_size_auto = False
        if parsed_args.chunk_size is None:
            # Auto-select optimal chunk size based on embedding model
            parsed_args.chunk_size = get_optimal_chunk_size(parsed_args.embedding_model)
            chunk_size_auto = True
        
        # Validate chunk size
        if parsed_args.chunk_size <= 0:
            parser.error("Chunk size must be positive")
        
        if parsed_args.chunk_size > 50000:
            parser.error("Chunk size too large (max 50000 characters)")
        
        if parsed_args.top_k <= 0:
            parser.error("Top-k must be positive")
        
        # Validate max_context
        if parsed_args.max_context <= 0:
            parser.error("Max context must be positive")
        
        # Validate timeout
        if parsed_args.timeout <= 0:
            parser.error("Timeout must be positive")
        
        if parsed_args.timeout > 5.0:
            parser.error("Timeout cannot exceed 5.0 minutes")
        
        # Validate LLM model is available
        if not check_llm_model_available(parsed_args.llm):
            available_models = get_available_llm_models()
            if available_models:
                available_str = ", ".join(available_models[:10])  # Show first 10
                if len(available_models) > 10:
                    available_str += f" (and {len(available_models) - 10} more)"
                parser.error(f"LLM model '{parsed_args.llm}' is not available.\n"
                           f"Available models: {available_str}\n"
                           f"Install the model with: ollama pull {parsed_args.llm}")
            else:
                parser.error(f"LLM model '{parsed_args.llm}' is not available.\n"
                           f"Could not check available models. Ensure Ollama is running.\n"
                           f"Install the model with: ollama pull {parsed_args.llm}")
        
        return CLIArgs(
            directory=str(validated_dir.resolve()) if validated_dir else None,
            verbose=parsed_args.verbose,
            chunk_size=parsed_args.chunk_size,
            top_k=parsed_args.top_k,
            rebuild=parsed_args.rebuild,
            show_sources=parsed_args.show_sources,
            chat=parsed_args.chat,
            max_context=parsed_args.max_context,
            embedding_model=parsed_args.embedding_model,
            llm=parsed_args.llm,
            chunk_size_auto=chunk_size_auto,
            timeout=parsed_args.timeout,
            content_aware=parsed_args.content_aware,
            llm_classification=parsed_args.llm_classification
        )
        
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))
    except Exception as e:
        parser.error(f"Unexpected error: {e}")


def main() -> CLIArgs:
    """Main entry point for CLI parsing."""
    try:
        return parse_args()
    except SystemExit as e:
        # Re-raise SystemExit to allow proper CLI behavior
        raise e
    except Exception as e:
        print(f"Error parsing arguments: {e}", file=sys.stderr)
        sys.exit(1)


class ChatInterface:
    """
    Interactive chat interface for DocuChat RAG system with conversation memory.
    
    Provides a user-friendly command-line chat experience with:
    - Native conversation memory using collections.deque for conversation context
    - Proper signal handling, source attribution, and error recovery
    - High-performance memory management with no external dependencies
    - Defensive programming with comprehensive error handling
    
    Memory Integration Features:
    - Maintains conversation context across questions using native Python structures
    - Automatically includes previous conversation history in RAG prompts
    - Configurable memory window (default: 10 conversation exchanges)
    - Thread-safe memory operations with proper error handling
    - Simple, reliable implementation without external dependencies
    
    Memory Lifecycle:
    1. Initialization: Sets up native deque-based conversation memory
    2. Question Processing: Retrieves context and enhances prompts
    3. Response Handling: Updates memory with Q&A pairs
    4. History Management: Provides unified clearing for conversation history
    """
    
    def __init__(self, rag_pipeline, args: CLIArgs):
        """
        Initialize the chat interface.
        
        Args:
            rag_pipeline: Initialized RAG pipeline
            args: CLI arguments
        """
        self.rag_pipeline = rag_pipeline
        self.args = args
        self.chat_history = []
        self.running = True
        
        # Initialize native conversation memory using collections.deque
        self.conversation_memory = deque(maxlen=10)  # Keep last 10 conversation exchanges
        self.memory_enabled = True
        self._log("Native conversation memory initialized successfully", "debug")
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        print("\n\n🛑 Chat session interrupted. Goodbye!")
        self.running = False
    
    def _get_conversation_context(self) -> str:
        """
        Retrieve conversation history from native memory for context.
        
        Returns:
            Formatted conversation context string for the LLM
        """
        if not self.memory_enabled or not self.conversation_memory:
            self._log("Native memory not available, returning empty context", "debug")
            return ""
        
        try:
            if not self.conversation_memory:
                self._log("No conversation history in memory", "debug")
                return ""
            
            # Format the conversation history for context
            context_parts = []
            context_parts.append("Previous conversation context:")
            
            # Get last 3 exchanges (6 messages total) for context
            recent_exchanges = list(self.conversation_memory)[-3:]
            
            for exchange in recent_exchanges:
                context_parts.append(f"Human: {exchange['user']}")
                context_parts.append(f"Assistant: {exchange['assistant']}")
            
            context = "\n".join(context_parts)
            self._log(f"Retrieved conversation context with {len(recent_exchanges)} exchanges", "debug")
            return context
            
        except Exception as e:
            self._log(f"Error retrieving conversation context: {e}", "warning")
            return ""
    
    def _update_conversation_memory(self, question: str, answer: str) -> None:
        """
        Save Q&A pairs to native conversation memory.
        
        Args:
            question: User's question
            answer: Assistant's answer
        """
        if not self.memory_enabled or self.conversation_memory is None:
            self._log("Native memory not available, skipping memory update", "debug")
            return
        
        try:
            # Create conversation exchange dictionary
            exchange = {
                "user": question,
                "assistant": answer,
                "timestamp": time.time()
            }
            
            # Add to deque (automatically manages max length)
            self.conversation_memory.append(exchange)
            
            self._log(f"Saved conversation exchange to native memory", "debug")
                
        except Exception as e:
            self._log(f"Error updating conversation memory: {e}", "warning")
    
    def _log(self, message: str, level: str = "info") -> None:
        """Log message if verbose mode is enabled."""
        if self.args.verbose:
            if level == "debug":
                logger.debug(f"[ChatInterface] {message}")
            elif level == "warning":
                logger.warning(f"[ChatInterface] {message}")
            elif level == "error":
                logger.error(f"[ChatInterface] {message}")
            else:
                logger.info(f"[ChatInterface] {message}")
    
    def _display_welcome(self) -> None:
        """Display welcome message and instructions."""
        print(f"\n{'='*60}")
        print("🤖 DocuChat Interactive Chat")
        print(f"{'='*60}")
        print("Ask questions about your documents!")
        print("\nCommands:")
        print("  /help     - Show this help message")
        print("  /stats    - Show pipeline statistics")
        print("  /perf     - Show performance timings")
        print("  /history  - Show chat history")
        print("  /clear    - Clear chat history")
        print("  /quit     - Exit chat")
        print("  /exit     - Exit chat")
        print(f"\nSettings:")
        print(f"  📊 Top-K results: {self.args.top_k}")
        print(f"  📝 Show sources: {'Yes' if self.args.show_sources else 'No'}")
        print(f"  🔍 Max context: {self.args.max_context:,} characters")
        print(f"  ⏱️  LLM timeout: {self.args.timeout:.1f} minutes ({self.args.get_timeout_seconds():.0f}s)")
        print(f"  💬 Verbose mode: {'Yes' if self.args.verbose else 'No'}")
        print(f"  🧠 Conversation memory: {'Enabled (Native)' if self.memory_enabled else 'Disabled'}")
        print(f"\n{'='*60}")
    
    def _display_help(self) -> None:
        """Display help information."""
        print("\n📚 DocuChat Help")
        print("-" * 40)
        print("Ask any question about your documents. Examples:")
        print("• What are the main topics covered?")
        print("• Summarize the key findings")
        print("• What does the document say about [topic]?")
        print("• Who are the main authors mentioned?")
        print("\nCommands:")
        print("• /help     - Show this help")
        print("• /stats    - Pipeline performance statistics")
        print("• /perf     - Show detailed timing breakdown")
        print("• /history  - View conversation history")
        print("• /clear    - Clear conversation history")
        print("• /quit     - Exit the chat")
        print("-" * 40)
    
    def _display_stats(self) -> None:
        """Display pipeline statistics."""
        try:
            stats = self.rag_pipeline.get_statistics()
            
            print("\n📊 Pipeline Statistics")
            print("-" * 40)
            print(f"Queries processed: {stats.get('queries_processed', 0)}")
            print(f"Successful queries: {stats.get('successful_queries', 0)}")
            print(f"Success rate: {stats.get('success_rate', 0):.1%}")
            print(f"Average processing time: {stats.get('average_processing_time', 0):.2f}s")
            print(f"Average context length: {stats.get('average_context_length', 0):,} chars")
            print(f"Average sources per query: {stats.get('average_sources_per_query', 0):.1f}")
            
            if self.args.verbose and 'component_info' in stats:
                component_info = stats['component_info']
                print(f"\nComponent Information:")
                print(f"• Embedding model: {component_info.get('embedding_model', 'Unknown')}")
                print(f"• LLM model: {component_info.get('llm_model', 'Unknown')}")
                
                if 'vector_db_info' in component_info:
                    db_info = component_info['vector_db_info']
                    print(f"• Database documents: {db_info.get('document_count', 0):,}")
            
            print("-" * 40)
            
        except Exception as e:
            print(f"❌ Error retrieving statistics: {e}")
    
    def _display_perf(self) -> None:
        """Display performance timing information."""
        try:
            from simple_timer import get_timer
            
            timer = get_timer()
            timing_results = timer.get_results()
            
            print("\n⚡ Performance Breakdown")
            print("-" * 40)
            print("Last Question Timing:")
            
            breakdown = timing_results.get_breakdown()
            total_time = sum(breakdown.values())
            
            if total_time > 0:
                for step, duration in breakdown.items():
                    if duration > 0:
                        percentage = (duration / total_time) * 100
                        print(f"• {step.title()}: {duration:.2f}s ({percentage:.1f}%)")
                
                print(f"\nTotal: {total_time:.2f}s")
            else:
                print("No timing data available. Ask a question first!")
            
            print("-" * 40)
            
        except Exception as e:
            print(f"❌ Error retrieving performance data: {e}")
    
    def _display_history(self) -> None:
        """Display chat history."""
        if not self.chat_history:
            print("\n📜 No chat history yet. Start asking questions!")
            return
        
        print(f"\n📜 Chat History ({len(self.chat_history)} messages)")
        print("-" * 40)
        
        for i, entry in enumerate(self.chat_history, 1):
            timestamp = entry.get('timestamp', '')
            question = entry.get('question', '')[:100]
            if len(entry.get('question', '')) > 100:
                question += "..."
            
            print(f"{i}. [{timestamp}] {question}")
        
        print("-" * 40)
    
    def _clear_history(self) -> None:
        """Clear chat history and conversation memory."""
        # Clear local display history
        self.chat_history.clear()
        self._log("Local chat history cleared", "debug")
        
        # Clear native conversation memory
        if self.memory_enabled and self.conversation_memory is not None:
            try:
                self.conversation_memory.clear()
                self._log("Native conversation memory cleared", "debug")
            except Exception as e:
                self._log(f"Warning: Could not clear native conversation memory: {e}", "warning")
        
        # Also clear any conversation memory on the RAG pipeline (legacy compatibility)
        if hasattr(self.rag_pipeline, 'conversation_memory') and self.rag_pipeline.conversation_memory:
            try:
                self.rag_pipeline.conversation_memory.clear()
                self._log("RAG pipeline conversation memory cleared", "debug")
            except Exception as e:
                self._log(f"Warning: Could not clear RAG pipeline conversation memory: {e}", "warning")
        
        print("🗑️  Chat history and conversation memory cleared.")
    
    def _format_sources(self, sources: list) -> str:
        """Format source information for display."""
        if not sources or not self.args.show_sources:
            return ""
        
        source_text = "\n\n📚 Sources:"
        source_text += "\n" + "-" * 20
        
        for i, source in enumerate(sources[:5], 1):  # Limit to top 5 sources
            metadata = source.get('metadata', {})
            source_file = metadata.get('source_file', 'Unknown')
            distance = source.get('distance', 0)
            
            # Convert cosine distance to relevance score  
            # ChromaDB uses cosine distance [0, 2] where 0=identical, 2=maximally different
            relevance = max(0, min(100, int(100 - (distance * 50))))
            
            source_text += f"\n{i}. {source_file} (relevance: {relevance}%)"
            
            # Show chunk preview if verbose
            if self.args.verbose:
                content = source.get('content', '')[:150]
                if len(source.get('content', '')) > 150:
                    content += "..."
                source_text += f"\n   Preview: {content}"
        
        return source_text
    
    def _process_question(self, question: str) -> None:
        """Process a user question through the RAG pipeline."""
        if not question.strip():
            print("❓ Please enter a question.")
            return
        
        try:
            # Import here to avoid circular imports
            from simple_timer import get_timer, reset_timer
            
            # Reset timer for new question
            reset_timer()
            timer = get_timer()
            
            start_time = time.time()
            
            if self.args.verbose:
                print(f"🔍 Processing question...")
            
            # Time the input processing
            with timer.time_block("input_processing"):
                # Get conversation context from native memory
                conversation_context = self._get_conversation_context()
                
                # Prepare the enhanced question with context if available
                enhanced_question = question
                if conversation_context:
                    enhanced_question = f"{conversation_context}\n\nCurrent question: {question}"
                    self._log(f"Enhanced question with conversation context", "debug")
                
                # Get answer from RAG pipeline
                # Check if the RAG pipeline supports conversation_context parameter
                try:
                    # Try with conversation context parameter first
                    import inspect
                    sig = inspect.signature(self.rag_pipeline.answer_question)
                    if 'conversation_context' in sig.parameters:
                        result = self.rag_pipeline.answer_question(
                            question=question,
                            top_k=self.args.top_k,
                            include_sources=self.args.show_sources,
                            conversation_context=conversation_context
                        )
                    else:
                        # Fallback: use enhanced question if we have context
                        result = self.rag_pipeline.answer_question(
                            question=enhanced_question if conversation_context else question,
                            top_k=self.args.top_k,
                            include_sources=self.args.show_sources
                        )
                except Exception as fallback_error:
                    # Final fallback: use basic question
                    self._log(f"Using fallback approach for RAG pipeline: {fallback_error}", "debug")
                    result = self.rag_pipeline.answer_question(
                        question=question,
                        top_k=self.args.top_k,
                        include_sources=self.args.show_sources
                    )
            
            processing_time = time.time() - start_time
            
            # Time the response display
            with timer.time_block("response_display"):
                # Display answer with performance info
                tokens_per_sec = f"{result.tokens_per_second:.1f}" if result.tokens_per_second else "0.0"
                
                print(f"\n🤖 [{tokens_per_sec} t/s] {result.answer}")
                
                # Display sources if requested
                if self.args.show_sources and result.sources:
                    print(self._format_sources(result.sources))
            
            # Display performance timings
            timing_display = timer.format_for_display(show_total=False)
            if timing_display:
                print(f"\n⏱️  {timing_display}")
            
            # Display additional timing info if verbose
            if self.args.verbose:
                print(f"📊 Confidence: {result.confidence_score:.2f}")
                print(f"🔍 Sources found: {len(result.sources)}")
                print(f"📈 Total processing: {processing_time:.2f}s")
            
            # Update conversation memory with Q&A pair
            self._update_conversation_memory(question, result.answer)
            
            # Add to history
            self.chat_history.append({
                'timestamp': time.strftime('%H:%M:%S'),
                'question': question,
                'answer': result.answer,
                'sources': len(result.sources),
                'processing_time': processing_time
            })
            
        except Exception as e:
            print(f"❌ Error processing question: {e}")
            self._log(f"Question processing error: {e}", "error")
    
    def _handle_command(self, command: str) -> bool:
        """
        Handle chat commands.
        
        Args:
            command: Command string (including /)
            
        Returns:
            True to continue chat, False to exit
        """
        command = command.lower().strip()
        
        if command in ['/quit', '/exit']:
            return False
        elif command == '/help':
            self._display_help()
        elif command == '/stats':
            self._display_stats()
        elif command == '/perf':
            self._display_perf()
        elif command == '/history':
            self._display_history()
        elif command == '/clear':
            self._clear_history()
        else:
            print(f"❓ Unknown command: {command}")
            print("Type /help for available commands.")
        
        return True
    
    def start_chat(self) -> None:
        """Start the interactive chat loop."""
        self._display_welcome()
        
        try:
            while self.running:
                try:
                    # Get user input
                    user_input = input("\n💬 You: ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle commands
                    if user_input.startswith('/'):
                        if not self._handle_command(user_input):
                            break
                        continue
                    
                    # Process question
                    self._process_question(user_input)
                    
                except EOFError:
                    # Handle Ctrl+D
                    print("\n\n👋 Goodbye!")
                    break
                except KeyboardInterrupt:
                    # Handle Ctrl+C
                    print("\n\n🛑 Chat interrupted. Use /quit to exit gracefully.")
                    continue
                except Exception as e:
                    print(f"❌ Unexpected error: {e}")
                    self._log(f"Chat loop error: {e}", "error")
                    continue
        
        finally:
            print("📊 Chat session ended.")
            if self.chat_history:
                print(f"Total questions asked: {len(self.chat_history)}")
            print("Thank you for using DocuChat! 🚀")


def start_interactive_chat(rag_pipeline, args: CLIArgs) -> None:
    """
    Start the interactive chat interface.
    
    Args:
        rag_pipeline: Initialized RAG pipeline
        args: CLI arguments
    """
    try:
        chat = ChatInterface(rag_pipeline, args)
        chat.start_chat()
    except Exception as e:
        logger.error(f"Chat interface error: {e}")
        print(f"❌ Chat interface failed: {e}")


if __name__ == "__main__":
    args = main()
    print(f"Parsed arguments: {vars(args)}")