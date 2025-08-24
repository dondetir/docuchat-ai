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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CLIArgs:
    """Container for parsed CLI arguments."""
    
    def __init__(self, directory: Optional[str], verbose: bool = False, chunk_size: int = 1000, 
                 top_k: int = 10, rebuild: bool = False, show_sources: bool = False,
                 chat: bool = False, max_context: int = 20000, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.directory = directory
        self.verbose = verbose
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.rebuild = rebuild
        self.show_sources = show_sources
        self.chat = chat
        self.max_context = max_context
        self.embedding_model = embedding_model


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

Embedding Model Performance:
  all-MiniLM-L6-v2   Fast processing (~1s startup, good for development)
  all-mpnet-base-v2  Better quality (~16s startup, 40-130% accuracy improvement)
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
        default=1000,
        metavar='SIZE',
        help='Size of text chunks for processing (default: 1000)'
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
        choices=['all-MiniLM-L6-v2', 'all-mpnet-base-v2'],
        default='all-MiniLM-L6-v2',
        metavar='MODEL',
        help='Embedding model: all-MiniLM-L6-v2 (fast, 384d) or all-mpnet-base-v2 (quality, 768d)'
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
        
        # Validate numeric arguments
        if parsed_args.chunk_size <= 0:
            parser.error("Chunk size must be positive")
        
        if parsed_args.top_k <= 0:
            parser.error("Top-k must be positive")
        
        # Validate max_context
        if parsed_args.max_context <= 0:
            parser.error("Max context must be positive")
        
        return CLIArgs(
            directory=str(validated_dir.resolve()) if validated_dir else None,
            verbose=parsed_args.verbose,
            chunk_size=parsed_args.chunk_size,
            top_k=parsed_args.top_k,
            rebuild=parsed_args.rebuild,
            show_sources=parsed_args.show_sources,
            chat=parsed_args.chat,
            max_context=parsed_args.max_context,
            embedding_model=parsed_args.embedding_model
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
    Interactive chat interface for DocuChat RAG system.
    
    Provides a user-friendly command-line chat experience with
    proper signal handling, source attribution, and error recovery.
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
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        print("\n\n🛑 Chat session interrupted. Goodbye!")
        self.running = False
    
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
        print(f"  💬 Verbose mode: {'Yes' if self.args.verbose else 'No'}")
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
        """Clear chat history."""
        self.chat_history.clear()
        print("🗑️  Chat history cleared.")
    
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
            
            # Convert distance to relevance score
            relevance = max(0, min(100, int((1.0 - distance) * 100)))
            
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
                # Get answer from RAG pipeline
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