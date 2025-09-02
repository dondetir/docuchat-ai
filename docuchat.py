#!/usr/bin/env python3
"""
DocuChat - Complete RAG System with Interactive Chat

Main entry point for the DocuChat RAG system.
Supports document processing, vectorization, and interactive chat functionality.
"""

import sys
import os
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from cli import parse_args
from document_loader import DocumentLoader

# Core component imports with error handling
try:
    from chunker import DocumentChunker
    from embeddings import EmbeddingGenerator
    from vector_db import VectorDatabase
    from llm_client import LLMClient
    from rag_pipeline import RAGPipeline
    from cli import start_interactive_chat
    from progress_manager import create_progress_manager
    CORE_COMPONENTS_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Core components not available: {e}", ImportWarning)
    CORE_COMPONENTS_AVAILABLE = False


def run_phase1_only(args) -> int:
    """
    Run Phase 1 processing only (document loading and text extraction).
    
    Args:
        args: Parsed CLI arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        if args.verbose:
            print(f"DocuChat RAG System - Phase 1 (Document Loading)")
            print(f"Processing directory: {args.directory}")
            print(f"Verbose mode: {args.verbose}")
            print("-" * 50)
        
        # Initialize document loader
        loader = DocumentLoader(verbose=False)  # Use progress manager instead
        
        # First, discover all files to get total count
        discovered_files = loader.discover_files(args.directory)
        
        if not discovered_files:
            print("⚠️  No supported documents found in directory.")
            return 1
        
        # Initialize progress manager
        progress = create_progress_manager(
            total_items=len(discovered_files),
            operation="Loading Documents",
            verbose=args.verbose,
            quiet=False  # Always show progress bar
        )
        
        # Process documents
        total_characters = 0
        
        for file_path in discovered_files:
            progress.update_file(file_path, "Loading")
            
            result = loader.load_document(file_path)
            if result is not None:
                filename, content = result
                total_characters += len(content)
                stats_update = {'total_characters': len(content), 'files_processed': 1}
                progress.complete_file(success=True, stats_update=stats_update)
            else:
                progress.complete_file(success=False)
        
        # Finish and show summary
        progress.finish()
        
        if progress.completed_items == 0:
            return 1
        
        return 0
        
    except Exception as e:
        print(f"Phase 1 Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def run_document_processing(args) -> int:
    """
    Run document processing pipeline (loading, chunking, embedding, and vectorization).
    
    Args:
        args: Parsed CLI arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Show configuration in verbose mode
        if args.verbose:
            print(f"DocuChat RAG System - Document Processing")
            print(f"Processing directory: {args.directory}")
            print(f"Chunk size: {args.chunk_size}")
            print(f"Top-k: {args.top_k}")
            print(f"Rebuild: {args.rebuild}")
            print(f"Show sources: {args.show_sources}")
            print(f"Chat mode: {args.chat}")
            print(f"Max context: {args.max_context}")
            print(f"Verbose mode: {args.verbose}")
            print("-" * 60)
        
        # Initialize all components (with quiet mode for most)
        loader = DocumentLoader(verbose=False)  # We'll manage progress ourselves
        
        # Display chunk size information
        print(args.get_chunk_size_info())
        
        chunker = DocumentChunker(
            chunk_size=args.chunk_size,
            chunk_overlap=200,  # Standard overlap
            verbose=False  # We'll manage progress ourselves
        )
        embedding_generator = EmbeddingGenerator(
            model_name=args.embedding_model,
            batch_size=32,
            verbose=False  # We'll manage progress ourselves
        )
        vector_db = VectorDatabase(
            persist_directory="./chroma",
            collection_name="docuchat_embeddings",
            verbose=False,  # We'll manage progress ourselves
            rebuild=args.rebuild
        )
        
        # First, discover all files to get total count
        discovered_files = loader.discover_files(args.directory)
        
        if not discovered_files:
            print("⚠️  No supported documents found in directory.")
            return 1
        
        # Initialize progress manager
        progress = create_progress_manager(
            total_items=len(discovered_files),
            operation="Processing Documents",
            verbose=args.verbose,
            quiet=False  # Always show progress bar unless explicitly disabled
        )
        
        start_time = time.time()
        
        # Thread-safe lock for progress updates
        progress_lock = threading.Lock()
        
        def process_single_document(file_path):
            """Process a single document through the complete pipeline."""
            try:
                # Update progress with current file (thread-safe)
                with progress_lock:
                    progress.update_file(file_path, "Loading document")
                
                # Load document (I/O bound - benefits from threading)
                result = loader.load_document(file_path)
                if result is None:
                    with progress_lock:
                        progress.log_error("Failed to load document", str(file_path))
                        progress.complete_file(success=False)
                    return False
                
                filename, content = result
                stats_update = {
                    'total_characters': len(content),
                    'files_processed': 1
                }
                
                # Step 2: Chunk the document (CPU bound but fast)
                with progress_lock:
                    progress.update_file(file_path, "Chunking")
                
                chunks = list(chunker.chunk_document(
                    text=content,
                    source_file=filename,
                    metadata={'processed_at': time.time()}
                ))
                
                if not chunks:
                    with progress_lock:
                        progress.log_verbose(f"No chunks created for {Path(filename).name}")
                        progress.complete_file(success=False)
                    return False
                
                stats_update['chunks_created'] = len(chunks)
                
                # Step 3: Generate embeddings (CPU bound - keep sequential per thread)
                with progress_lock:
                    progress.update_file(file_path, "Generating embeddings")
                
                embedded_chunks = embedding_generator.embed_chunks(chunks)
                
                if not embedded_chunks:
                    with progress_lock:
                        progress.log_error("No embeddings generated", str(file_path))
                        progress.complete_file(success=False, stats_update=stats_update)
                    return False
                
                stats_update['embeddings_generated'] = len(embedded_chunks)
                
                # Step 4: Store in vector database (I/O bound - benefits from threading)
                with progress_lock:
                    progress.update_file(file_path, "Storing in database")
                
                # Remove existing chunks for this file if rebuild is not global
                if not args.rebuild:
                    removed_count = vector_db.remove_source_file(filename)
                    if removed_count > 0:
                        with progress_lock:
                            progress.log_verbose(f"Removed {removed_count} existing chunks from {Path(filename).name}")
                
                # Add new chunks
                success = vector_db.add_chunks(embedded_chunks)
                
                if success:
                    stats_update['chunks_stored'] = len(embedded_chunks)
                    with progress_lock:
                        progress.complete_file(success=True, stats_update=stats_update)
                    return True
                else:
                    with progress_lock:
                        progress.log_error("Failed to store chunks", str(file_path))
                        progress.complete_file(success=False, stats_update=stats_update)
                    return False
                
            except Exception as e:
                with progress_lock:
                    progress.log_error(f"Processing error: {e}", str(file_path))
                    progress.complete_file(success=False)
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                return False
        
        # Process documents in parallel with optimal thread count
        # Use min(4, len(discovered_files)) to avoid over-threading for small batches
        max_workers = min(4, len(discovered_files))
        successful_files = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(process_single_document, file_path): file_path 
                for file_path in discovered_files
            }
            
            # Process completed tasks
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    success = future.result()
                    if success:
                        successful_files += 1
                except Exception as e:
                    with progress_lock:
                        progress.log_error(f"Unexpected error processing {file_path}: {e}", str(file_path))
        
        # Calculate total processing time
        processing_time = time.time() - start_time
        
        # Finish progress display and show summary
        progress.finish()
        
        # Show additional info in verbose mode
        if args.verbose:
            print(f"\n📊 Vector Database Information:")
            db_info = vector_db.get_info()
            for key, value in db_info.items():
                print(f"  {key}: {value}")
        
        # Close database connection
        vector_db.close()
        
        # Determine exit code based on results
        if progress.completed_items == 0:
            return 1  # No files processed successfully
        elif progress.failed_items > 0:
            return 1 if progress.completed_items == 0 else 0  # Some failures but some success
        else:
            return 0  # All successful
        
    except Exception as e:
        print(f"Document Processing Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


# Remove the old processing loop and summary code


def run_interactive_mode(args) -> int:
    """
    Run the complete RAG system with interactive chat.
    
    Args:
        args: Parsed CLI arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        if args.verbose:
            print(f"DocuChat RAG System - Interactive Mode")
            print(f"Database directory: {args.directory}")
            print(f"Top-k: {args.top_k}")
            print(f"Show sources: {args.show_sources}")
            print(f"Max context: {args.max_context}")
            print(f"Verbose mode: {args.verbose}")
            print("-" * 60)
        
        # Initialize all components
        if args.verbose:
            print("🔄 Initializing RAG pipeline components...")
        
        # Display chunk size information (even in non-verbose mode for chat)
        print(args.get_chunk_size_info())
        
        embedding_generator = EmbeddingGenerator(
            model_name=args.embedding_model,
            batch_size=32,
            verbose=args.verbose
        )
        
        vector_database = VectorDatabase(
            persist_directory="./chroma",
            collection_name="docuchat_embeddings",
            verbose=args.verbose,
            rebuild=False  # Don't rebuild in chat mode
        )
        
        llm_client = LLMClient(
            base_url="http://localhost:11434",
            model=args.llm,
            timeout=args.get_timeout_seconds(),
            verbose=args.verbose
        )
        
        # Test LLM availability
        if not llm_client.is_available():
            print(f"❌ LLM service not available. Please ensure Ollama is running with {args.llm} model.")
            print("   Start Ollama: ollama serve")
            print(f"   Pull model: ollama pull {args.llm}")
            return 1
        
        # Check if database has content
        db_info = vector_database.get_info()
        if db_info.get('document_count', 0) == 0:
            print("⚠️  No documents found in the vector database.")
            print("   Process documents first: python docuchat.py /path/to/documents")
            print("   Then run chat mode: python docuchat.py . --chat")
            return 1
        
        # Create RAG pipeline
        if args.verbose:
            print("🤖 Creating RAG pipeline...")
        
        rag_pipeline = RAGPipeline(
            embedding_generator=embedding_generator,
            vector_database=vector_database,
            llm_client=llm_client,
            verbose=args.verbose
        )
        
        # Test pipeline
        if args.verbose:
            print("🧪 Testing RAG pipeline...")
            if not rag_pipeline.test_pipeline():
                print("❌ RAG pipeline test failed")
                return 1
            print("✅ RAG pipeline test successful")
        
        # Display database info
        print(f"\n📊 Database Information:")
        print(f"  Documents: {db_info.get('document_count', 0):,}")
        if 'unique_source_files_sample' in db_info:
            print(f"  Source files (sample): {db_info['unique_source_files_sample']}")
        
        # Start interactive chat
        print(f"\n🚀 Starting interactive chat mode...")
        start_interactive_chat(rag_pipeline, args)
        
        # Cleanup
        rag_pipeline.close()
        
        return 0
        
    except Exception as e:
        print(f"Interactive Mode Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def main():
    """Main application entry point."""
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Check if core components are available
        if not CORE_COMPONENTS_AVAILABLE:
            print("❌ Core components not available.")
            print("   Install dependencies: pip install -r requirements.txt")
            print("   Required: torch transformers sentence-transformers chromadb requests")
            return 1
        
        # Determine mode to run
        if args.chat:
            # Run interactive chat mode
            return run_interactive_mode(args)
        else:
            # Run document processing pipeline
            result = run_document_processing(args)
            
            # If processing successful and chat requested, start chat
            if result == 0 and args.chat:
                print("\n🤖 Starting interactive chat mode...")
                return run_interactive_mode(args)
            
            return result
    
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user.")
        return 130
    
    except Exception as e:
        print(f"💥 Unexpected error: {e}", file=sys.stderr)
        # Try to determine if we have args for verbose mode
        try:
            args = parse_args()
            if hasattr(args, 'verbose') and args.verbose:
                import traceback
                traceback.print_exc()
        except:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())