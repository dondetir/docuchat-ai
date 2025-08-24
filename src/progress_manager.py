"""
Professional progress management system for DocuChat.
Provides clean, unified progress reporting across all components.
"""

import sys
import time
import threading
from typing import Optional, Dict, Any
from pathlib import Path


class ProgressManager:
    """Thread-safe progress manager with professional console output."""
    
    def __init__(self, total_items: int, operation: str = "Processing", verbose: bool = False):
        """
        Initialize progress manager.
        
        Args:
            total_items: Total number of items to process
            operation: Description of the operation being performed
            verbose: Enable detailed logging
        """
        self.total_items = total_items
        self.operation = operation
        self.verbose = verbose
        
        # Progress tracking
        self.current_item = 0
        self.completed_items = 0
        self.failed_items = 0
        self.current_file = ""
        self.current_step = ""
        
        # Timing
        self.start_time = time.time()
        self.last_update_time = 0
        self.update_interval = 0.1  # Minimum seconds between updates
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'total_characters': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'chunks_stored': 0,
            'errors': 0
        }
        
        # Initialize display
        self._initialize_display()
    
    def _initialize_display(self):
        """Initialize the progress display."""
        if self.total_items > 0:
            print(f"\n{self.operation}: {self.total_items} files")
            print("─" * 60)
            self._draw_progress()
    
    def _draw_progress(self):
        """Draw the progress bar and current status."""
        current_time = time.time()
        
        # Skip update if too frequent
        if current_time - self.last_update_time < self.update_interval:
            return
            
        self.last_update_time = current_time
        
        # Calculate progress
        progress = self.completed_items / max(self.total_items, 1)
        bar_width = 40
        filled = int(bar_width * progress)
        
        # Create progress bar
        bar = "█" * filled + "░" * (bar_width - filled)
        percentage = progress * 100
        
        # Calculate timing
        elapsed = current_time - self.start_time
        if progress > 0:
            eta = (elapsed / progress) * (1 - progress)
            eta_str = f"{int(eta//60):02d}:{int(eta%60):02d}"
        else:
            eta_str = "--:--"
        
        # Current file display (truncate if too long)
        current_display = self.current_file
        if len(current_display) > 45:
            current_display = "..." + current_display[-42:]
        
        # Clear previous lines and draw new progress
        print(f"\r\033[K[{bar}] {percentage:5.1f}% | {self.completed_items}/{self.total_items}", end="")
        if self.failed_items > 0:
            print(f" | ❌ {self.failed_items} errors", end="")
        print(f" | ETA: {eta_str}")
        
        if current_display:
            print(f"\r\033[K📄 {current_display}", end="")
            if self.current_step:
                print(f" | {self.current_step}", end="")
        
        print(flush=True)
    
    def update_file(self, filename: str, step: str = ""):
        """
        Update current file being processed.
        
        Args:
            filename: Name or path of the current file
            step: Current processing step
        """
        with self._lock:
            if isinstance(filename, Path):
                self.current_file = filename.name
            else:
                self.current_file = Path(filename).name
            self.current_step = step
            self._draw_progress()
    
    def complete_file(self, success: bool = True, stats_update: Optional[Dict[str, Any]] = None):
        """
        Mark current file as completed.
        
        Args:
            success: Whether the file was processed successfully
            stats_update: Dictionary of statistics to update
        """
        with self._lock:
            if success:
                self.completed_items += 1
                if self.verbose:
                    print(f"\n✅ Completed: {self.current_file}")
            else:
                self.failed_items += 1
                if self.verbose:
                    print(f"\n❌ Failed: {self.current_file}")
            
            # Update statistics
            if stats_update:
                for key, value in stats_update.items():
                    if key in self.stats:
                        self.stats[key] += value
            
            self._draw_progress()
    
    def log_error(self, message: str, filename: str = ""):
        """
        Log an error message.
        
        Args:
            message: Error message
            filename: Optional filename where error occurred
        """
        with self._lock:
            if filename:
                error_msg = f"❌ {Path(filename).name}: {message}"
            else:
                error_msg = f"❌ {message}"
            
            if self.verbose:
                print(f"\n{error_msg}")
            
            self.stats['errors'] += 1
    
    def log_verbose(self, message: str):
        """
        Log a verbose message (only shown in verbose mode).
        
        Args:
            message: Message to log
        """
        if self.verbose:
            with self._lock:
                print(f"\n💬 {message}")
    
    def finish(self):
        """Complete the progress display and show summary."""
        with self._lock:
            # Final progress update
            self.completed_items = min(self.completed_items, self.total_items)
            self._draw_progress()
            
            # Calculate final statistics
            elapsed = time.time() - self.start_time
            success_rate = (self.completed_items / max(self.total_items, 1)) * 100
            
            print(f"\n\n{'=' * 60}")
            print(f"PROCESSING SUMMARY")
            print(f"{'=' * 60}")
            print(f"✅ Completed: {self.completed_items}/{self.total_items} ({success_rate:.1f}%)")
            if self.failed_items > 0:
                print(f"❌ Failed: {self.failed_items}")
            print(f"⏱️  Total time: {elapsed:.2f}s")
            
            if self.completed_items > 0:
                avg_time = elapsed / self.completed_items
                print(f"📈 Average: {avg_time:.2f}s per file")
            
            # Show detailed statistics if available
            if any(self.stats.values()):
                print(f"\n📊 DETAILED STATISTICS:")
                if self.stats['total_characters'] > 0:
                    print(f"📝 Characters processed: {self.stats['total_characters']:,}")
                if self.stats['chunks_created'] > 0:
                    print(f"✂️  Chunks created: {self.stats['chunks_created']:,}")
                if self.stats['embeddings_generated'] > 0:
                    print(f"🧠 Embeddings generated: {self.stats['embeddings_generated']:,}")
                if self.stats['chunks_stored'] > 0:
                    print(f"💾 Chunks stored: {self.stats['chunks_stored']:,}")
            
            print()


class QuietProgressManager:
    """Minimal progress manager for non-verbose mode."""
    
    def __init__(self, total_items: int, operation: str = "Processing", verbose: bool = False):
        self.total_items = total_items
        self.operation = operation
        self.completed_items = 0
        self.failed_items = 0
        self.start_time = time.time()
        self.stats = {
            'files_processed': 0,
            'total_characters': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'chunks_stored': 0,
            'errors': 0
        }
        
        if total_items > 0:
            print(f"{operation} {total_items} files...")
    
    def update_file(self, filename: str, step: str = ""):
        """Update current file (no display in quiet mode)."""
        pass
    
    def complete_file(self, success: bool = True, stats_update: Optional[Dict[str, Any]] = None):
        """Mark file as completed."""
        if success:
            self.completed_items += 1
        else:
            self.failed_items += 1
        
        if stats_update:
            for key, value in stats_update.items():
                if key in self.stats:
                    self.stats[key] += value
    
    def log_error(self, message: str, filename: str = ""):
        """Log error (always shown)."""
        if filename:
            print(f"❌ {Path(filename).name}: {message}")
        else:
            print(f"❌ {message}")
        self.stats['errors'] += 1
    
    def log_verbose(self, message: str):
        """Log verbose message (ignored in quiet mode)."""
        pass
    
    def finish(self):
        """Show final summary."""
        elapsed = time.time() - self.start_time
        success_rate = (self.completed_items / max(self.total_items, 1)) * 100
        
        print(f"✅ Completed {self.completed_items}/{self.total_items} files ({success_rate:.1f}%) in {elapsed:.1f}s")
        if self.failed_items > 0:
            print(f"❌ {self.failed_items} files failed")


def create_progress_manager(total_items: int, operation: str = "Processing", 
                          verbose: bool = False, quiet: bool = False, 
                          parallel: bool = False, num_workers: int = 4) -> ProgressManager:
    """
    Factory function to create appropriate progress manager.
    
    Args:
        total_items: Total number of items to process
        operation: Description of operation
        verbose: Enable verbose output with detailed progress
        quiet: Use minimal progress display
        parallel: Enable parallel processing mode
        num_workers: Number of parallel workers (only used if parallel=True)
        
    Returns:
        Progress manager instance
    """
    if parallel:
        return ParallelProgressManager(total_items, num_workers, operation, verbose)
    elif quiet:
        return QuietProgressManager(total_items, operation, verbose)
    else:
        return ProgressManager(total_items, operation, verbose)


# Parallel processing support
class ParallelProgressManager:
    """Progress manager for parallel processing."""
    
    def __init__(self, total_items: int, num_workers: int = 4, 
                 operation: str = "Processing", verbose: bool = False):
        """
        Initialize parallel progress manager.
        
        Args:
            total_items: Total items to process
            num_workers: Number of parallel workers
            operation: Operation description
            verbose: Enable verbose output
        """
        self.total_items = total_items
        self.num_workers = num_workers
        self.operation = operation
        self.verbose = verbose
        
        # Per-worker progress tracking
        self.worker_files = [""] * num_workers
        self.worker_steps = [""] * num_workers
        
        # Global statistics
        self.completed_items = 0
        self.failed_items = 0
        self.start_time = time.time()
        self.last_update_time = 0
        self.update_interval = 0.2  # Update every 200ms for parallel
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'total_characters': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'chunks_stored': 0,
            'errors': 0
        }
        
        print(f"\n{operation}: {total_items} files ({num_workers} workers)")
        print("─" * 60)
        self._draw_parallel_progress()
    
    def _draw_parallel_progress(self):
        """Draw parallel progress display."""
        current_time = time.time()
        
        # Skip update if too frequent
        if current_time - self.last_update_time < self.update_interval:
            return
        
        self.last_update_time = current_time
        
        # Calculate overall progress
        progress = self.completed_items / max(self.total_items, 1)
        bar_width = 30
        filled = int(bar_width * progress)
        
        # Create overall progress bar
        bar = "█" * filled + "░" * (bar_width - filled)
        percentage = progress * 100
        
        # Calculate timing
        elapsed = current_time - self.start_time
        if progress > 0:
            eta = (elapsed / progress) * (1 - progress)
            eta_str = f"{int(eta//60):02d}:{int(eta%60):02d}"
        else:
            eta_str = "--:--"
        
        # Clear previous display and show overall progress
        print(f"\r\033[K[{bar}] {percentage:5.1f}% | {self.completed_items}/{self.total_items}", end="")
        if self.failed_items > 0:
            print(f" | ❌ {self.failed_items}", end="")
        print(f" | ETA: {eta_str}")
        
        # Show worker status
        for i in range(min(4, self.num_workers)):  # Show max 4 workers
            worker_file = self.worker_files[i]
            worker_step = self.worker_steps[i]
            
            if worker_file:
                # Truncate long filenames
                display_file = worker_file
                if len(display_file) > 30:
                    display_file = "..." + display_file[-27:]
                
                status = f"Worker {i+1}: {display_file}"
                if worker_step:
                    status += f" | {worker_step}"
                
                print(f"\r\033[K{status}")
            elif i < self.num_workers:
                print(f"\r\033[KWorker {i+1}: Idle")
        
        print(flush=True)
    
    def update_worker(self, worker_id: int, filename: str, step: str = ""):
        """Update progress for specific worker."""
        with self._lock:
            if worker_id < len(self.worker_files):
                self.worker_files[worker_id] = Path(filename).name if filename else ""
                self.worker_steps[worker_id] = step
                self._draw_parallel_progress()
    
    def complete_worker_file(self, worker_id: int, success: bool = True, 
                           stats_update: Optional[Dict[str, Any]] = None):
        """Mark file as completed by specific worker."""
        with self._lock:
            if success:
                self.completed_items += 1
            else:
                self.failed_items += 1
                self.stats['errors'] += 1
            
            # Update statistics
            if stats_update:
                for key, value in stats_update.items():
                    if key in self.stats:
                        self.stats[key] += value
            
            # Clear worker status
            if worker_id < len(self.worker_files):
                self.worker_files[worker_id] = ""
                self.worker_steps[worker_id] = ""
            
            self._draw_parallel_progress()
    
    def log_error(self, message: str, filename: str = "", worker_id: int = -1):
        """Log an error from a worker."""
        with self._lock:
            if worker_id >= 0:
                error_msg = f"❌ Worker {worker_id+1}: {message}"
            elif filename:
                error_msg = f"❌ {Path(filename).name}: {message}"
            else:
                error_msg = f"❌ {message}"
            
            if self.verbose:
                print(f"\n{error_msg}")
            
            self.stats['errors'] += 1
    
    def log_verbose(self, message: str, worker_id: int = -1):
        """Log a verbose message from a worker."""
        if self.verbose:
            with self._lock:
                if worker_id >= 0:
                    print(f"\n💬 Worker {worker_id+1}: {message}")
                else:
                    print(f"\n💬 {message}")
    
    def finish(self):
        """Complete parallel processing and show summary."""
        with self._lock:
            # Final progress update
            self._draw_parallel_progress()
            
            elapsed = time.time() - self.start_time
            success_rate = (self.completed_items / max(self.total_items, 1)) * 100
            
            print(f"\n\n{'=' * 60}")
            print(f"PARALLEL PROCESSING SUMMARY")
            print(f"{'=' * 60}")
            print(f"✅ Completed: {self.completed_items}/{self.total_items} ({success_rate:.1f}%)")
            if self.failed_items > 0:
                print(f"❌ Failed: {self.failed_items}")
            print(f"⏱️  Total time: {elapsed:.2f}s with {self.num_workers} workers")
            
            if self.completed_items > 0:
                throughput = self.completed_items / elapsed
                print(f"🚀 Throughput: {throughput:.1f} files/second")
                print(f"📈 Speedup: ~{throughput * self.num_workers / max(throughput, 1):.1f}x potential")
            
            # Show detailed statistics if available
            if any(self.stats.values()):
                print(f"\n📊 DETAILED STATISTICS:")
                if self.stats['total_characters'] > 0:
                    print(f"📝 Characters processed: {self.stats['total_characters']:,}")
                if self.stats['chunks_created'] > 0:
                    print(f"✂️  Chunks created: {self.stats['chunks_created']:,}")
                if self.stats['embeddings_generated'] > 0:
                    print(f"🧠 Embeddings generated: {self.stats['embeddings_generated']:,}")
                if self.stats['chunks_stored'] > 0:
                    print(f"💾 Chunks stored: {self.stats['chunks_stored']:,}")
            
            print()


# Example of how to use ParallelProgressManager in the future
def example_parallel_processing(files, num_workers=4):
    """
    Example of how parallel processing would work with the progress manager.
    This is for future implementation.
    """
    import concurrent.futures
    from threading import Thread
    
    progress = ParallelProgressManager(
        total_items=len(files),
        num_workers=num_workers,
        operation="Processing Documents (Parallel)",
        verbose=True
    )
    
    def process_file_worker(worker_id, file_path):
        """Worker function to process a single file."""
        try:
            progress.update_worker(worker_id, file_path, "Loading")
            # Simulate file loading
            time.sleep(0.1)
            
            progress.update_worker(worker_id, file_path, "Chunking")
            # Simulate chunking
            time.sleep(0.2)
            
            progress.update_worker(worker_id, file_path, "Embedding")
            # Simulate embedding
            time.sleep(0.3)
            
            progress.update_worker(worker_id, file_path, "Storing")
            # Simulate storing
            time.sleep(0.1)
            
            # Mark as completed
            stats_update = {
                'files_processed': 1,
                'total_characters': 1000,  # Example
                'chunks_created': 10,
                'embeddings_generated': 10,
                'chunks_stored': 10
            }
            progress.complete_worker_file(worker_id, success=True, stats_update=stats_update)
            
        except Exception as e:
            progress.log_error(f"Processing failed: {e}", file_path, worker_id)
            progress.complete_worker_file(worker_id, success=False)
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all files to workers
        future_to_file = {}
        for i, file_path in enumerate(files):
            worker_id = i % num_workers
            future = executor.submit(process_file_worker, worker_id, file_path)
            future_to_file[future] = file_path
        
        # Wait for completion
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                future.result()
            except Exception as e:
                progress.log_error(f"Worker error: {e}", file_path)
    
    progress.finish()