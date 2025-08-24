"""
Simple timer utility for measuring performance blocks in DocuChat.
Provides easy-to-use timing functionality with minimal overhead.
"""

import time
from contextlib import contextmanager
from typing import Dict, Optional
from dataclasses import dataclass, field


@dataclass
class TimingResult:
    """Container for timing results from a chat interaction."""
    input_processing: float = 0.0
    question_embedding: float = 0.0
    vector_search: float = 0.0
    context_preparation: float = 0.0
    llm_generation: float = 0.0
    response_display: float = 0.0
    total_time: float = 0.0
    
    def get_breakdown(self) -> Dict[str, float]:
        """Get timing breakdown as dictionary."""
        return {
            "input": self.input_processing,
            "embedding": self.question_embedding,
            "search": self.vector_search,
            "context": self.context_preparation,
            "generation": self.llm_generation,
            "display": self.response_display
        }
    
    def format_display(self) -> str:
        """Format timings for user display."""
        breakdown = self.get_breakdown()
        timing_parts = []
        
        for name, duration in breakdown.items():
            if duration > 0:
                timing_parts.append(f"{name.title()}: {duration:.1f}s")
        
        if timing_parts:
            return " | ".join(timing_parts)
        else:
            return f"Total: {self.total_time:.1f}s"


class SimpleTimer:
    """
    Simple timer for measuring performance blocks with minimal overhead.
    Thread-safe and designed for chat interaction timing.
    """
    
    def __init__(self):
        """Initialize the timer with empty results."""
        self.current_timings = TimingResult()
        self._start_times = {}
        self._session_start = time.time()
    
    @contextmanager
    def time_block(self, block_name: str):
        """
        Context manager for timing code blocks.
        
        Args:
            block_name: Name of the block being timed
            
        Usage:
            with timer.time_block("embedding"):
                # code to time
                result = embed_question()
        """
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            self._record_timing(block_name, duration)
    
    def start_block(self, block_name: str):
        """
        Start timing a named block.
        
        Args:
            block_name: Name of the block to start timing
        """
        self._start_times[block_name] = time.time()
    
    def end_block(self, block_name: str):
        """
        End timing a named block and record the duration.
        
        Args:
            block_name: Name of the block to end timing
            
        Returns:
            Duration in seconds
        """
        if block_name not in self._start_times:
            return 0.0
        
        end_time = time.time()
        start_time = self._start_times.pop(block_name)
        duration = end_time - start_time
        self._record_timing(block_name, duration)
        return duration
    
    def _record_timing(self, block_name: str, duration: float):
        """Record timing for a specific block."""
        # Map block names to TimingResult fields
        if block_name == "input_processing":
            self.current_timings.input_processing = duration
        elif block_name == "question_embedding":
            self.current_timings.question_embedding = duration
        elif block_name == "vector_search":
            self.current_timings.vector_search = duration
        elif block_name == "context_preparation":
            self.current_timings.context_preparation = duration
        elif block_name == "llm_generation":
            self.current_timings.llm_generation = duration
        elif block_name == "response_display":
            self.current_timings.response_display = duration
    
    def get_results(self) -> TimingResult:
        """
        Get current timing results and calculate total.
        
        Returns:
            TimingResult with all recorded timings
        """
        # Calculate total time
        breakdown = self.current_timings.get_breakdown()
        self.current_timings.total_time = sum(breakdown.values())
        
        return self.current_timings
    
    def reset(self):
        """Reset all timings for a new chat interaction."""
        self.current_timings = TimingResult()
        self._start_times.clear()
        self._session_start = time.time()
    
    def format_for_display(self, show_total: bool = True) -> str:
        """
        Format current timings for user display.
        
        Args:
            show_total: Whether to show total time
            
        Returns:
            Formatted timing string
        """
        results = self.get_results()
        display = results.format_display()
        
        if show_total and results.total_time > 0:
            return f"[{results.total_time:.1f}s total] {display}"
        else:
            return display


# Global timer instance for easy access
_global_timer = SimpleTimer()


def get_timer() -> SimpleTimer:
    """Get the global timer instance."""
    return _global_timer


def reset_timer():
    """Reset the global timer."""
    _global_timer.reset()


@contextmanager  
def time_chat_block(block_name: str):
    """Convenience function for timing chat blocks using global timer."""
    with _global_timer.time_block(block_name):
        yield


def get_chat_timings() -> TimingResult:
    """Get current chat timing results."""
    return _global_timer.get_results()


def format_chat_timings() -> str:
    """Format current chat timings for display."""
    return _global_timer.format_for_display()