#!/usr/bin/env python3
"""
Test script for the performance logging system.
Verifies that timing works across all chat components.
"""

import sys
import time
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simple_timer import SimpleTimer, get_timer, reset_timer


def test_timer_basic_functionality():
    """Test basic timer functionality."""
    print("Testing basic timer functionality...")
    
    timer = SimpleTimer()
    
    # Test individual timing blocks
    with timer.time_block("question_embedding"):
        time.sleep(0.1)
    
    with timer.time_block("vector_search"):
        time.sleep(0.05)
    
    with timer.time_block("llm_generation"):
        time.sleep(0.2)
    
    results = timer.get_results()
    print(f"Total time: {results.total_time:.2f}s")
    print(f"Breakdown: {results.format_display()}")
    
    # Verify timing values are reasonable
    assert results.question_embedding > 0.09, f"Embedding time too low: {results.question_embedding}"
    assert results.vector_search > 0.04, f"Search time too low: {results.vector_search}"
    assert results.llm_generation > 0.19, f"Generation time too low: {results.llm_generation}"
    assert results.total_time > 0.34, f"Total time too low: {results.total_time}"
    
    print("✅ Basic timer test passed!")


def test_global_timer():
    """Test the global timer functionality."""
    print("\nTesting global timer...")
    
    reset_timer()
    timer = get_timer()
    
    # Simulate a chat interaction
    timer.start_block("input_processing")
    time.sleep(0.02)
    timer.end_block("input_processing")
    
    with timer.time_block("question_embedding"):
        time.sleep(0.03)
    
    with timer.time_block("context_preparation"):
        time.sleep(0.01)
    
    results = timer.get_results()
    formatted = timer.format_for_display()
    
    print(f"Global timer results: {formatted}")
    assert "Input" in formatted, "Input timing not in display"
    assert "Embedding" in formatted, "Embedding timing not in display"
    
    print("✅ Global timer test passed!")


def test_chat_simulation():
    """Simulate a complete chat interaction with timing."""
    print("\nSimulating complete chat interaction...")
    
    reset_timer()
    timer = get_timer()
    
    # Simulate the complete chat flow
    steps = [
        ("input_processing", 0.01),
        ("question_embedding", 0.05),
        ("vector_search", 0.03),
        ("context_preparation", 0.02),
        ("llm_generation", 0.15),
        ("response_display", 0.01)
    ]
    
    for step_name, duration in steps:
        with timer.time_block(step_name):
            time.sleep(duration)
    
    results = timer.get_results()
    breakdown = results.get_breakdown()
    
    print(f"Chat simulation results:")
    print(f"Total time: {results.total_time:.2f}s")
    for step, timing in breakdown.items():
        if timing > 0:
            percentage = (timing / results.total_time) * 100
            print(f"  {step.title()}: {timing:.3f}s ({percentage:.1f}%)")
    
    # Verify all steps were timed
    assert all(breakdown[step.split('_')[0]] > 0 for step, _ in steps if step.split('_')[0] in breakdown), "Missing timing data"
    
    print("✅ Chat simulation test passed!")


def test_performance_display():
    """Test performance display formatting."""
    print("\nTesting performance display formatting...")
    
    timer = SimpleTimer()
    
    # Add some test timings
    timer.current_timings.question_embedding = 0.1
    timer.current_timings.vector_search = 0.2
    timer.current_timings.llm_generation = 0.5
    
    display = timer.format_for_display()
    print(f"Display format: {display}")
    
    # Verify display contains expected elements
    assert "[0.8s total]" in display, "Total time not in display"
    assert "Embedding: 0.1s" in display, "Embedding time not in display"
    assert "Generation: 0.5s" in display, "Generation time not in display"
    
    print("✅ Performance display test passed!")


def main():
    """Run all performance logging tests."""
    print("🧪 Testing DocuChat Performance Logging System")
    print("=" * 50)
    
    try:
        test_timer_basic_functionality()
        test_global_timer()
        test_chat_simulation()
        test_performance_display()
        
        print("\n" + "=" * 50)
        print("🎉 All performance logging tests passed!")
        print("✅ The system is ready to measure chat performance")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())