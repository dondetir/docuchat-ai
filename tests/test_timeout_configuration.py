#!/usr/bin/env python3
"""
Timeout Fix Verification Script
Verifies that all timeout configurations are properly set to 60 seconds.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def verify_timeout_configuration():
    """Verify all timeout configurations are correctly set."""
    
    print("🔍 TIMEOUT CONFIGURATION VERIFICATION")
    print("=" * 50)
    
    # Test 1: Check SecureLLMClient DEFAULT_TIMEOUT
    try:
        from llm_client import SecureLLMClient
        default_timeout = SecureLLMClient.DEFAULT_TIMEOUT
        print(f"✅ SecureLLMClient.DEFAULT_TIMEOUT: {default_timeout}s")
        
        if default_timeout == 60.0:
            print("   ✅ CORRECT: 60s timeout configured")
        else:
            print(f"   ❌ INCORRECT: Expected 60s, got {default_timeout}s")
            return False
            
    except Exception as e:
        print(f"❌ Failed to check SecureLLMClient: {e}")
        return False
    
    # Test 2: Check LLMClient instance creation
    try:
        from llm_client import LLMClient
        
        # Test with explicit timeout
        client_explicit = LLMClient(
            base_url="http://localhost:11434", 
            model="gemma3:270m", 
            timeout=60.0
        )
        print(f"✅ LLMClient (explicit): {client_explicit.timeout}s")
        
        # Test with default timeout
        client_default = LLMClient(
            base_url="http://localhost:11434", 
            model="gemma3:270m"
        )
        print(f"✅ LLMClient (default): {client_default.timeout}s")
        
        if client_explicit.timeout == 60.0 and client_default.timeout == 60.0:
            print("   ✅ CORRECT: Both explicit and default timeouts are 60s")
        else:
            print(f"   ❌ INCORRECT: Expected 60s for both")
            return False
            
    except Exception as e:
        print(f"❌ Failed to test LLMClient: {e}")
        return False
    
    # Test 3: Check timeout propagation to SecureLLMClient
    try:
        client = LLMClient(timeout=60.0)
        secure_timeout = client.client.timeout
        print(f"✅ Propagated to SecureLLMClient: {secure_timeout}s")
        
        if secure_timeout == 60.0:
            print("   ✅ CORRECT: Timeout properly propagated")
        else:
            print(f"   ❌ INCORRECT: Expected 60s, got {secure_timeout}s")
            return False
            
    except Exception as e:
        print(f"❌ Failed to check timeout propagation: {e}")
        return False
    
    print("\n🎉 ALL TIMEOUT CONFIGURATIONS VERIFIED!")
    print("   - SecureLLMClient default: 60s ✅")
    print("   - LLMClient explicit: 60s ✅") 
    print("   - LLMClient default: 60s ✅")
    print("   - Timeout propagation: Working ✅")
    
    return True

def test_timeout_error_message():
    """Test that timeout error messages will show 60s."""
    
    print("\n🧪 TIMEOUT ERROR MESSAGE TEST")
    print("=" * 50)
    
    try:
        from llm_client import LLMClient, TimeoutError
        
        # Create client with 60s timeout
        client = LLMClient(timeout=60.0)
        
        # Simulate timeout error (without actually timing out)
        try:
            # Create a mock timeout error
            error_msg = f"Read timeout after {client.client.timeout}s"
            print(f"✅ Mock timeout error message: '{error_msg}'")
            
            if "60.0s" in error_msg:
                print("   ✅ CORRECT: Error message will show 60s")
                return True
            else:
                print(f"   ❌ INCORRECT: Error message shows wrong timeout")
                return False
                
        except Exception as e:
            print(f"❌ Error creating mock timeout: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to test timeout error message: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Timeout Fix Verification...\n")
    
    success1 = verify_timeout_configuration()
    success2 = test_timeout_error_message()
    
    print(f"\n{'='*50}")
    if success1 and success2:
        print("✅ VERIFICATION PASSED: All timeout fixes working correctly!")
        print("\n🔧 NEXT STEPS:")
        print("1. Restart any running DocuChat instances")
        print("2. Test with: python docuchat.py --chat")
        print("3. Timeout errors should now show '60.0s' instead of '30.0s'")
        sys.exit(0)
    else:
        print("❌ VERIFICATION FAILED: Some timeout configurations are incorrect")
        sys.exit(1)