#!/usr/bin/env python3
"""
Test script to verify error handling in the app.
This will test what happens when API clients fail to initialize.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_missing_api_keys():
    """Test what happens when API keys are missing."""
    print("🔍 Testing Missing API Keys Scenario...")
    print("=" * 50)
    
    # Check what's currently in .env
    print("📋 Current .env file contents:")
    env_vars = [
        'OPENAI_API_KEY',
        'AWS_ACCESS_KEY_ID', 
        'AWS_SECRET_ACCESS_KEY',
        'PINECONE_API_KEY'
    ]
    
    missing_vars = []
    for var in env_vars:
        value = os.getenv(var)
        if value:
            print(f"   ✅ {var}: {'*' * min(len(value), 8)}...")
        else:
            print(f"   ❌ {var}: NOT SET")
            missing_vars.append(var)
    
    print(f"\n📊 Summary: {len(env_vars) - len(missing_vars)}/{len(env_vars)} API keys configured")
    
    if missing_vars:
        print(f"\n❌ Missing API keys: {', '.join(missing_vars)}")
        print("\n💡 To fix this, add the missing keys to your .env file:")
        print("   cp env_template.txt .env")
        print("   # Then edit .env with your actual API keys")
    else:
        print("\n🎉 All API keys are configured!")
        print("   You can now run: streamlit run streamlit_app.py")
    
    return len(missing_vars) == 0

def test_api_initialization():
    """Test the API initialization function."""
    print("\n🔧 Testing API Client Initialization...")
    print("=" * 50)
    
    try:
        # Import the app to test initialization
        import streamlit_app
        
        # Try to initialize clients
        clients = streamlit_app.initialize_clients()
        
        if clients:
            print("✅ API clients initialized successfully!")
            print(f"   Returned {len(clients)} clients")
            return True
        else:
            print("❌ API client initialization failed")
            print("   This means there's an issue with your API keys or configuration")
            return False
            
    except Exception as e:
        print(f"❌ Error during API initialization: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Error Handling and API Configuration")
    print("=" * 60)
    
    # Test 1: Check API keys
    keys_ok = test_missing_api_keys()
    
    # Test 2: Test API initialization
    if keys_ok:
        init_ok = test_api_initialization()
    else:
        print("\n⏭️ Skipping API initialization test (missing keys)")
        init_ok = False
    
    print("\n" + "=" * 60)
    
    if keys_ok and init_ok:
        print("🎉 All tests passed! Your app is ready to run.")
        print("✅ Run: streamlit run streamlit_app.py")
    elif keys_ok and not init_ok:
        print("⚠️ API keys are configured but initialization failed.")
        print("📖 Check the error messages above for details.")
    else:
        print("❌ Configuration issues found.")
        print("📖 Please fix the missing API keys and try again.")
    
    return keys_ok and init_ok

if __name__ == "__main__":
    main()
