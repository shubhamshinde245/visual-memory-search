#!/usr/bin/env python3
"""
Pinecone Index Migration Utility
This script helps migrate your existing Pinecone index to work with new embedding dimensions.
"""

import os
import pinecone
from dotenv import load_dotenv
import json
from openai_config import EMBEDDING_MODELS, get_model_recommendations

# Load environment variables
load_dotenv()

def get_index_info():
    """Get information about the current Pinecone index."""
    try:
        api_key = os.getenv('PINECONE_API_KEY')
        if not api_key:
            print("❌ PINECONE_API_KEY not found in .env file")
            return None
        
        pc = pinecone.Pinecone(api_key=api_key)
        index_name = os.getenv('PINECONE_INDEX_NAME', 'visual-memory-search')
        
        try:
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            return {
                'name': index_name,
                'dimensions': stats.dimension,
                'total_vectors': stats.total_vector_count,
                'index': index
            }
        except Exception as e:
            print(f"❌ Failed to connect to index '{index_name}': {str(e)}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def migrate_index():
    """Migrate the index to handle dimension changes."""
    print("🔍 Pinecone Index Migration Utility")
    print("=" * 50)
    
    # Get current index info
    index_info = get_index_info()
    if not index_info:
        return False
    
    print(f"📊 Current Index Information:")
    print(f"   Name: {index_info['name']}")
    print(f"   Dimensions: {index_info['dimensions']}")
    print(f"   Total Vectors: {index_info['total_vectors']}")
    
    # Check for dimension mismatches
    print(f"\n🔍 Checking for dimension mismatches...")
    
    # Get current model from environment
    current_model = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-large')
    expected_dimensions = EMBEDDING_MODELS[current_model]["dimensions"]
    
    print(f"   Current Model: {current_model}")
    print(f"   Expected Dimensions: {expected_dimensions}")
    
    if index_info['dimensions'] == expected_dimensions:
        print("✅ No dimension mismatch detected!")
        print("   Your index is compatible with the current model.")
        return True
    
    print(f"⚠️  Dimension mismatch detected!")
    print(f"   Current: {index_info['dimensions']}")
    print(f"   Required: {expected_dimensions}")
    
    # Provide migration options
    print(f"\n🔧 Migration Options:")
    print(f"1. Use compatible model (recommended)")
    print(f"2. Delete and recreate index (data will be lost)")
    print(f"3. Keep current setup")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        # Option 1: Use compatible model
        if index_info['dimensions'] == 1536:
            compatible_model = "text-embedding-3-small"
        elif index_info['dimensions'] == 3072:
            compatible_model = "text-embedding-3-large"
        else:
            print(f"❌ Unknown dimension {index_info['dimensions']}")
            return False
        
        print(f"\n🔄 Switching to compatible model: {compatible_model}")
        
        # Update .env file
        env_file = ".env"
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                content = f.read()
            
            # Replace or add the model setting
            if 'OPENAI_EMBEDDING_MODEL=' in content:
                content = content.replace(
                    f'OPENAI_EMBEDDING_MODEL={current_model}',
                    f'OPENAI_EMBEDDING_MODEL={compatible_model}'
                )
            else:
                content += f'\nOPENAI_EMBEDDING_MODEL={compatible_model}'
            
            with open(env_file, 'w') as f:
                f.write(content)
            
            print(f"✅ Updated .env file with OPENAI_EMBEDDING_MODEL={compatible_model}")
            print(f"   Please restart your application for changes to take effect.")
        else:
            print(f"⚠️  .env file not found. Please manually add:")
            print(f"   OPENAI_EMBEDDING_MODEL={compatible_model}")
        
        return True
    
    elif choice == "2":
        # Option 2: Delete and recreate
        print(f"\n🗑️  Deleting current index...")
        
        try:
            pc = pinecone.Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            pc.delete_index(index_info['name'])
            print(f"✅ Index '{index_info['name']}' deleted successfully!")
            print(f"   Please restart your application to create a new index.")
            return True
        except Exception as e:
            print(f"❌ Failed to delete index: {str(e)}")
            return False
    
    elif choice == "3":
        # Option 3: Keep current setup
        print(f"\nℹ️  Keeping current setup.")
        print(f"   Note: You may encounter dimension mismatch errors.")
        return True
    
    else:
        print(f"❌ Invalid choice. Please run the script again.")
        return False

def show_recommendations():
    """Show model recommendations based on current setup."""
    print(f"\n💡 Model Recommendations:")
    print(f"=" * 30)
    
    recommendations = get_model_recommendations("production", "high")
    print(f"Production (High Quality):")
    print(f"   Embedding: {recommendations['embedding']}")
    print(f"   Vision: {recommendations['vision']}")
    print(f"   Reason: {recommendations['reason']}")
    
    recommendations = get_model_recommendations("cost_optimized")
    print(f"\nCost Optimized:")
    print(f"   Embedding: {recommendations['embedding']}")
    print(f"   Vision: {recommendations['vision']}")
    print(f"   Reason: {recommendations['reason']}")

def main():
    """Main migration function."""
    print("🚀 Pinecone Index Migration Utility")
    print("This utility helps resolve dimension mismatches between your Pinecone index and OpenAI models.")
    print()
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("   Please create a .env file with your API keys first.")
        return
    
    # Check required environment variables
    required_vars = ['PINECONE_API_KEY', 'OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("   Please add them to your .env file.")
        return
    
    # Run migration
    success = migrate_index()
    
    if success:
        print(f"\n✅ Migration completed successfully!")
        show_recommendations()
    else:
        print(f"\n❌ Migration failed. Please check the errors above.")

if __name__ == "__main__":
    main()
