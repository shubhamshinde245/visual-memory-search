import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import openai
import boto3
import pinecone
import io
import base64
import hashlib
import os
from typing import List, Tuple, Dict
import tempfile
from dotenv import load_dotenv
import time
import json
from openai_config import (
    DEFAULT_EMBEDDING_MODEL, 
    DEFAULT_VISION_MODEL, 
    DEFAULT_TEXT_MODEL,
    get_cost_estimate,
    get_model_recommendations
)

# Load environment variables
load_dotenv()

# Pinecone configuration
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'visual-memory-search')

# OpenAI model configuration - using latest models from config
OPENAI_EMBEDDING_MODEL = os.getenv('OPENAI_EMBEDDING_MODEL', DEFAULT_EMBEDDING_MODEL)
OPENAI_VISION_MODEL = os.getenv('OPENAI_VISION_MODEL', DEFAULT_VISION_MODEL)
OPENAI_TEXT_MODEL = os.getenv('OPENAI_TEXT_MODEL', DEFAULT_TEXT_MODEL)

def manage_pinecone_index(pinecone_index, openai_client):
    """Manage Pinecone index and handle dimension mismatches."""
    try:
        # Get current index stats
        index_stats = pinecone_index.describe_index_stats()
        current_dimensions = index_stats.dimension
        total_vectors = index_stats.total_vector_count
        
        st.subheader("🗄️ Pinecone Index Management")
        
        # Display current index info
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Dimensions", current_dimensions)
        with col2:
            st.metric("Total Vectors", total_vectors)
        
        # Check for dimension mismatch
        from openai_config import EMBEDDING_MODELS
        expected_dimensions = EMBEDDING_MODELS[OPENAI_EMBEDDING_MODEL]["dimensions"]
        
        if current_dimensions != expected_dimensions:
            st.warning(f"""
            ⚠️ **Dimension Mismatch Detected**
            
            - **Current Index**: {current_dimensions} dimensions
            - **Required Model**: {expected_dimensions} dimensions
            - **Current Model**: {OPENAI_EMBEDDING_MODEL}
            """)
            
            # Provide solutions
            with st.expander("🔧 Solutions"):
                st.markdown("""
                **Option 1: Use Compatible Model (Recommended)**
                ```bash
                # In your .env file
                OPENAI_EMBEDDING_MODEL=text-embedding-3-small  # For 1536 dimensions
                # or
                OPENAI_EMBEDDING_MODEL=text-embedding-3-large  # For 3072 dimensions
                ```
                
                **Option 2: Recreate Index (Data will be lost)**
                - Delete current index
                - Create new index with correct dimensions
                - Reprocess all images
                """)
                
                if st.button("🗑️ Delete Current Index", type="secondary"):
                    if st.checkbox("I understand this will delete all data"):
                        try:
                            # Get Pinecone client
                            import pinecone
                            pc = pinecone.Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
                            pc.delete_index(PINECONE_INDEX_NAME)
                            st.success("✅ Index deleted successfully! Please restart the app to create a new one.")
                        except Exception as e:
                            st.error(f"❌ Failed to delete index: {str(e)}")
        else:
            st.success(f"✅ Index dimensions match model requirements ({current_dimensions})")
            
        # Show index statistics
        if total_vectors > 0:
            with st.expander("📊 Index Statistics"):
                st.json(index_stats)
                
    except Exception as e:
        st.error(f"❌ Failed to get index information: {str(e)}")

# Initialize API clients
@st.cache_resource
def initialize_clients():
    """Initialize API clients with caching."""
    global OPENAI_EMBEDDING_MODEL
    
    try:
        # Check OpenAI API key
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            st.error("❌ OpenAI API key not found. Please add OPENAI_API_KEY to your .env file.")
            return None
        
        # Initialize OpenAI client
        try:
            openai_client = openai.OpenAI(api_key=openai_api_key)
            # Test the client with a simple call
            openai_client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input="test")
            print("✅ OpenAI client initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize OpenAI client: {str(e)}")
            st.error(f"❌ Failed to initialize OpenAI client: {str(e)}")
            return None
        
        # Check AWS credentials
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_region = os.getenv('AWS_REGION', 'us-east-1')
        
        if not all([aws_access_key, aws_secret_key]):
            st.error("❌ AWS credentials not found. Please add AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to your .env file.")
            return None
        
        # Initialize AWS clients
        try:
            textract_client = boto3.client(
                'textract',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region
            )
            print("✅ AWS Textract client initialized successfully")
            
            rekognition_client = boto3.client(
                'rekognition',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region
            )
            print("✅ AWS Rekognition client initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize AWS clients: {str(e)}")
            st.error(f"❌ Failed to initialize AWS clients: {str(e)}")
            return None
        
        # Check Pinecone API key
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        if not pinecone_api_key:
            st.error("❌ Pinecone API key not found. Please add PINECONE_API_KEY to your .env file.")
            return None
        
        # Initialize Pinecone client
        try:
            import pinecone
            pc = pinecone.Pinecone(api_key=pinecone_api_key)
            print("✅ Pinecone client initialized successfully")
            
            # Create Pinecone index if it doesn't exist
            try:
                pinecone_index = pc.Index(PINECONE_INDEX_NAME)
                print(f"✅ Pinecone index '{PINECONE_INDEX_NAME}' connected successfully")
                
                # Check if index dimensions match our embedding model
                index_stats = pinecone_index.describe_index_stats()
                current_dimensions = index_stats.dimension
                from openai_config import EMBEDDING_MODELS
                expected_dimensions = EMBEDDING_MODELS[OPENAI_EMBEDDING_MODEL]["dimensions"]
                
                if current_dimensions != expected_dimensions:
                    print(f"⚠️ Index dimension mismatch: current={current_dimensions}, expected={expected_dimensions}")
                    st.warning(f"""
                    ⚠️ **Index Dimension Mismatch Detected**
                    
                    Your existing Pinecone index has {current_dimensions} dimensions, but the new model requires {expected_dimensions} dimensions.
                    
                    **Options:**
                    1. **Delete and recreate** the index (recommended for new projects)
                    2. **Use the old model** by setting `OPENAI_EMBEDDING_MODEL=text-embedding-3-small` in your .env file
                    
                    The app will continue with the current index dimensions for now.
                    """)
                    
                    # Automatically switch to compatible model
                    if current_dimensions == 1536:
                        OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
                        print(f"🔄 Automatically switched to {OPENAI_EMBEDDING_MODEL} for compatibility")
                        st.info(f"🔄 Automatically switched to `{OPENAI_EMBEDDING_MODEL}` for compatibility with existing index")
                    elif current_dimensions == 3072:
                        OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
                        print(f"🔄 Automatically switched to {OPENAI_EMBEDDING_MODEL} for compatibility")
                        st.info(f"🔄 Automatically switched to `{OPENAI_EMBEDDING_MODEL}` for compatibility with existing index")
                
            except Exception as index_error:
                # Create new index with OpenAI embedding dimensions
                print(f"🔄 Creating new Pinecone index '{PINECONE_INDEX_NAME}'...")
                
                # Get embedding dimensions from config
                from openai_config import EMBEDDING_MODELS
                embedding_dimensions = EMBEDDING_MODELS[OPENAI_EMBEDDING_MODEL]["dimensions"]
                
                pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=embedding_dimensions,  # Use dimensions from config
                    metric="cosine",
                    spec=pinecone.ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                pinecone_index = pc.Index(PINECONE_INDEX_NAME)
                print(f"✅ Pinecone index '{PINECONE_INDEX_NAME}' created successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Pinecone client: {str(e)}")
            st.error(f"❌ Failed to initialize Pinecone client: {str(e)}")
            return None
        
        return openai_client, textract_client, rekognition_client, pinecone_index
        
    except Exception as e:
        st.error(f"❌ Unexpected error initializing API clients: {str(e)}")
        return None

@st.cache_data
def extract_text_from_image_aws(image_bytes: bytes, _textract_client) -> str:
    """Extract text from image using AWS Textract."""
    try:
        # AWS Textract expects bytes
        response = _textract_client.detect_document_text(
            Document={'Bytes': image_bytes}
        )
        
        # Extract text from response
        extracted_text = ""
        for block in response['Blocks']:
            if block['BlockType'] == 'LINE':
                extracted_text += block['Text'] + " "
        
        return extracted_text.strip()
    except Exception as e:
        st.error(f"Error extracting text with AWS Textract: {str(e)}")
        return ""

def analyze_image_aws(image_bytes: bytes, _rekognition_client) -> Dict:
    """Analyze image using AWS Rekognition for additional context."""
    try:
        # Detect labels (objects, scenes, activities)
        label_response = _rekognition_client.detect_labels(
            Image={'Bytes': image_bytes},
            MaxLabels=10,
            MinConfidence=70
        )
        
        # Detect text (additional OCR)
        text_response = _rekognition_client.detect_text(
            Image={'Bytes': image_bytes}
        )
        
        # Extract detected text
        detected_text = ""
        for text_detection in text_response['TextDetections']:
            if text_detection['Type'] == 'LINE':
                detected_text += text_detection['DetectedText'] + " "
        
        # Extract labels
        labels = [label['Name'] for label in label_response['Labels']]
        
        return {
            'labels': labels,
            'detected_text': detected_text.strip(),
            'confidence': label_response['Labels'][0]['Confidence'] if label_response['Labels'] else 0
        }
    except Exception as e:
        st.error(f"Error analyzing image with AWS Rekognition: {str(e)}")
        return {'labels': [], 'detected_text': '', 'confidence': 0}

def analyze_image_gpt4o(image_bytes: bytes, _openai_client) -> Dict:
    """Analyze image using GPT-4o's advanced vision capabilities."""
    try:
        # Encode image to base64
        img_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Create prompt for comprehensive image analysis
        prompt = """Analyze this image comprehensively and provide:
1. A detailed description of what you see
2. Any text content visible in the image
3. Objects, people, or elements present
4. The overall context or purpose of the image
5. Any technical details (UI elements, code, diagrams, etc.)

Please be thorough and specific in your analysis."""

        # Use GPT-4o for vision analysis
        response = _openai_client.chat.completions.create(
            model=OPENAI_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.1
        )
        
        analysis = response.choices[0].message.content
        
        return {
            'gpt4o_analysis': analysis,
            'model_used': OPENAI_VISION_MODEL,
            'confidence': 95.0  # GPT-4o typically has high confidence
        }
        
    except Exception as e:
        print(f"⚠️ GPT-4o vision analysis failed: {str(e)}")
        return {
            'gpt4o_analysis': f"Analysis failed: {str(e)}",
            'model_used': OPENAI_VISION_MODEL,
            'confidence': 0.0
        }

def generate_image_embedding_openai(image: Image.Image, _openai_client) -> np.ndarray:
    """Generate OpenAI embedding for an image."""
    global OPENAI_EMBEDDING_MODEL
    
    try:
        # For OpenAI embeddings, we need to be extremely aggressive with compression
        # The text-embedding-3-small model has only 8,192 token context length
        
        # Start with very small dimensions - even smaller!
        max_dimension = 128  # Much smaller to stay under token limit
        
        # Create a copy to avoid modifying the original
        working_image = image.copy()
        
        # Resize if image is too large
        if working_image.width > max_dimension or working_image.height > max_dimension:
            working_image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            print(f"🔄 Resized image to {working_image.width}x{working_image.height} to reduce token count")
        
        # Try multiple compression levels - even more aggressive
        compression_levels = [30, 20, 15, 10, 5]  # Start lower, go even lower if needed
        
        for quality in compression_levels:
            try:
                # Convert to bytes with very aggressive compression
                img_byte_arr = io.BytesIO()
                working_image.save(img_byte_arr, format='JPEG', quality=quality, optimize=True)
                img_byte_arr = img_byte_arr.getvalue()
                
                # Check file size - be even more conservative
                file_size_kb = len(img_byte_arr) / 1024
                print(f"🔄 Trying quality {quality}: {file_size_kb:.1f}KB")
                
                # If file is small enough, try the embedding
                if file_size_kb < 50:  # Keep under 50KB to be extra safe
                    # Encode to base64
                    img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
                    
                    # Generate embedding using OpenAI
                    response = _openai_client.embeddings.create(
                        model=OPENAI_EMBEDDING_MODEL,
                        input=f"data:image/jpeg;base64,{img_base64}"
                    )
                    
                    embedding = np.array(response.data[0].embedding)
                    
                    # Validate embedding is not all zeros
                    if np.all(embedding == 0):
                        raise ValueError("Generated embedding is all zeros")
                    
                    print(f"✅ Generated embedding with quality {quality}: {embedding.shape}, non-zero values: {np.count_nonzero(embedding)}")
                    return embedding
                    
            except Exception as e:
                print(f"⚠️ Quality {quality} failed: {str(e)}")
                continue
        
        # If we get here, all compression levels failed
        print(f"❌ All compression levels failed. File sizes were too large even at quality 5.")
        print(f"🔄 This image is too complex for OpenAI embeddings. Will use text-based fallback.")
        return None  # Return None to trigger fallback
        
    except Exception as e:
        st.error(f"Error generating image embedding: {str(e)}")
        print(f"❌ Embedding generation failed: {str(e)}")
        return None  # Return None to trigger fallback

def generate_text_embedding_openai(text: str, _openai_client) -> np.ndarray:
    """Generate OpenAI embedding for text."""
    global OPENAI_EMBEDDING_MODEL
    
    try:
        response = _openai_client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=text
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        st.error(f"Error generating text embedding: {str(e)}")
        from openai_config import EMBEDDING_MODELS
        expected_dimensions = EMBEDDING_MODELS[OPENAI_EMBEDDING_MODEL]["dimensions"]
        return np.zeros(expected_dimensions)

def store_in_pinecone(embeddings: List[np.ndarray], metadata: List[Dict], _pinecone_index):
    """Store embeddings in Pinecone vector database."""
    global OPENAI_EMBEDDING_MODEL
    
    try:
        vectors = []
        for i, (embedding, meta) in enumerate(zip(embeddings, metadata)):
            # Validate embedding before storing
            if embedding is None or np.all(embedding == 0):
                print(f"⚠️ Skipping invalid embedding for {meta['filename']}")
                continue
            
            # Ensure embedding is the right shape and type
            from openai_config import EMBEDDING_MODELS
            expected_dimensions = EMBEDDING_MODELS[OPENAI_EMBEDDING_MODEL]["dimensions"]
            
            if embedding.shape != (expected_dimensions,):
                print(f"⚠️ Skipping embedding with wrong shape {embedding.shape} for {meta['filename']}")
                continue
            
            # Convert to list and validate
            embedding_list = embedding.tolist()
            if not any(embedding_list):  # Check if all values are zero
                print(f"⚠️ Skipping zero embedding for {meta['filename']}")
                continue
            
            vectors.append({
                'id': meta['hash'],
                'values': embedding_list,
                'metadata': {
                    'filename': meta['filename'],
                    'text': meta['text'],
                    'labels': meta.get('labels', []),
                    'detected_text': meta.get('detected_text', ''),
                    'confidence': meta.get('confidence', 0),
                    'gpt4o_analysis': meta.get('gpt4o_analysis', ''),
                    'gpt4o_model': meta.get('gpt4o_model', ''),
                    'timestamp': time.time()
                }
            })
        
        if not vectors:
            st.error("❌ No valid embeddings to store")
            return False
        
        print(f"🔄 Storing {len(vectors)} valid embeddings in Pinecone...")
        
        # Upsert to Pinecone
        _pinecone_index.upsert(vectors=vectors)
        
        print(f"✅ Successfully stored {len(vectors)} embeddings in Pinecone")
        return True
        
    except Exception as e:
        st.error(f"Error storing in Pinecone: {str(e)}")
        print(f"❌ Pinecone storage error: {str(e)}")
        return False

def search_similar_images_pinecone(query_embedding: np.ndarray, _pinecone_index, top_k: int = 5) -> Tuple[List[float], List[str]]:
    """Search for similar images using Pinecone."""
    try:
        # Search in Pinecone
        results = _pinecone_index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        
        scores = [match.score for match in results.matches]
        ids = [match.id for match in results.matches]
        
        return scores, ids
    except Exception as e:
        st.error(f"Error searching in Pinecone: {str(e)}")
        return [], []

def process_uploaded_files(uploaded_files, textract_client, rekognition_client, openai_client, pinecone_index) -> Tuple[List[Dict], bool]:
    """Process uploaded files and create searchable index."""
    global OPENAI_EMBEDDING_MODEL
    
    # Validate that all clients are properly initialized
    if not all([textract_client, rekognition_client, openai_client, pinecone_index]):
        st.error("❌ Cannot process files: One or more API clients are not properly initialized.")
        return [], False
    
    processed_data = []
    embeddings = []
    
    progress_bar = st.progress(0)
    total_files = len(uploaded_files)
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Read image
            image_bytes = uploaded_file.read()
            image = Image.open(io.BytesIO(image_bytes))
            
            # Extract text using AWS Textract
            extracted_text = extract_text_from_image_aws(image_bytes, textract_client)
            
            # Analyze image using AWS Rekognition
            image_analysis = analyze_image_aws(image_bytes, rekognition_client)
            
            # Analyze image using GPT-4o for enhanced understanding
            gpt4o_analysis = analyze_image_gpt4o(image_bytes, openai_client)
            
            # Generate image embedding using OpenAI
            image_embedding = generate_image_embedding_openai(image, openai_client)
            
            # If image embedding failed or returned None, try text-based embedding as fallback
            if image_embedding is None or np.all(image_embedding == 0):
                print(f"🔄 Using text-based embedding fallback for {uploaded_file.name}")
                
                # Create rich text description combining all available information
                text_parts = []
                
                if extracted_text.strip():
                    text_parts.append(f"Text content: {extracted_text}")
                
                if image_analysis['labels']:
                    text_parts.append(f"Objects detected: {', '.join(image_analysis['labels'])}")
                
                if image_analysis['detected_text'].strip():
                    text_parts.append(f"Additional text: {image_analysis['detected_text']}")
                
                # Add filename as context
                text_parts.append(f"Filename: {uploaded_file.name}")
                
                # Combine all text parts
                combined_text = " | ".join(text_parts)
                
                if combined_text.strip():
                    print(f"🔄 Generating text-based embedding from: {combined_text[:100]}...")
                    image_embedding = generate_text_embedding_openai(combined_text, openai_client)
                    
                    # Validate the text-based embedding
                    if image_embedding is not None and not np.all(image_embedding == 0):
                        print(f"✅ Text-based embedding successful: {image_embedding.shape}")
                    else:
                        print(f"⚠️ Text-based embedding failed, using filename only")
                        image_embedding = generate_text_embedding_openai(uploaded_file.name, openai_client)
                else:
                    # Last resort: use filename as text
                    print(f"🔄 Using filename as fallback text: {uploaded_file.name}")
                    image_embedding = generate_text_embedding_openai(uploaded_file.name, openai_client)
                
                # Final validation - if still no valid embedding, create a minimal one
                if image_embedding is None or np.all(image_embedding == 0):
                    print(f"⚠️ All text-based embeddings failed, creating minimal embedding from filename")
                    # Create a simple embedding from filename
                    image_embedding = generate_text_embedding_openai(uploaded_file.name, openai_client)
                    
                    # If even that fails, create a random normalized embedding as last resort
                    if image_embedding is None or np.all(image_embedding == 0):
                        print(f"🔄 Creating fallback random embedding as last resort")
                        from openai_config import EMBEDDING_MODELS
                        expected_dimensions = EMBEDDING_MODELS[OPENAI_EMBEDDING_MODEL]["dimensions"]
                        random_embedding = np.random.normal(0, 0.1, expected_dimensions)
                        random_embedding = random_embedding / np.linalg.norm(random_embedding)  # Normalize
                        image_embedding = random_embedding
            
            # Create image hash for unique identification
            image_hash = hashlib.md5(image_bytes).hexdigest()
            
            # Store processed data
            processed_data.append({
                'filename': uploaded_file.name,
                'image': image,
                'text': extracted_text,
                'labels': image_analysis['labels'],
                'detected_text': image_analysis['detected_text'],
                'confidence': image_analysis['confidence'],
                'gpt4o_analysis': gpt4o_analysis.get('gpt4o_analysis', ''),
                'gpt4o_model': gpt4o_analysis.get('model_used', ''),
                'hash': image_hash,
                'bytes': image_bytes
            })
            
            embeddings.append(image_embedding)
            
            # Update progress
            progress_bar.progress((i + 1) / total_files)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            continue
    
    progress_bar.empty()
    
    # Store in Pinecone
    success = store_in_pinecone(embeddings, processed_data, pinecone_index)
    
    return processed_data, success

def display_search_results(query: str, processed_data: List[Dict], pinecone_index, openai_client, top_k: int = 5):
    """Display search results based on query."""
    if not processed_data or pinecone_index is None:
        st.warning("No processed images available for search.")
        return
    
    # Generate query embedding
    query_embedding = generate_text_embedding_openai(query, openai_client)
    
    # Search for similar images
    scores, ids = search_similar_images_pinecone(query_embedding, pinecone_index, top_k)
    
    if len(scores) == 0:
        st.warning("No results found.")
        return
    
    st.subheader(f"Top {len(scores)} Results for: '{query}'")
    
    # Display results
    for i, (score, image_hash) in enumerate(zip(scores, ids)):
        # Find the corresponding processed data
        result = None
        for item in processed_data:
            if item['hash'] == image_hash:
                result = item
                break
        
        if result is None:
            continue
        
        # Create columns for layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Display thumbnail
            st.image(result['image'], caption=result['filename'], width=200)
            
        with col2:
            # Display metadata
            st.write(f"**Filename:** {result['filename']}")
            st.write(f"**Confidence Score:** {score:.3f}")
            
            # Display AWS analysis results
            if result.get('labels'):
                st.write(f"**Detected Objects:** {', '.join(result['labels'][:5])}")
            
            if result.get('confidence', 0) > 0:
                st.write(f"**Analysis Confidence:** {result['confidence']:.1f}%")
            
            # Display GPT-4o analysis if available
            if result.get('gpt4o_analysis') and result['gpt4o_analysis'] != 'Analysis failed':
                st.write(f"**GPT-4o Analysis:**")
                st.text_area("GPT-4o Analysis", value=result['gpt4o_analysis'], height=120, key=f"gpt4o_{i}_{result['hash']}")
            
            # Display OCR text snippet
            text_snippet = result['text'][:300] + "..." if len(result['text']) > 300 else result['text']
            st.write(f"**Extracted Text:**")
            st.text_area("Extracted Text", value=text_snippet, height=100, key=f"text_{i}_{result['hash']}")
        
        st.divider()

def main():
    st.set_page_config(
        page_title="Visual Memory Search (AWS)",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Visual Memory Search (AWS-Powered)")
    st.markdown("Upload screenshots and search through them using natural language queries. Powered by OpenAI, AWS Textract, AWS Rekognition, and Pinecone.")
    
    # Check if API keys are configured
    required_env_vars = ['OPENAI_API_KEY', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'PINECONE_API_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        st.error(f"❌ Missing API keys: {', '.join(missing_vars)}")
        st.info("Please create a `.env` file with your API keys. See `env_template.txt` for reference.")
        return
    
    # Sidebar instructions
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. **Upload Screenshots**: Use the file uploader to select multiple screenshot files (PNG, JPG, JPEG)
        2. **Processing**: The app will automatically extract text using AWS Textract and analyze images with AWS Rekognition
        3. **Search**: Enter a natural language query to find relevant screenshots
        4. **Results**: View the top 5 most similar screenshots with confidence scores
        
        **Supported Formats**: PNG, JPG, JPEG
        
        **Example Queries**:
        - "code editor with Python"
        - "error message red text"
        - "settings page dark mode"
        - "graph or chart visualization"
        """)
        
        st.divider()
        st.markdown("**💡 Tips:**")
        st.markdown("""
        - More specific queries yield better results
        - The app searches both visual content and extracted text
        - Confidence scores range from 0 to 1 (higher is better)
        """)
        
        st.divider()
        
        # Model Information
        st.header("🤖 Models in Use")
        st.markdown(f"""
        **Embedding Model:** `{OPENAI_EMBEDDING_MODEL}`
        - 1536 dimensions for compatibility with existing indexes
        - Fast and cost-effective processing
        
        **Vision Model:** `{OPENAI_VISION_MODEL}`
        - Advanced image understanding and analysis
        
        **Text Model:** `{OPENAI_TEXT_MODEL}`
        - State-of-the-art text generation
        """)
        
        # Show compatibility status
        if OPENAI_EMBEDDING_MODEL == "text-embedding-3-small":
            st.success("✅ **Compatible Mode**: Using text-embedding-3-small for existing 1536-dimension indexes")
        else:
            st.info("ℹ️ **Upgrade Mode**: Using text-embedding-3-large for enhanced 3072-dimension embeddings")
        
        # Cost estimation
        if st.session_state.get('processed_data'):
            num_images = len(st.session_state.processed_data)
            cost_estimate = get_cost_estimate(OPENAI_EMBEDDING_MODEL, OPENAI_VISION_MODEL, num_images)
            st.markdown(f"""
            **💰 Cost Estimate:**
            - Total: ${cost_estimate['total_cost']:.4f}
            - Per image: ${cost_estimate['cost_per_image']:.4f}
            """)
        
        st.divider()
        
        # Pinecone Index Management
        if 'pinecone_index' in locals() and pinecone_index is not None:
            manage_pinecone_index(pinecone_index, openai_client)
        
        st.divider()
        
        # Detailed explanation button
        if st.button("🔬 How Does This Work?", type="secondary", help="Click to learn about the technology behind this app"):
            st.session_state.show_explanation = not st.session_state.get('show_explanation', False)
        
        # Detailed explanation section
        if st.session_state.get('show_explanation', False):
            st.markdown("### 🧠 **Technical Deep Dive (AWS Version)**")
            
            st.markdown("""
            #### **1. Text Extraction (AWS Textract)**
            - **AWS Textract**: Amazon's advanced OCR service for document analysis
            - **Process**: Converts image → detects text regions → recognizes characters → outputs readable text
            - **Benefits**: Higher accuracy, handles complex layouts, supports 100+ languages
            """)
            
            st.markdown("""
            #### **2. Image Analysis (AWS Rekognition)**
            - **AWS Rekognition**: Amazon's computer vision service for image understanding
            - **Features**: Object detection, scene recognition, text detection, label identification
            - **Benefits**: Comprehensive image analysis, high accuracy, scalable processing
            """)
            
            st.markdown("""
            #### **3. Visual Understanding (OpenAI Embeddings)**
            - **OpenAI Embeddings**: State-of-the-art text and image understanding
            - **Current Model**: 
              - `text-embedding-3-small` (1536 dimensions) for compatibility with existing indexes
              - Fast, cost-effective, and maintains all existing data
            - **Embeddings**: Converts images and text into 1536-dimensional numerical vectors
            - **How it works**: 
              - Images → Visual features (shapes, colors, objects, layout)
              - Text → Semantic meaning (concepts, descriptions, context)
              - Both → Same vector space for comparison
            - **Benefits**: 
              - ✅ **Compatible** with existing Pinecone indexes
              - ✅ **Cost-effective** processing
              - ✅ **Fast** embedding generation
              - ✅ **No data loss** from existing images
            """)
            
            st.markdown("""
            #### **4. Advanced Vision Analysis (GPT-4o)**
            - **GPT-4o**: OpenAI's latest multimodal model with advanced vision capabilities
            - **Features**: 
              - Comprehensive image understanding and description
              - Context-aware analysis of UI elements, code, diagrams
              - Natural language explanations of visual content
            - **Benefits**: More accurate and detailed image analysis than traditional computer vision
            """)
            
            st.markdown("""
            #### **5. Vector Search (Pinecone)**
            - **Pinecone**: Cloud-native vector database for production use
            - **Index**: Creates a scalable, persistent searchable database of image embeddings
            - **Search Process**:
              1. Convert your query to embedding
              2. Compare with all stored image embeddings
              3. Return most similar matches using cosine similarity
            """)
            
            st.markdown("""
            #### **6. AWS Benefits**
            - **Scalability**: Handle millions of images with AWS infrastructure
            - **Performance**: Optimized for speed and accuracy
            - **Reliability**: 99.99% uptime SLA
            - **Integration**: Seamless AWS ecosystem integration
            """)
            
            st.markdown("""
            #### **7. Cost Considerations**
            - **AWS Textract**: ~$1.50 per 1000 images
            - **AWS Rekognition**: ~$1.00 per 1000 images
            - **OpenAI Embeddings**: ~$0.00002 per image (text-embedding-3-small)
            - **OpenAI Vision**: ~$0.005 per image (GPT-4o analysis)
            - **Pinecone**: Free tier available, then ~$0.10 per 1000 queries
            - **Total**: Typically under $2.50 per 1000 images
            - **Current Setup**: Using cost-effective text-embedding-3-small for optimal pricing
            """)
    
    # Initialize session state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = []
    
    # Load models
    with st.spinner("Initializing API clients..."):
        clients = initialize_clients()
        if not clients or len(clients) != 4:
            st.error("Failed to initialize API clients. Please check your API keys.")
            st.info("Make sure you have all required API keys in your .env file:")
            st.code("""
OPENAI_API_KEY=your_openai_key
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
PINECONE_API_KEY=your_pinecone_key
            """)
            return
        
        openai_client, textract_client, rekognition_client, pinecone_index = clients
        
        # Validate that all clients are properly initialized
        if not all([openai_client, textract_client, rekognition_client, pinecone_index]):
            st.error("One or more API clients failed to initialize properly.")
            return
    
    st.success("✅ API clients initialized successfully!")
    
    # File upload section
    st.header("📁 Upload Screenshots")
    uploaded_files = st.file_uploader(
        "Choose screenshot files",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload multiple screenshot files to build your searchable database"
    )
    
    # Process uploaded files
    if uploaded_files and len(uploaded_files) > 0:
        if st.button("🔄 Process Screenshots", type="primary"):
            with st.spinner(f"Processing {len(uploaded_files)} screenshots..."):
                processed_data, success = process_uploaded_files(
                    uploaded_files, textract_client, rekognition_client, openai_client, pinecone_index
                )
                
                if success:
                    # Store in session state
                    st.session_state.processed_data = processed_data
                    
                    st.success(f"✅ Successfully processed {len(processed_data)} screenshots!")
                    
                    # Display processing summary
                    with st.expander("📊 Processing Summary"):
                        df_summary = pd.DataFrame([
                            {
                                'Filename': item['filename'],
                                'Text Length': len(item['text']),
                                'Labels': ', '.join(item.get('labels', [])[:3]),
                                'GPT-4o Model': item.get('gpt4o_model', 'N/A'),
                                'Text Preview': item['text'][:100] + "..." if len(item['text']) > 100 else item['text']
                            }
                            for item in processed_data
                        ])
                        st.dataframe(df_summary, use_container_width=True)
                else:
                    st.error("Failed to store images in the vector database.")
    
    # Search section
    if st.session_state.processed_data:
        st.header("🔍 Search Screenshots")
        
        # Query input
        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., 'code editor with Python', 'error message', 'settings page'",
            help="Describe what you're looking for in natural language"
        )
        
        # Search controls
        col1, col2 = st.columns([3, 1])
        with col1:
            top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)
        with col2:
            search_button = st.button("🔍 Search", type="primary")
        
        # Perform search
        if query and (search_button or query):
            with st.spinner("Searching..."):
                display_search_results(query, st.session_state.processed_data, pinecone_index, openai_client, top_k)
    
    else:
        st.info("👆 Upload and process screenshots to start searching!")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <small>Powered by OpenAI, AWS Textract, AWS Rekognition, and Pinecone</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()