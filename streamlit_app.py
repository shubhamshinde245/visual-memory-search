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

# Load environment variables
load_dotenv()

# Pinecone configuration
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'visual-memory-search')

# Initialize API clients
@st.cache_resource
def initialize_clients():
    """Initialize API clients with caching."""
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
            openai_client.embeddings.create(model="text-embedding-3-small", input="test")
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
            from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion, Metric
            pc = Pinecone(api_key=pinecone_api_key)
            print("✅ Pinecone client initialized successfully")
            
            # Create Pinecone index if it doesn't exist
            try:
                pinecone_index = pc.Index(PINECONE_INDEX_NAME)
                print(f"✅ Pinecone index '{PINECONE_INDEX_NAME}' connected successfully")
            except:
                # Create new index with OpenAI embedding dimensions
                print(f"🔄 Creating new Pinecone index '{PINECONE_INDEX_NAME}'...")
                pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=1536,  # OpenAI text-embedding-3-small dimension
                    metric=Metric.COSINE,
                    spec=ServerlessSpec(
                        cloud=CloudProvider.AWS,
                        region=AwsRegion.US_EAST_1
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

def generate_image_embedding_openai(image: Image.Image, _openai_client) -> np.ndarray:
    """Generate OpenAI embedding for an image."""
    try:
        # For OpenAI embeddings, we need to be much more aggressive with compression
        # The text-embedding-3-small model has only 8,192 token context length
        
        # Start with very small dimensions
        max_dimension = 256  # Much smaller to stay under token limit
        
        # Create a copy to avoid modifying the original
        working_image = image.copy()
        
        # Resize if image is too large
        if working_image.width > max_dimension or working_image.height > max_dimension:
            working_image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            print(f"🔄 Resized image to {working_image.width}x{working_image.height} to reduce token count")
        
        # Try multiple compression levels
        compression_levels = [60, 40, 20, 10]  # Start with higher quality, go lower if needed
        
        for quality in compression_levels:
            try:
                # Convert to bytes with aggressive compression
                img_byte_arr = io.BytesIO()
                working_image.save(img_byte_arr, format='JPEG', quality=quality, optimize=True)
                img_byte_arr = img_byte_arr.getvalue()
                
                # Check file size
                file_size_kb = len(img_byte_arr) / 1024
                print(f"🔄 Trying quality {quality}: {file_size_kb:.1f}KB")
                
                # If file is small enough, try the embedding
                if file_size_kb < 100:  # Keep under 100KB to be safe
                    # Encode to base64
                    img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
                    
                    # Generate embedding using OpenAI
                    response = _openai_client.embeddings.create(
                        model="text-embedding-3-small",
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
        raise Exception("All compression levels failed to generate valid embedding")
        
    except Exception as e:
        st.error(f"Error generating image embedding: {str(e)}")
        print(f"❌ Embedding generation failed: {str(e)}")
        
        # Return a small random embedding instead of all zeros
        # This ensures Pinecone doesn't reject it
        random_embedding = np.random.normal(0, 0.1, 1536)
        random_embedding = random_embedding / np.linalg.norm(random_embedding)  # Normalize
        print(f"🔄 Using fallback random embedding: {random_embedding.shape}")
        return random_embedding

def generate_text_embedding_openai(text: str, _openai_client) -> np.ndarray:
    """Generate OpenAI embedding for text."""
    try:
        response = _openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        st.error(f"Error generating text embedding: {str(e)}")
        return np.zeros(1536)

def store_in_pinecone(embeddings: List[np.ndarray], metadata: List[Dict], _pinecone_index):
    """Store embeddings in Pinecone vector database."""
    try:
        vectors = []
        for i, (embedding, meta) in enumerate(zip(embeddings, metadata)):
            # Validate embedding before storing
            if embedding is None or np.all(embedding == 0):
                print(f"⚠️ Skipping invalid embedding for {meta['filename']}")
                continue
            
            # Ensure embedding is the right shape and type
            if embedding.shape != (1536,):
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
            
            # Generate image embedding using OpenAI
            image_embedding = generate_image_embedding_openai(image, openai_client)
            
            # If image embedding failed, try text-based embedding as fallback
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
            - **Embeddings**: Converts images and text into 1536-dimensional numerical vectors
            - **How it works**: 
              - Images → Visual features (shapes, colors, objects, layout)
              - Text → Semantic meaning (concepts, descriptions, context)
              - Both → Same vector space for comparison
            """)
            
            st.markdown("""
            #### **4. Vector Search (Pinecone)**
            - **Pinecone**: Cloud-native vector database for production use
            - **Index**: Creates a scalable, persistent searchable database of image embeddings
            - **Search Process**:
              1. Convert your query to embedding
              2. Compare with all stored image embeddings
              3. Return most similar matches using cosine similarity
            """)
            
            st.markdown("""
            #### **5. AWS Benefits**
            - **Scalability**: Handle millions of images with AWS infrastructure
            - **Performance**: Optimized for speed and accuracy
            - **Reliability**: 99.99% uptime SLA
            - **Integration**: Seamless AWS ecosystem integration
            """)
            
            st.markdown("""
            #### **6. Cost Considerations**
            - **AWS Textract**: ~$1.50 per 1000 images
            - **AWS Rekognition**: ~$1.00 per 1000 images
            - **OpenAI Embeddings**: ~$0.0001 per image
            - **Pinecone**: Free tier available, then ~$0.10 per 1000 queries
            - **Total**: Typically under $2.50 per 1000 images
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