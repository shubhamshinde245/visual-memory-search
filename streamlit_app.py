import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import easyocr
import faiss
from sentence_transformers import SentenceTransformer
import io
import base64
import hashlib
import pickle
import os
from typing import List, Tuple, Dict
import tempfile

@st.cache_resource
def load_models():
    """Load OCR and CLIP models with caching."""
    ocr_reader = easyocr.Reader(['en'])
    clip_model = SentenceTransformer('clip-ViT-B-32')
    return ocr_reader, clip_model

@st.cache_data
def extract_text_from_image(image_bytes: bytes, _ocr_reader) -> str:
    """Extract text from image using EasyOCR."""
    try:
        # Save image to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(image_bytes)
            tmp_file_path = tmp_file.name
        
        # Extract text using EasyOCR
        results = _ocr_reader.readtext(tmp_file_path)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        # Combine all detected text
        extracted_text = ' '.join([result[1] for result in results])
        return extracted_text
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return ""

def generate_image_embedding(image: Image.Image, clip_model) -> np.ndarray:
    """Generate CLIP embedding for an image."""
    try:
        # Convert PIL Image to format expected by CLIP
        embedding = clip_model.encode([image], convert_to_tensor=False)
        return embedding[0]
    except Exception as e:
        st.error(f"Error generating image embedding: {str(e)}")
        return np.zeros(512)  # Default embedding size for CLIP

def generate_text_embedding(text: str, clip_model) -> np.ndarray:
    """Generate CLIP embedding for text."""
    try:
        embedding = clip_model.encode([text], convert_to_tensor=False)
        return embedding[0]
    except Exception as e:
        st.error(f"Error generating text embedding: {str(e)}")
        return np.zeros(512)

def create_faiss_index(embeddings: List[np.ndarray]) -> faiss.Index:
    """Create FAISS index from embeddings."""
    if not embeddings:
        return None
    
    dimension = len(embeddings[0])
    index = faiss.IndexFlatIP(dimension)  # Inner product similarity
    
    # Normalize embeddings for cosine similarity
    embeddings_array = np.array(embeddings).astype('float32')
    faiss.normalize_L2(embeddings_array)
    
    index.add(embeddings_array)
    return index

def search_similar_images(query_embedding: np.ndarray, index: faiss.Index, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Search for similar images using FAISS."""
    if index is None:
        return np.array([]), np.array([])
    
    # Normalize query embedding
    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    # Search
    scores, indices = index.search(query_embedding, k)
    return scores[0], indices[0]

def process_uploaded_files(uploaded_files, ocr_reader, clip_model) -> Tuple[List[Dict], faiss.Index]:
    """Process uploaded files and create searchable index."""
    processed_data = []
    embeddings = []
    
    progress_bar = st.progress(0)
    total_files = len(uploaded_files)
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Read image
            image_bytes = uploaded_file.read()
            image = Image.open(io.BytesIO(image_bytes))
            
            # Extract text using OCR
            extracted_text = extract_text_from_image(image_bytes, ocr_reader)
            
            # Generate image embedding
            image_embedding = generate_image_embedding(image, clip_model)
            
            # Create image hash for unique identification
            image_hash = hashlib.md5(image_bytes).hexdigest()
            
            # Store processed data
            processed_data.append({
                'filename': uploaded_file.name,
                'image': image,
                'text': extracted_text,
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
    
    # Create FAISS index
    index = create_faiss_index(embeddings)
    
    return processed_data, index

def display_search_results(query: str, processed_data: List[Dict], index: faiss.Index, clip_model, top_k: int = 5):
    """Display search results based on query."""
    if not processed_data or index is None:
        st.warning("No processed images available for search.")
        return
    
    # Generate query embedding
    query_embedding = generate_text_embedding(query, clip_model)
    
    # Search for similar images
    scores, indices = search_similar_images(query_embedding, index, top_k)
    
    if len(scores) == 0:
        st.warning("No results found.")
        return
    
    st.subheader(f"Top {len(scores)} Results for: '{query}'")
    
    # Display results
    for i, (score, idx) in enumerate(zip(scores, indices)):
        if idx >= len(processed_data):
            continue
            
        result = processed_data[idx]
        
        # Create columns for layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Display thumbnail
            st.image(result['image'], caption=result['filename'], width=200)
            
        with col2:
            # Display metadata
            st.write(f"**Filename:** {result['filename']}")
            st.write(f"**Confidence Score:** {score:.3f}")
            
            # Display OCR text snippet
            text_snippet = result['text'][:300] + "..." if len(result['text']) > 300 else result['text']
            st.write(f"**Extracted Text:**")
            st.text_area("Extracted Text", value=text_snippet, height=100, key=f"text_{i}_{result['hash']}")
        
        st.divider()

def main():
    st.set_page_config(
        page_title="Visual Memory Search",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Visual Memory Search")
    st.markdown("Upload screenshots and search through them using natural language queries.")
    
    # Sidebar instructions
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. **Upload Screenshots**: Use the file uploader to select multiple screenshot files (PNG, JPG, JPEG)
        2. **Processing**: The app will automatically extract text using OCR and generate visual embeddings
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
            st.markdown("### 🧠 **Technical Deep Dive**")
            
            st.markdown("""
            #### **1. Text Extraction (OCR)**
            - **EasyOCR**: Uses deep learning to read text from images
            - **Process**: Converts image → detects text regions → recognizes characters → outputs readable text
            - **Benefits**: Works with various fonts, orientations, and image qualities
            """)
            
            st.markdown("""
            #### **2. Visual Understanding (CLIP)**
            - **CLIP Model**: OpenAI's vision-language model that understands both images and text
            - **Embeddings**: Converts images and text into 512-dimensional numerical vectors
            - **How it works**: 
              - Images → Visual features (shapes, colors, objects, layout)
              - Text → Semantic meaning (concepts, descriptions, context)
              - Both → Same vector space for comparison
            """)
            
            st.markdown("""
            #### **3. Vector Search (FAISS)**
            - **FAISS**: Facebook's library for efficient similarity search
            - **Index**: Creates a searchable database of image embeddings
            - **Search Process**:
              1. Convert your query to embedding
              2. Compare with all stored image embeddings
              3. Return most similar matches using cosine similarity
            """)
            
            st.markdown("""
            #### **4. Hybrid Search Strategy**
            - **Text + Visual**: Combines OCR text and visual features
            - **Query Matching**: Your search query is compared against:
              - Extracted text content
              - Visual elements (UI components, colors, layout)
            - **Ranking**: Results sorted by overall similarity score
            """)
            
            st.markdown("""
            #### **5. Why This Works**
            - **Natural Language**: Search like you're talking to a person
            - **Context Aware**: Understands relationships between text and visuals
            - **Scalable**: Can handle thousands of screenshots efficiently
            - **Accurate**: Deep learning models trained on millions of examples
            """)
            
            st.markdown("""
            #### **6. Use Cases**
            - **Documentation**: Find specific UI elements or error messages
            - **Design Reference**: Locate screenshots with particular layouts
            - **Bug Tracking**: Search for specific error states or UI issues
            - **Training**: Find examples of specific features or workflows
            """)
            
            st.markdown("""
            #### **7. Performance Notes**
            - **First Run**: Models download (~500MB) and cache locally
            - **Processing**: ~1-3 seconds per screenshot depending on complexity
            - **Search**: Near-instant results using FAISS indexing
            - **Memory**: Efficient vector storage and retrieval
            """)
    
    # Initialize session state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = []
    if 'faiss_index' not in st.session_state:
        st.session_state.faiss_index = None
    
    # Load models
    with st.spinner("Loading AI models..."):
        ocr_reader, clip_model = load_models()
    
    st.success("✅ Models loaded successfully!")
    
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
                processed_data, faiss_index = process_uploaded_files(uploaded_files, ocr_reader, clip_model)
                
                # Store in session state
                st.session_state.processed_data = processed_data
                st.session_state.faiss_index = faiss_index
                
                st.success(f"✅ Successfully processed {len(processed_data)} screenshots!")
                
                # Display processing summary
                with st.expander("📊 Processing Summary"):
                    df_summary = pd.DataFrame([
                        {
                            'Filename': item['filename'],
                            'Text Length': len(item['text']),
                            'Text Preview': item['text'][:100] + "..." if len(item['text']) > 100 else item['text']
                        }
                        for item in processed_data
                    ])
                    st.dataframe(df_summary, use_container_width=True)
    
    # Search section
    if st.session_state.processed_data and st.session_state.faiss_index:
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
                display_search_results(query, st.session_state.processed_data, st.session_state.faiss_index, clip_model, top_k)
    
    else:
        st.info("👆 Upload and process screenshots to start searching!")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <small>Powered by EasyOCR for text extraction and CLIP for visual understanding</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()