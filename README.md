# 🔍 Visual Memory Search

A powerful AI-powered screenshot search application that lets you find images using natural language queries. Built with Streamlit and powered by OpenAI, AWS Textract, AWS Rekognition, and Pinecone.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://visual-memory-search.streamlit.app/)

## ✨ Features

- **🔍 Natural Language Search**: Find screenshots using everyday language
- **📝 OCR Text Extraction**: Automatically reads text from images using AWS Textract
- **🔍 Image Analysis**: Detects objects and scenes using AWS Rekognition
- **🧠 AI-Powered Understanding**: Uses OpenAI embeddings for intelligent image and text matching
- **⚡ Fast & Scalable**: Cloud-based processing with Pinecone vector database
- **📱 User-Friendly**: Clean Streamlit interface with detailed explanations

## 🚀 Quick Start

### 1. Get API Keys

You'll need API keys from these services:
- **OpenAI**: [Get API Key](https://platform.openai.com/api-keys)
- **AWS**: [Create IAM User](https://console.aws.amazon.com/iam/) (for Textract & Rekognition)
- **Pinecone**: [Sign Up](https://app.pinecone.io/)

### 2. Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd visual-memory-search

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp env_template.txt .env

# Edit .env with your API keys
nano .env
```

### 3. Test APIs

```bash
# Verify all APIs are working
python3 test_apis.py

# Start the app
streamlit run streamlit_app.py
```

## 📋 Requirements

- Python 3.8+
- API keys for OpenAI, AWS (Textract & Rekognition), and Pinecone
- Internet connection for API calls

## 💰 Pricing

- **AWS Textract**: ~$1.50 per 1000 images
- **AWS Rekognition**: ~$1.00 per 1000 images
- **OpenAI Embeddings**: ~$0.0001 per image
- **Pinecone**: Free tier available
- **Total**: ~$2.60 for 1000 images (~$0.26 for 100 images)

## 🔧 How It Works

1. **Upload Screenshots**: Drag and drop multiple image files
2. **AI Processing**: 
   - AWS Textract extracts text using OCR
   - AWS Rekognition analyzes images for objects and scenes
   - OpenAI generates embeddings for images and text
   - Pinecone stores vectors for fast searching
3. **Natural Language Search**: Type queries like "error message" or "settings page"
4. **Smart Results**: Get relevant screenshots ranked by similarity

## 📚 Documentation

- **API Setup Guide**: [API_SETUP_GUIDE.md](API_SETUP_GUIDE.md)
- **Test Script**: [test_apis.py](test_apis.py)
- **Environment Template**: [env_template.txt](env_template.txt)

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**🎉 Transform your screenshot collection into a searchable knowledge base!**
