# 🔍 Visual Memory Search

[APP LINK](https://visual-memory-search.streamlit.app/)

A powerful visual search application that allows you to upload screenshots and search through them using natural language queries. Built with the latest AI technologies including **OpenAI's newest models** (GPT-4o, text-embedding-3-large), AWS services, and Pinecone vector database.

## 🚀 Latest Updates

**December 2024**: Now using the latest OpenAI models:
- **`text-embedding-3-large`** - 3072 dimensions for highest quality embeddings
- **`gpt-4o`** - Advanced multimodal vision understanding
- **Enhanced image analysis** with GPT-4o's vision capabilities
- **Cost monitoring** and model recommendations
- **Automatic fallback** mechanisms for robust processing

## ✨ Features

- **🔍 Natural Language Search**: Search screenshots using plain English descriptions
- **🤖 Latest AI Models**: Powered by OpenAI's newest GPT-4o and text-embedding-3-large
- **📱 Screenshot Processing**: Upload multiple screenshots (PNG, JPG, JPEG)
- **📝 Text Extraction**: Advanced OCR using AWS Textract
- **🖼️ Image Analysis**: Comprehensive analysis using AWS Rekognition + GPT-4o
- **🧠 Vector Search**: Fast similarity search using Pinecone
- **💰 Cost Monitoring**: Real-time cost estimates and optimization tips
- **🔄 Fallback Support**: Robust processing with multiple analysis paths

## 🚀 Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd visual-memory-search
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys**:
   - Copy `env_template.txt` to `.env`
   - Add your API keys for OpenAI, AWS, and Pinecone

4. **Test your setup**:
   ```bash
   python test_apis.py
   ```

5. **Run the app**:
   ```bash
   streamlit run streamlit_app.py
   ```

## 🔧 Pinecone Migration

If you have an existing Pinecone index with different dimensions, use the migration utility:

```bash
python migrate_pinecone_index.py
```

This will help resolve dimension mismatches between your index and the new OpenAI models.

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
