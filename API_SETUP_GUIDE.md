# 🚀 API Setup Guide for Visual Memory Search (AWS Version)

Your app has been successfully converted to use AWS services instead of Azure! Here's how to get everything set up.

## 📋 **What Changed**

| Component | Before (Azure) | After (AWS) | Benefits |
|-----------|----------------|-------------|----------|
| **Text Extraction** | Azure Computer Vision | AWS Textract | Better document analysis, handles complex layouts |
| **Image Analysis** | N/A | AWS Rekognition | Object detection, scene recognition, additional OCR |
| **Visual Understanding** | OpenAI | OpenAI | Same powerful embeddings |
| **Vector Database** | Pinecone | Pinecone | Same scalable vector storage |

## 🔑 **Required API Keys**

### 1. **OpenAI API Key**
- **Get it from**: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Cost**: ~$0.0001 per image processed
- **Used for**: Generating image and text embeddings

### 2. **AWS Credentials**
- **Get it from**: [AWS IAM Console](https://console.aws.amazon.com/iam/)
- **Cost**: 
  - AWS Textract: ~$1.50 per 1000 images
  - AWS Rekognition: ~$1.00 per 1000 images
- **Used for**: OCR text extraction and image analysis

### 3. **Pinecone Vector Database**
- **Get it from**: [https://app.pinecone.io/](https://app.pinecone.io/)
- **Cost**: Free tier available, then ~$0.10 per 1000 queries
- **Used for**: Storing and searching image embeddings
- **Note**: Uses modern Pinecone client (no environment needed)

## ⚙️ **Setup Steps**

### Step 1: Create a `.env` file
Copy the template and fill in your API keys:

```bash
# Copy the template
cp env_template.txt .env

# Edit with your actual keys
nano .env
```

### Step 2: Fill in your API keys
```bash
# .env file content
OPENAI_API_KEY=sk-your-actual-openai-key-here
AWS_ACCESS_KEY_ID=your_aws_access_key_id_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key_here
AWS_REGION=us-east-1
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=visual-memory-search
```

### Step 3: Test the setup
```bash
# Verify all APIs are working
python3 test_apis.py

# Start the app
streamlit run streamlit_app.py
```

## 🚀 **AWS Setup Details**

### **Creating AWS IAM User**

1. **Go to AWS IAM Console**: [https://console.aws.amazon.com/iam/](https://console.aws.amazon.com/iam/)
2. **Create New User**: Click "Users" → "Add user"
3. **Set Permissions**: Attach these policies:
   - `AmazonTextractFullAccess`
   - `AmazonRekognitionFullAccess`
4. **Generate Access Keys**: Create access key and secret key
5. **Save Credentials**: Store them in your `.env` file

### **AWS Services Used**

- **AWS Textract**: Advanced OCR for document text extraction
- **AWS Rekognition**: Computer vision for object detection and scene analysis
- **AWS Region**: Choose closest to you (e.g., `us-east-1`, `eu-west-1`)

## 💰 **Cost Breakdown**

| Service | Cost per 1000 images | Notes |
|---------|---------------------|-------|
| **AWS Textract** | ~$1.50 | OCR text extraction |
| **AWS Rekognition** | ~$1.00 | Image analysis & object detection |
| **OpenAI Embeddings** | ~$0.10 | Image + text embeddings |
| **Pinecone** | Free tier | Vector storage & search |
| **Total** | **~$2.60** | Per 1000 images |

**For personal use**: Processing 100 images costs about **$0.26**

## 🎯 **AWS Benefits**

### **Performance Improvements**
- ✅ **Better OCR**: AWS Textract handles complex layouts better
- ✅ **Rich Analysis**: AWS Rekognition provides object detection
- ✅ **Scalability**: AWS infrastructure handles millions of images
- ✅ **Reliability**: 99.99% uptime SLA

### **User Experience**
- ✅ **Dual OCR**: Both Textract and Rekognition for better text extraction
- ✅ **Object Detection**: Understands what's in images, not just text
- ✅ **Scene Recognition**: Identifies contexts and activities
- ✅ **Higher Accuracy**: Enterprise-grade AWS AI services

## 🔧 **Troubleshooting**

### **Common Issues**

1. **"AWS credentials not found" error**
   - Check your `.env` file exists
   - Verify AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are filled
   - Ensure AWS_REGION is set (defaults to us-east-1)

2. **AWS permission errors**
   - Verify IAM user has Textract and Rekognition permissions
   - Check if services are available in your chosen region
   - Ensure your AWS account is active

3. **Pinecone connection error**
   - Check API key and environment
   - Ensure index name is unique
   - Verify your Pinecone account is active

### **Testing Individual APIs**

```python
# Test OpenAI
import openai
client = openai.OpenAI(api_key="your-key")
response = client.embeddings.create(model="text-embedding-3-small", input="test")
print("OpenAI: ✅ Working")

# Test AWS Textract
import boto3
client = boto3.client('textract', 
    aws_access_key_id="key", 
    aws_secret_access_key="secret", 
    region_name="us-east-1")
response = client.list_document_analysis_jobs(MaxResults=1)
print("AWS Textract: ✅ Working")

# Test AWS Rekognition
client = boto3.client('rekognition', 
    aws_access_key_id="key", 
    aws_secret_access_key="secret", 
    region_name="us-east-1")
response = client.list_collections(MaxResults=1)
print("AWS Rekognition: ✅ Working")

# Test Pinecone
import pinecone
pinecone.init(api_key="key", environment="env")
print("Pinecone: ✅ Working")
```

## 🚀 **Next Steps**

1. **Get your API keys** from the services above
2. **Create the `.env` file** with your keys
3. **Test APIs** with `python3 test_apis.py`
4. **Run the app** with `streamlit run streamlit_app.py`
5. **Enjoy enhanced image analysis!**

## 📞 **Need Help?**

- **OpenAI Issues**: [OpenAI Help Center](https://help.openai.com/)
- **AWS Issues**: [AWS Documentation](https://docs.aws.amazon.com/)
- **Pinecone Issues**: [Pinecone Support](https://docs.pinecone.io/)

---

**🎉 Congratulations!** You now have a production-ready, scalable visual memory search system powered by AWS, OpenAI, and Pinecone. The AWS services provide enhanced OCR capabilities and rich image analysis that will make your searches even more accurate and insightful.
