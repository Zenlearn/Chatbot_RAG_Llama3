# Zenlearn RAG ChatBot

This is a FastAPI-based project, hosted on [GitHub](https://github.com/Zenlearn/Chatbot_RAG_Llama3/), that provides a Document Search API for uploading, processing, querying, and managing documents. It leverages text processing, language detection, translation, vector embeddings, and a vector database for efficient and intelligent document retrieval.

---

## Features

- **Upload Documents**: Upload documents in PDF or text format and process them into chunks for better storage and retrieval.
- **Language Detection**: Automatically detect the language of the document or query.
- **Translation**: Translate non-English text or queries into English for processing.
- **Embedding Generation**: Generate embeddings for chunks and queries to store and compare them in a vector database.
- **Search Query**: Query the document database to find relevant content.
- **Delete Documents**: Remove documents and their associated chunks from the database.

---

## File Structure

```plaintext
.
├── .venv/                      # Virtual environment folder
├── config/                     # Configuration files
│   ├── config_aws_bedrock.py   # AWS Bedrock configuration (if applicable)
│   ├── config_env.py           # Environment configuration (host, port, etc.)
│   ├── config_ollama.py        # Configuration for Ollama service
│   ├── config_qdrant.py        # Qdrant vector database configuration
├── database/                   # Database management
│   ├── vector_db_manager.py    # Vector database interaction logic
├── llm/                        # Language Model integration
│   ├── llm.py                  # Language model querying and response generation
├── pre_processing/             # Preprocessing utilities
│   ├── file_processing.py      # Text extraction and chunking logic
├── services/                   # Service modules
│   ├── embedding.py            # Embedding generation utilities
│   ├── query_model.py          # Query request data model
│   ├── translation.py          # Language detection and translation logic
├── test/                       # Test cases and testing scripts
├── .env                        # Environment variables file
├── .env.example                # Example environment variables template
├── .gitignore                  # Git ignore rules
├── LICENSE                     # License file
├── main.py                     # Entry point for the application
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```

---

## API Endpoints

### **1. Root**

- **Method**: `GET`
- **Path**: `/`
- **Description**: A simple welcome message for the API.

#### Response

```json
{
  "message": "Welcome to the Document Search API!"
}
```

### **2. Upload Document**

- **Method**: `POST`
- **Path**: `/upload`
- **Description**: Upload a document (PDF or text) and store it in chunks after processing.

#### Query Parameters

- **`singleChunk`** (optional): If `True`, processes the document in a single chunk. Default is `False`.
- **`priority`** (optional): Priority of the document (1-5). Default is `1`.

#### Request Body

- **File**: A PDF or plain text file.

#### Response

```json
{
  "message": "Document uploaded successfully",
  "doc_id": "document_id",
  "chunks": {
    "count": 5,
    "ids": ["chunk_id_1", "chunk_id_2", "..."]
  },
  "detected_language": "en"
}
```

### **3. Query Document**

- **Method**: `POST`
- **Path**: `/query`
- **Description**: Query the database and return relevant results.

#### Request Body

```json
{
  "query": "Your query text here"
}
```

#### Response

```json
{
  "message": "Query successful",
  "output": "Relevant results from the document",
  "detected_language": "en"
}
```

### **4. Delete Document**

- **Method**: `DELETE`
- **Path**: `/delete/{doc_id}`
- **Description**: Delete a document by its ID.

#### Path Parameters

- **`doc_id`**: The ID of the document to delete.

#### Response

```json
{
  "message": "Document deleted successfully",
  "success": true,
  "data": {
    "doc_id": "document_id",
    "chunks_deleted": ["chunk_id_1", "chunk_id_2", "..."]
  }
}
```

---

## Installation and Setup

### Prerequisites

- Python 3.8 or above.
- Ensure `pip` is installed.

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/lakshya324/QueryDocs.git
   cd QueryDocs
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

### Environment Variables

Create a `.env` file in the root directory based on the `.env.example` file. Below are the environment variables used in the project:

```plaintext
BHASHINI_API_URL=https://demo-api.models.ai4bharat.org/inference/translation/v2
BHASHINI_API_KEY=your_bhashini_api_key
QDRANT_HOST=https://your_qdrant_host
QDRANT_PORT=6333
QDRANT_API_KEY=your_qdrant_api_key
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=256
VECTOR_QUERY_SIZE=5
QDRANT_COLLECTION_NAME=gritworks
# LLAMA_MODEL=llama3.2
UVICORN_APP=main:app
HOST=localhost
PORT=8000
RELOAD=True
AWS_BEDROCK_MODEL_ID=meta.llama3-1-8b-instruct-v1:0
AWS_BEDROCK_SERVICE_NAME=bedrock-runtime
AWS_BEDROCK_REGION_NAME=us-west-2
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
```

### Key Notes

- Replace placeholders like `your_bhashini_api_key` and `your_qdrant_api_key` with actual API keys.
- Update `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` with your AWS credentials if using AWS services.

### Run the Application

```bash
python main.py
```

---

## Dependencies and Requirements

The project relies on several Python libraries to handle document processing, embeddings, vector storage, and serving the API. The complete list of dependencies can be found in `requirements.txt`. Below are the main packages used and their purposes:

### Key Libraries
- **FastAPI**: Framework for building the API.
- **Uvicorn**: ASGI server for serving FastAPI applications.
- **Qdrant-Client**: Client library for interacting with the Qdrant vector database.
- **Sentence-Transformers**: Used for embedding generation.
- **Transformers**: Hugging Face library for working with transformer-based models.
- **PyPDF2**: Extracts text from PDF files.
- **LangDetect**: Detects the language of the text.
- **Python-Dotenv**: Loads environment variables from `.env` files.
- **Boto3**: AWS SDK for Python, used for integrating with AWS Bedrock.

---

## Future Enhancements

1. Support for additional file formats like Word documents.
2. Advanced summarization for large files processed as a single chunk.
3. Multi-language support for queries and documents.
4. Improved query relevance ranking using external LLMs.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Author

Developed by [Lakshya](https://github.com/lakshya324). Contributions and feedback are welcome! 😊
