# Study Assist API

A RAG-powered study assistant backend API for learning from course documents. This FastAPI application enables users to upload documents (PDFs, images, URLs), automatically generate structured notebooks with pages, create flashcards, and interact with an AI assistant powered by Retrieval Augmented Generation (RAG).

## Features

### 📚 Document Management
- **Multi-format Support**: Upload PDFs, images, and URLs
- **Intelligent Processing**: Extract structured content from documents using OCR and text extraction
- **Automatic Organization**: Documents are organized by notebook and stored securely

### 📝 Notebook & Page Generation
- **AI-Powered Notebook Creation**: Automatically generates notebook titles and descriptions from document content
- **Smart Page Generation**: Extracts table of contents and creates comprehensive pages using LLM
- **Rich Text Editor**: Pages support TipTap/ProseMirror JSON format with markdown conversion
- **Parallel Processing**: Generates multiple pages concurrently for faster processing

### 🧠 RAG-Powered AI Assistant
- **Context-Aware Chat**: Chat with AI assistant that retrieves relevant context from your documents
- **Semantic Search**: Uses FAISS vector store with OpenAI embeddings for efficient document retrieval
- **User-Scoped Storage**: Each user has their own vector store for privacy and isolation
- **Intelligent Retrieval**: Combines selected text with document context for comprehensive answers

### 🎴 Flashcard System
- **AI-Generated Flashcards**: Automatically creates flashcards from selected text using RAG and LLM
- **Smart Answer Evaluation**: Uses LLM for semantic answer comparison (not just exact match)
- **Progress Tracking**: Tracks how many times flashcards are answered correctly
- **Context-Aware**: Uses RAG to enhance flashcard generation with relevant document context

### 🔐 Authentication & Security
- **JWT-Based Auth**: Secure token-based authentication
- **Password Hashing**: Uses bcrypt for secure password storage
- **User Isolation**: All data is scoped to individual users

## Tech Stack

### Core Framework
- **FastAPI**: Modern, fast web framework for building APIs
- **Python 3.12+**: Latest Python features and performance improvements

### Database
- **MongoDB**: NoSQL database for flexible document storage
- **Motor**: Async MongoDB driver for FastAPI

### AI & ML
- **OpenAI API**: GPT models for content generation and chat
- **LangChain**: Framework for building LLM applications
- **FAISS**: Vector similarity search for RAG
- **OpenAI Embeddings**: Text embeddings for semantic search

### Document Processing
- **pdfplumber**: PDF text extraction
- **Pillow**: Image processing
- **pytesseract**: OCR for image text extraction
- **BeautifulSoup4**: HTML parsing for URL content

### Utilities
- **Pydantic**: Data validation and settings management
- **python-jose**: JWT token handling
- **bcrypt**: Password hashing
- **uvicorn**: ASGI server

## Architecture

### Project Structure

```
backend/
├── main.py                 # Application entry point
├── pyproject.toml          # Dependencies and project config
├── src/
│   ├── app.py             # FastAPI app factory and setup
│   ├── config.py          # Configuration management
│   ├── database.py        # MongoDB connection
│   ├── auth.py            # Authentication utilities
│   ├── models/            # Pydantic models
│   │   ├── user.py        # User models
│   │   └── notebook.py    # Notebook, page, flashcard models
│   ├── routers/           # API route handlers
│   │   ├── auth.py        # Authentication endpoints
│   │   ├── notebooks.py   # Notebook and page management
│   │   ├── pages.py       # Page content endpoints
│   │   └── flashcards.py  # Flashcard endpoints
│   ├── services/          # Business logic
│   │   ├── document_processor.py  # Document ingestion
│   │   └── llm_service.py         # LLM content generation
│   └── rag/               # RAG system
│       ├── store.py       # Vector store (FAISS)
│       ├── query.py       # RAG query service
│       └── ingestion/      # Document ingestion modules
│           ├── pdf.py     # PDF processing
│           ├── image.py   # Image OCR
│           └── url.py     # URL content extraction
└── data/                  # Data storage
    ├── notebooks/         # Uploaded documents
    └── vector_stores/     # FAISS indices per user
```

### Key Components

#### 1. Document Processing Pipeline
1. **Upload**: Documents are uploaded and saved to disk
2. **Ingestion**: Content is extracted (PDF text, image OCR, URL scraping)
3. **Chunking**: Text is split into semantic chunks
4. **Embedding**: Chunks are embedded using OpenAI embeddings
5. **Storage**: Embeddings stored in FAISS vector store per user

#### 2. Notebook Generation Flow
1. **Document Analysis**: Extract table of contents from documents using LLM
2. **Metadata Generation**: Generate notebook title and description
3. **Page Creation**: For each TOC item, generate comprehensive page content
4. **Parallel Processing**: Generate multiple pages concurrently
5. **Storage**: Save pages to MongoDB with TipTap JSON format

#### 3. RAG Query Flow
1. **Query Embedding**: Convert user query to embedding vector
2. **Similarity Search**: Search FAISS index for relevant chunks
3. **Context Retrieval**: Retrieve top-k most relevant document chunks
4. **Response Generation**: Use LLM to generate answer from context

#### 4. Flashcard Generation
1. **Content Selection**: User selects text from page
2. **RAG Context**: Retrieve related content from documents
3. **LLM Generation**: Generate question, answer, and explanation
4. **Storage**: Save flashcard with tracking metadata

## Setup Instructions

### Prerequisites

- Python 3.12 or higher
- MongoDB (local or remote)
- OpenAI API key
- Tesseract OCR (for image processing)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd backend
   ```

2. **Install dependencies**
   ```bash
   # Using uv (recommended)
   uv sync
   
   # Or using pip
   pip install -e .
   ```

3. **Install Tesseract OCR** (for image processing)
   ```bash
   # macOS
   brew install tesseract
   
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr
   
   # Windows
   # Download from: https://github.com/UB-Mannheim/tesseract/wiki
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   # OpenAI Configuration
   OPENAI_API_KEY=your_openai_api_key_here
   LLM_MODEL=gpt-4o-mini
   EMBEDDING_MODEL=text-embedding-3-small
   TEMPERATURE=0.7
   
   # MongoDB Configuration
   MONGODB_URL=mongodb://localhost:27017
   MONGODB_DB_NAME=study_assist
   
   # JWT Configuration
   JWT_SECRET_KEY=your-secret-key-change-in-production
   JWT_ALGORITHM=HS256
   JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
   
   # RAG Configuration
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   TOP_K_RETRIEVAL=5
   ```

5. **Start MongoDB**
   ```bash
   # Using Docker
   docker run -d -p 27017:27017 --name mongodb mongo:latest
   
   # Or use your existing MongoDB instance
   ```

6. **Run the application**
   ```bash
   python main.py
   ```
   
   The API will be available at `http://localhost:8000`

7. **Access API Documentation**
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for LLM and embeddings | Required |
| `LLM_MODEL` | OpenAI model for text generation | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | OpenAI model for embeddings | `text-embedding-3-small` |
| `TEMPERATURE` | LLM temperature (0-1) | `0.7` |
| `MONGODB_URL` | MongoDB connection string | `mongodb://localhost:27017` |
| `MONGODB_DB_NAME` | Database name | `study_assist` |
| `JWT_SECRET_KEY` | Secret key for JWT tokens | Required |
| `JWT_ALGORITHM` | JWT algorithm | `HS256` |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | Token expiration time | `30` |
| `CHUNK_SIZE` | Text chunk size for RAG | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `TOP_K_RETRIEVAL` | Default number of RAG results | `5` |

## API Endpoints

### Authentication

- `POST /auth/signup` - Register a new user
- `POST /auth/login` - Login and get access token
- `GET /auth/me` - Get current user information

### Notebooks

- `GET /notebooks` - List all notebooks for current user
- `GET /notebooks/{notebook_id}` - Get notebook details with pages
- `POST /notebooks` - Create a new notebook (with optional documents)
- `PATCH /notebooks/{notebook_id}` - Update notebook metadata
- `DELETE /notebooks/{notebook_id}` - Delete notebook and all pages

### Pages

- `GET /notebooks/{notebook_id}/pages` - List pages in a notebook
- `GET /notebooks/{notebook_id}/pages/{page_id}` - Get page details
- `POST /notebooks/{notebook_id}/pages` - Create a new page
- `PATCH /notebooks/{notebook_id}/pages/{page_id}` - Update page content
- `DELETE /notebooks/{notebook_id}/pages/{page_id}` - Delete a page
- `POST /notebooks/{notebook_id}/pages/{page_id}/chat` - Chat with AI assistant

### Flashcards

- `GET /flashcards` - List flashcards (optionally filtered by notebook)
- `POST /flashcards` - Create a flashcard from selected text
- `POST /flashcards/{flashcard_id}/submit` - Submit answer and get feedback
- `DELETE /flashcards/{flashcard_id}` - Delete a flashcard

### Pages (Legacy)

- `GET /pages` - List all pages
- `GET /pages/{page_id}/content` - Get page content
- `POST /pages/{page_id}/content` - Save page content
- `DELETE /pages/{page_id}` - Delete a page

## Usage Examples

### 1. Create a Notebook with Documents

```python
import requests

# Login
response = requests.post("http://localhost:8000/auth/login", json={
    "email": "user@example.com",
    "password": "password123"
})
token = response.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# Create notebook with PDF
with open("document.pdf", "rb") as f:
    pdf_data = base64.b64encode(f.read()).decode()
    
response = requests.post("http://localhost:8000/notebooks", 
    json={
        "title": "My Study Notebook",
        "documents": [{
            "type": "pdf",
            "name": "document.pdf",
            "data": pdf_data
        }]
    },
    headers=headers
)
notebook = response.json()
```

### 2. Chat with AI Assistant

```python
response = requests.post(
    f"http://localhost:8000/notebooks/{notebook_id}/pages/{page_id}/chat",
    json={
        "message": "Explain the key concepts in this section",
        "context": "Selected text from page...",
        "conversation_id": "conv_123"
    },
    headers=headers
)
chat_response = response.json()
```

### 3. Create Flashcard

```python
response = requests.post(
    "http://localhost:8000/flashcards",
    json={
        "notebookId": notebook_id,
        "pageId": page_id,
        "content": "Selected text to create flashcard from"
    },
    headers=headers
)
flashcard = response.json()
```

## Data Models

### Notebook
- `id`: Unique identifier
- `title`: Notebook title
- `description`: Optional description
- `documents`: List of attached documents
- `pagesCount`: Number of pages
- `documentsCount`: Number of documents
- `flashCardsCount`: Number of flashcards
- `createdAt`: Creation timestamp
- `updatedAt`: Last update timestamp

### Page
- `id`: Unique identifier
- `notebookId`: Parent notebook ID
- `title`: Page title
- `content`: TipTap JSON content
- `createdAt`: Creation timestamp
- `updatedAt`: Last update timestamp

### Flashcard
- `id`: Unique identifier
- `notebookId`: Associated notebook ID
- `question`: Question text
- `answer`: Answer text
- `explanation`: Additional explanation
- `timesAnswered`: Total attempts
- `timesAnsweredCorrectly`: Correct attempts

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest
```

### Code Style

The project follows PEP 8 style guidelines. Consider using:
- `black` for code formatting
- `flake8` for linting
- `mypy` for type checking

### Adding New Document Types

To add support for new document types:

1. Create a new ingestion module in `src/rag/ingestion/`
2. Implement the `ingest()` function that returns `{"titles": str, "markdown": str}`
3. Update `src/services/document_processor.py` to handle the new type

## Performance Considerations

- **Vector Store**: FAISS indices are stored per-user for isolation and performance
- **Parallel Processing**: Page generation runs in parallel (default: 5 concurrent)
- **Chunking Strategy**: Documents are chunked with overlap for better context retention
- **Caching**: Consider implementing Redis for frequently accessed data

## Security Considerations

- **Password Hashing**: Uses bcrypt with salt
- **JWT Tokens**: Secure token-based authentication
- **User Isolation**: All data is scoped to users
- **Input Validation**: Pydantic models validate all inputs
- **CORS**: Configured for specific origins (update in production)

## Troubleshooting

### Common Issues

1. **FAISS Import Error**
   ```bash
   pip install faiss-cpu
   ```

2. **Tesseract Not Found**
   - Ensure Tesseract is installed and in PATH
   - On macOS: `brew install tesseract`

3. **MongoDB Connection Error**
   - Verify MongoDB is running
   - Check `MONGODB_URL` in `.env`

4. **OpenAI API Errors**
   - Verify `OPENAI_API_KEY` is set correctly
   - Check API rate limits and quotas

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Support

For issues and questions, please open an issue on GitHub or contact [your contact information].

