# Document AI RAG Web App

A web-based Retrieval-Augmented Generation (RAG) application for querying your own PDF documents using OpenAI's language models. Upload a PDF, ask questions, and get AI-powered answers based on your document content.

---

## Features
- **Upload PDF**: Drag and drop or select a PDF to analyze.
- **Ask Questions**: Query the content of your uploaded PDF using natural language.
- **AI-Powered Answers**: Uses OpenAI embeddings and LLMs for context-aware responses.
- **Conversation History**: View your previous questions and answers for each document.
- **Multiple Documents**: Manage and switch between multiple uploaded PDFs in your session.

---

## Quick Start

### 1. Clone the repository
```bash
git clone <repo-url>
cd document-ai-rag
```

### 2. Install dependencies
We recommend using Python 3.9+ and a virtual environment.

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Copy the example environment file and add your OpenAI API key:

```bash
cp env_example.txt .env
# Edit .env and set your actual OpenAI API key
```

Or manually create `.env` in the project root:
```
OPENAI_API_KEY=your-actual-api-key-here
```

---

## Running the App

Start the Flask web server:
```bash
python web_rag_app.py
```

The app will be available at [http://localhost:5000](http://localhost:5000).

---

## Usage
1. **Open the web app** in your browser.
2. **Upload a PDF** document.
3. Wait for processing (status will update to "ready").
4. **Ask questions** about the document in natural language.
5. View answers and conversation history.
6. Upload more PDFs and switch between them as needed.

---

## Project Structure
```
document-ai-rag/
├── web_rag_app.py        # Main Flask app
├── requirements.txt      # Python dependencies
├── env_example.txt       # Example environment file
├── uploads/              # Uploaded PDF files (auto-created)
├── chroma/               # Vector database storage
├── templates/
│   └── index.html        # Web UI template
└── README.md             # This file
```

---

## Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)

---

## Dependencies
- Flask
- langchain, langchain-community, langchain-openai
- chromadb
- openai
- python-dotenv
- pypdf

See `requirements.txt` for exact versions.

---

## Notes
- Only PDF files are supported for upload.
- Uploaded files are stored in the `uploads/` directory.
- Vector databases are stored in the `chroma/` directory.
- You must have an OpenAI API key to use the app.

---

## License
MIT License
# document-ai-rag
# document-ai-rag
