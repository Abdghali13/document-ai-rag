from flask import Flask, render_template, request, jsonify
import os
import tempfile
import shutil
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import uuid
from datetime import datetime

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your-secret-key-here'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for session management
sessions = {}
uploaded_files = []  # Track all uploaded files

class RAGSession:
    def __init__(self):
        self.vector_db = None
        self.qa_chain = None
        self.pdf_path = None
        self.status = "ready"
        self.conversation_history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        sessions[session_id] = RAGSession()
        
        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        file.save(file_path)
        
        # Add to uploaded files list
        file_info = {
            'session_id': session_id,
            'filename': filename,
            'original_filename': file.filename,
            'upload_time': datetime.now().isoformat(),
            'file_size': os.path.getsize(file_path),
            'status': 'processing'
        }
        uploaded_files.append(file_info)
        
        sessions[session_id].pdf_path = file_path
        sessions[session_id].status = "processing"
        
        # Build database in background
        import threading
        threading.Thread(target=build_database, args=(session_id,), daemon=True).start()
        
        return jsonify({
            'session_id': session_id,
            'filename': filename,
            'status': 'uploaded'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def build_database(session_id):
    """Build vector database from PDF"""
    try:
        session = sessions[session_id]
        session.status = "building_database"
        
        # Load PDF
        loader = PyPDFLoader(session.pdf_path)
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Create vector database
        embeddings = OpenAIEmbeddings()
        session.vector_db = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=f"./chroma_db_{session_id}"
        )
        
        # Create QA chain
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        session.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=session.vector_db.as_retriever(search_kwargs={"k": 3})
        )
        
        session.status = "ready"
        
        # Update file status in uploaded_files
        for file_info in uploaded_files:
            if file_info['session_id'] == session_id:
                file_info['status'] = 'ready'
                break
        
    except Exception as e:
        session.status = "error"
        
        # Update file status in uploaded_files
        for file_info in uploaded_files:
            if file_info['session_id'] == session_id:
                file_info['status'] = 'error'
                break
                
        print(f"Error building database: {str(e)}")

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        question = data.get('question', '').strip()
        
        if not session_id or session_id not in sessions:
            return jsonify({'error': 'Invalid session'}), 400
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        session = sessions[session_id]
        
        if session.status != "ready":
            return jsonify({'error': 'Database not ready yet'}), 400
        
        if not session.qa_chain:
            return jsonify({'error': 'No QA chain available'}), 400
        
        # Get answer
        result = session.qa_chain.invoke({"query": question})
        answer = result["result"]
        
        # Add to conversation history
        conversation_entry = {
            'id': len(session.conversation_history) + 1,
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        }
        session.conversation_history.append(conversation_entry)
        
        return jsonify({
            'answer': answer,
            'question': question,
            'conversation_id': conversation_entry['id']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status/<session_id>')
def get_status(session_id):
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400
    
    session = sessions[session_id]
    return jsonify({'status': session.status})

@app.route('/history/<session_id>')
def get_history(session_id):
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400
    
    session = sessions[session_id]
    return jsonify({
        'history': session.conversation_history,
        'total_questions': len(session.conversation_history)
    })

@app.route('/files')
def get_uploaded_files():
    return jsonify({
        'files': uploaded_files,
        'total_files': len(uploaded_files)
    })

@app.route('/switch_session/<session_id>')
def switch_session(session_id):
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400
    
    session = sessions[session_id]
    if session.status != "ready":
        return jsonify({'error': 'Session not ready'}), 400
    
    return jsonify({
        'session_id': session_id,
        'status': 'ready',
        'message': 'Session switched successfully'
    })

if __name__ == '__main__':
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set your OPENAI_API_KEY in the .env file")
        exit(1)
    
    app.run(debug=True, host='0.0.0.0', port=3000) 