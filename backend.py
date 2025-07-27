import os
import sys
from fastapi import FastAPI, Request, status, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional
import traceback
import shutil
from pathlib import Path

# Load environment variables
load_dotenv()

# Import the RAG system
try:
    from main import RAGSystem
except ImportError as e:
    print(f"❌ Error importing RAGSystem: {e}")
    sys.exit(1)

app = FastAPI(title="RAG Q&A Backend API")

# Allow CORS for all origins (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG system instance
rag_system = None

# Create documents directory if it doesn't exist
documents_dir = Path("documents")
documents_dir.mkdir(exist_ok=True)

def initialize_rag():
    global rag_system
    if rag_system is None:
        try:
            print("🚀 Initializing RAG System...")
            rag_system = RAGSystem()
            rag_system.setup_documents()
            print("✅ RAG System initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ Error initializing RAG system: {e}")
            traceback.print_exc()
            return False
    return True

def reinitialize_rag():
    """Reinitialize RAG system after new document upload"""
    global rag_system
    rag_system = None
    return initialize_rag()

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    question: str
    answer: str
    status: str = "success"

class ErrorResponse(BaseModel):
    error: str

class UploadResponse(BaseModel):
    message: str
    filename: str
    file_size: int

@app.get("/")
def root():
    return {"message": "RAG Q&A Backend is running!"}

@app.post("/upload", response_model=UploadResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def upload_file(file: UploadFile = File(...)):
    # Check file type
    allowed_extensions = ['.pdf', '.docx', '.doc', '.txt']
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        return JSONResponse(
            status_code=400, 
            content={"error": f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}"}
        )
    
    try:
        # Save file to documents directory
        file_path = documents_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Reinitialize RAG system to include new document
        if reinitialize_rag():
            return {
                "message": "File uploaded successfully and RAG system updated!",
                "filename": file.filename,
                "file_size": file.size
            }
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "File uploaded but failed to update RAG system"}
            )
            
    except Exception as e:
        print(f"❌ Error uploading file: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Error uploading file: {str(e)}"}
        )

@app.post("/ask", response_model=AskResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
def ask_question(request: AskRequest):
    question = request.question.strip()
    if not question:
        return JSONResponse(status_code=400, content={"error": "Please enter a question."})
    if not initialize_rag():
        return JSONResponse(status_code=500, content={"error": "RAG system not initialized. Please check your configuration."})
    try:
        answer = rag_system.get_answer(question, k=3)
        return {"question": question, "answer": answer, "status": "success"}
    except Exception as e:
        print(f"❌ Error processing question: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Error processing your question: {str(e)}"})

@app.get("/health")
def health_check():
    if rag_system is None:
        return JSONResponse(status_code=503, content={"status": "not_initialized"})
    try:
        _ = rag_system.get_answer("test", k=1)
        return {"status": "healthy", "rag_system": "initialized", "message": "RAG system is working properly"}
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})

@app.get("/status")
def system_status():
    status_dict = {
        "rag_initialized": rag_system is not None,
        "documents_dir": "documents/",
        "index_name": "langchainvector"
    }
    # Check if documents directory exists
    status_dict["documents_exist"] = os.path.exists("documents/")
    if status_dict["documents_exist"]:
        pdf_files = [f for f in os.listdir("documents/") if f.endswith('.pdf')]
        status_dict["document_count"] = len(pdf_files)
        status_dict["documents"] = pdf_files[:5]
    else:
        status_dict["document_count"] = 0
        status_dict["documents"] = []
    # Check environment variables
    status_dict["pinecone_key_set"] = bool(os.getenv('PINECONE_API_KEY'))
    status_dict["google_key_set"] = bool(os.getenv('GOOGLE_API_KEY'))
    return status_dict
