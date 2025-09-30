# main.py - Complete Fixed Version

import os
import tempfile
import time
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, storage
from dotenv import load_dotenv

# Fixed JWT imports
import jwt
from jwt.exceptions import InvalidTokenError

# LangChain imports
from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# =============================================================================
# CONFIGURATION & INITIALIZATION
# =============================================================================
load_dotenv()

app = FastAPI(
    title="KG Hospital AI Chatbot API", 
    version="1.0.0",
    description="AI-powered chatbot system for KG Hospital with role-based access control"
)

security = HTTPBearer()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "kg-hospital-secret-key-2024")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 480  # 8 hours

# Initialize Firebase Admin SDK
try:
    if not firebase_admin._apps:
        firebase_config = {
            "type": "service_account",
            "project_id": os.getenv("FIREBASE_PROJECT_ID"),
            "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
            "private_key": os.getenv("FIREBASE_PRIVATE_KEY", "").replace('\\n', '\n'),
            "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
            "client_id": os.getenv("FIREBASE_CLIENT_ID"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        
        cred = credentials.Certificate(firebase_config)
        firebase_admin.initialize_app(cred, {
            'storageBucket': f"{firebase_config['project_id']}.firebasestorage.app"
        })
    
    bucket = storage.bucket()
    FIREBASE_INITIALIZED = True
    print("✅ Firebase initialized successfully")
except Exception as e:
    print(f"❌ Firebase initialization failed: {e}")
    FIREBASE_INITIALIZED = False

# Global variables for chatbot
vectorstore = None
conversation_chain = None
loaded_documents = []

# =============================================================================
# PYDANTIC MODELS
# =============================================================================
class UserLogin(BaseModel):
    username: str
    password: str

class ChatMessage(BaseModel):
    message: str
    user_role: str

class ChatResponse(BaseModel):
    response: str
    timestamp: str

class Token(BaseModel):
    access_token: str
    token_type: str
    role: str
    username: str

# =============================================================================
# USER DATABASE
# =============================================================================
USERS_DB = {
    "admin": {
        "username": "admin",
        "password": "admin123",
        "role": "admin",
        "full_name": "Administrator"
    },
    "staff1": {
        "username": "staff1",
        "password": "staff123",
        "role": "staff",
        "full_name": "Hospital Staff"
    },
    "patient1": {
        "username": "patient1", 
        "password": "patient123",
        "role": "patient",
        "full_name": "Patient User"
    },
    "visitor1": {
        "username": "visitor1",
        "password": "visitor123", 
        "role": "visitor",
        "full_name": "Hospital Visitor"
    }
}

# =============================================================================
# AUTHENTICATION FUNCTIONS - COMPLETELY FIXED
# =============================================================================
def verify_password(plain_password: str, stored_password: str) -> bool:
    """Simple password verification."""
    return plain_password == stored_password

def authenticate_user(username: str, password: str):
    """Authenticate user credentials."""
    user = USERS_DB.get(username)
    if not user:
        return False
    if not verify_password(password, user["password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    
    try:
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    except Exception as e:
        print(f"JWT encoding error: {e}")
        raise HTTPException(status_code=500, detail="Token creation failed")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None:
            raise credentials_exception
        return {"username": username, "role": role}
    except InvalidTokenError:
        raise credentials_exception
    except Exception as e:
        print(f"Token verification error: {e}")
        raise credentials_exception

def require_admin_role(current_user: dict = Depends(verify_token)):
    """Require admin role for protected endpoints."""
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

# =============================================================================
# DOCUMENT PROCESSING FUNCTIONS
# =============================================================================
def load_document(file_path: str):
    """Load and process PDF document with multiple fallback methods."""
    documents = []
    file_name = os.path.basename(file_path)
    
    try:
        loader = UnstructuredPDFLoader(file_path)
        documents = loader.load()
        if documents:
            print(f"✅ Loaded {file_name} using UnstructuredPDFLoader")
            return documents
    except Exception as e:
        print(f"⚠️ UnstructuredPDFLoader failed for {file_name}: {e}")
    
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        if documents:
            print(f"✅ Loaded {file_name} using PyPDFLoader")
            return documents
    except Exception as e:
        print(f"⚠️ PyPDFLoader failed for {file_name}: {e}")
    
    raise Exception(f"❌ All PDF processing methods failed for {file_name}")

def setup_vectorstore(documents):
    """Create FAISS vectorstore with optimized settings."""
    if not documents:
        raise ValueError("No documents provided for vectorstore creation")
    
    print(f"📄 Processing {len(documents)} document pages...")
    
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    
    doc_chunks = text_splitter.split_documents(documents)
    print(f"📝 Created {len(doc_chunks)} text chunks")
    
    if len(doc_chunks) > 2000:
        print(f"⚡ Large document detected. Limiting to 2000 chunks for performance.")
        doc_chunks = doc_chunks[:2000]
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print("🔄 Creating vector store...")
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    print("✅ Vector store created successfully!")
    
    return vectorstore

def create_chain(vectorstore):
    """Create conversational retrieval chain."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    memory = ConversationBufferMemory(
        llm=llm,
        output_key='answer',
        memory_key='chat_history',
        return_messages=True
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=False,
        return_source_documents=False
    )
    
    return chain

# =============================================================================
# FIREBASE FUNCTIONS
# =============================================================================
def upload_file_to_firebase(file_path: str, file_name: str):
    """Upload file to Firebase Storage."""
    if not FIREBASE_INITIALIZED:
        return False, "Firebase not initialized"
    
    try:
        blob = bucket.blob(f"documents/{file_name}")
        blob.upload_from_filename(file_path)
        print(f"✅ Uploaded {file_name} to Firebase Storage")
        return True, f"File '{file_name}' uploaded successfully"
    except Exception as e:
        print(f"❌ Upload failed for {file_name}: {e}")
        return False, f"Upload failed: {str(e)}"

def list_firebase_files():
    """List all PDF files in Firebase Storage."""
    if not FIREBASE_INITIALIZED:
        return []
    
    try:
        blobs = bucket.list_blobs(prefix="documents/")
        files_info = []
        
        for blob in blobs:
            if blob.name.lower().endswith('.pdf'):
                files_info.append({
                    'name': blob.name.replace('documents/', ''),
                    'size': blob.size or 0,
                    'created': blob.time_created.isoformat() if blob.time_created else '',
                    'status': 'loaded'
                })
        
        return files_info
    except Exception as e:
        print(f"❌ Error listing files: {e}")
        return []

def download_firebase_file(file_name: str):
    """Download file from Firebase Storage."""
    if not FIREBASE_INITIALIZED:
        return None
    
    try:
        blob = bucket.blob(f"documents/{file_name}")
        if not blob.exists():
            return None
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file_path = temp_file.name
        temp_file.close()
        
        blob.download_to_filename(temp_file_path)
        return temp_file_path
    except Exception as e:
        print(f"❌ Download failed for {file_name}: {e}")
        return None

def reload_all_documents():
    """Reload all documents from Firebase and update vectorstore."""
    global vectorstore, conversation_chain, loaded_documents
    
    print("🔄 Reloading all documents from Firebase...")
    firebase_files = list_firebase_files()
    if not firebase_files:
        return False, "No documents found in Firebase"
    
    all_documents = []
    successful_loads = 0
    
    for file_info in firebase_files:
        file_name = file_info['name']
        print(f"📥 Processing {file_name}...")
        
        temp_file_path = download_firebase_file(file_name)
        if temp_file_path:
            try:
                documents = load_document(temp_file_path)
                all_documents.extend(documents)
                successful_loads += 1
                os.remove(temp_file_path)
            except Exception as e:
                print(f"❌ Failed to process {file_name}: {e}")
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
    
    if all_documents:
        print(f"📚 Total documents loaded: {len(all_documents)}")
        vectorstore = setup_vectorstore(all_documents)
        conversation_chain = create_chain(vectorstore)
        loaded_documents = all_documents
        return True, f"Successfully loaded {successful_loads} out of {len(firebase_files)} documents"
    
    return False, "No documents could be processed"

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint with API status."""
    return {
        "message": "KG Hospital AI Chatbot API", 
        "status": "running",
        "version": "1.0.0",
        "firebase_initialized": FIREBASE_INITIALIZED,
        "documents_loaded": len(loaded_documents) > 0
    }

@app.post("/auth/login", response_model=Token)
async def login(user_credentials: UserLogin):
    """User login endpoint."""
    try:
        user = authenticate_user(user_credentials.username, user_credentials.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["username"], "role": user["role"]},
            expires_delta=access_token_expires
        )
        
        print(f"✅ User {user['username']} logged in successfully")
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "role": user["role"],
            "username": user["username"]
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login processing failed"
        )

@app.get("/auth/verify")
async def verify_auth(current_user: dict = Depends(verify_token)):
    """Verify authentication token."""
    return {
        "username": current_user["username"], 
        "role": current_user["role"],
        "authenticated": True
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage, current_user: dict = Depends(verify_token)):
    """Chat endpoint with role-based responses."""
    global conversation_chain
    
    try:
        print(f"💬 Chat request from {current_user['username']} ({current_user['role']}): {message.message}")
        
        if conversation_chain:
            response = conversation_chain.invoke({'question': message.message})
            answer = response.get('answer', 'I could not find relevant information.')
        else:
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
            
            system_prompts = {
                "patient": """You are a helpful KG Hospital AI assistant helping patients. 
                Provide information about:
                - Doctor appointments and specializations
                - Hospital services and departments  
                - Treatment information and medical procedures
                - Emergency contacts and protocols
                Always be compassionate and professional.""",
                
                "visitor": """You are a helpful KG Hospital AI assistant helping visitors.
                Provide information about:
                - Visiting hours and policies
                - Hospital location and directions
                - Parking information and facilities
                - Hospital amenities and services
                Be welcoming and informative.""",
                
                "staff": """You are a helpful KG Hospital AI assistant helping hospital staff.
                Provide information about:
                - Patient inquiry responses
                - Department information and contacts
                - Emergency protocols and procedures
                - Hospital policies and guidelines
                Be efficient and professional.""",
                
                "admin": """You are a helpful KG Hospital AI assistant helping administrators.
                Provide information about:
                - Hospital operations and management
                - System status and analytics
                - Administrative procedures
                - Staff coordination and policies
                Be comprehensive and analytical."""
            }
            
            system_prompt = system_prompts.get(message.user_role, system_prompts["patient"])
            full_prompt = f"{system_prompt}\n\nUser Question: {message.message}\n\nResponse:"
            
            response = llm.invoke(full_prompt)
            answer = response.content
        
        return ChatResponse(
            response=answer,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        print(f"❌ Chat error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Error generating response: {str(e)}"
        )

@app.post("/admin/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(require_admin_role)
):
    """Admin-only document upload endpoint."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed"
        )
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    temp_file_path = temp_file.name
    
    try:
        content = await file.read()
        temp_file.write(content)
        temp_file.close()
        
        success, message = upload_file_to_firebase(temp_file_path, file.filename)
        
        if success:
            reload_success, reload_message = reload_all_documents()
            os.remove(temp_file_path)
            
            if reload_success:
                return {
                    "message": f"Document uploaded and processed successfully: {message}",
                    "reload_status": reload_message,
                    "filename": file.filename
                }
            else:
                return {
                    "message": f"Document uploaded but processing failed: {reload_message}",
                    "filename": file.filename
                }
        else:
            os.remove(temp_file_path)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=message
            )
            
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )

@app.get("/admin/documents")
async def list_documents(current_user: dict = Depends(require_admin_role)):
    """Admin-only endpoint to list all documents."""
    documents = list_firebase_files()
    return {
        "documents": documents, 
        "count": len(documents),
        "firebase_status": FIREBASE_INITIALIZED
    }

@app.post("/admin/reload-documents")
async def reload_documents_endpoint(current_user: dict = Depends(require_admin_role)):
    """Admin-only endpoint to reload all documents."""
    success, message = reload_all_documents()
    if success:
        return {
            "message": message, 
            "status": "success",
            "documents_loaded": len(loaded_documents)
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=message
        )

@app.get("/system/status")
async def system_status():
    """System status endpoint."""
    return {
        "firebase_initialized": FIREBASE_INITIALIZED,
        "documents_loaded": len(loaded_documents),
        "vectorstore_ready": vectorstore is not None,
        "conversation_chain_ready": conversation_chain is not None,
        "groq_api_configured": bool(os.getenv("GROQ_API_KEY")),
        "timestamp": datetime.now().isoformat()
    }

# =============================================================================
# STARTUP EVENT
# =============================================================================
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    print("🚀 Starting KG Hospital Chatbot API...")
    print(f"🔧 Firebase Status: {'✅ Connected' if FIREBASE_INITIALIZED else '❌ Not Connected'}")
    
    if FIREBASE_INITIALIZED:
        print("📚 Loading initial documents...")
        success, message = reload_all_documents()
        if success:
            print(f"✅ {message}")
        else:
            print(f"⚠️ {message}")
    
    print("🎉 KG Hospital Chatbot API is ready!")

# =============================================================================
# MAIN
# =============================================================================
# =============================================================================
# PRODUCTION CONFIGURATION
# =============================================================================
import os

# Get port from environment variable (Render sets this)
PORT = int(os.getenv("PORT", 8000))

# Configure CORS for production
if os.getenv("RENDER"):
    # Production CORS settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://hospital-chat-bot-frontend.vercel.app/",  # Update with your actual frontend URL
            "https://your-custom-domain.com",           # Add your custom domain if any
            "http://localhost:5173",                    # Keep for development
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=PORT,
        log_level="info"
    )
