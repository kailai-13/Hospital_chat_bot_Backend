import os
import tempfile
import gc
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv

import firebase_admin
from firebase_admin import credentials, storage

import jwt
from jwt.exceptions import InvalidTokenError

from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# =========================================================================
# CONFIGURATION
# =========================================================================
load_dotenv()

FRONTEND_URLS = [
    "https://hospital-chat-bot-frontend.vercel.app",
]
# You may add "http://localhost:5173" for local dev.

MAX_DOCS = int(os.getenv("MAX_DOCS", "2"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "300"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "250"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))

app = FastAPI(
    title="KG Hospital AI Chatbot API", 
    version="1.0.0",
    description="AI-powered chatbot system for KG Hospital (optimized for Render free plan)"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_URLS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "kg-hospital-secret-key-2024")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 480  # 8h

# =========================================================================
# FIREBASE INITIALIZATION
# =========================================================================
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
except Exception as e:
    print(f"❌ Firebase init failed: {e}")
    FIREBASE_INITIALIZED = False

vectorstore = None
conversation_chain = None
loaded_documents = []

# =========================================================================
# DATAMODELS
# =========================================================================
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

USERS_DB = {
    "admin": {"username": "admin", "password": "admin123", "role": "admin", "full_name": "Administrator"},
    "staff1": {"username": "staff1", "password": "staff123", "role": "staff", "full_name": "Hospital Staff"},
    "patient1": {"username": "patient1", "password": "patient123", "role": "patient", "full_name": "Patient User"},
    "visitor1": {"username": "visitor1", "password": "visitor123", "role": "visitor", "full_name": "Hospital Visitor"}
}

# =========================================================================
# AUTHENTICATION FUNCTIONS
# =========================================================================
def verify_password(plain: str, stored: str) -> bool:
    return plain == stored

def authenticate_user(username, password):
    user = USERS_DB.get(username)
    if not user or not verify_password(password, user["password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=15))
    to_encode.update({"exp": expire})
    try:
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    except Exception as e:
        print(f"JWT error: {e}")
        raise HTTPException(status_code=500, detail="Token creation failed")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        role = payload.get("role")
        if username is None:
            raise credentials_exception
        return {"username": username, "role": role}
    except InvalidTokenError:
        raise credentials_exception
    except Exception as e:
        print(f"Token verification error: {e}")
        raise credentials_exception

def require_admin_role(current_user: dict = Depends(verify_token)):
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

# =========================================================================
# DOCUMENT FUNCTIONS (MEMORY OPTIMIZED)
# =========================================================================
def load_document(file_path: str):
    file_name = os.path.basename(file_path)
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        if documents:
            print(f"✅ {file_name} loaded (PyPDFLoader)")
            return documents
    except Exception as e:
        print(f"⚠️ PyPDFLoader failed for {file_name}: {e}")
    try:
        loader = UnstructuredPDFLoader(file_path)
        documents = loader.load()
        if documents:
            print(f"✅ {file_name} loaded (UnstructuredPDFLoader)")
            return documents
    except Exception as e:
        print(f"⚠️ UnstructuredPDFLoader failed for {file_name}: {e}")
    raise Exception(f"❌ PDF failed: {file_name}")

def setup_vectorstore(documents):
    print(f"📄 Processing {len(documents)} pages...")
    splitter = CharacterTextSplitter(
        separator='\n', chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len
    )
    doc_chunks = splitter.split_documents(documents)[:MAX_CHUNKS]
    print(f"📝 {len(doc_chunks)} chunks made (max {MAX_CHUNKS})")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    gc.collect()
    return vectorstore

def create_chain(vectorstore):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    memory = ConversationBufferMemory(llm=llm, output_key='answer', memory_key='chat_history', return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=False,
        return_source_documents=False
    )
    return chain

def upload_file_to_firebase(file_path: str, file_name: str):
    if not FIREBASE_INITIALIZED:
        return False, "Firebase not initialized"
    try:
        blob = bucket.blob(f"documents/{file_name}")
        blob.upload_from_filename(file_path)
        print(f"✅ Uploaded {file_name} to Firebase")
        return True, f"File '{file_name}' uploaded"
    except Exception as e:
        print(f"❌ Upload failed for {file_name}: {e}")
        return False, str(e)

def list_firebase_files():
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
        print(f"❌ File list error: {e}")
        return []

def download_firebase_file(file_name: str):
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
    global vectorstore, conversation_chain, loaded_documents
    print("🔄 Reloading all documents from Firebase...")
    firebase_files = list_firebase_files()[:MAX_DOCS]  # only first N docs
    all_documents = []
    successful_loads = 0
    for file_info in firebase_files:
        file_name = file_info['name']
        print(f"📥 Processing {file_name}...")
        temp_file_path = download_firebase_file(file_name)
        if temp_file_path:
            try:
                docs = load_document(temp_file_path)
                all_documents.extend(docs)
                successful_loads += 1
                os.remove(temp_file_path)
            except Exception as e:
                print(f"❌ Failed: {file_name}: {e}")
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
    if all_documents:
        print(f"📚 {len(all_documents)} docs loaded")
        vectorstore = setup_vectorstore(all_documents)
        conversation_chain = create_chain(vectorstore)
        loaded_documents = all_documents
        gc.collect()
        return True, f"{successful_loads} of {len(firebase_files)} documents loaded"
    return False, "No documents loaded"

# =========================================================================
# API ENDPOINTS
# =========================================================================
@app.get("/")
async def root():
    return {
        "message": "KG Hospital AI Chatbot API",
        "status": "running",
        "version": "1.0.0",
        "firebase_initialized": FIREBASE_INITIALIZED,
        "documents_loaded": len(loaded_documents) > 0
    }

@app.post("/auth/login", response_model=Token)
async def login(user_credentials: UserLogin):
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
        print(f"✅ User {user['username']} logged in")
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
    return {
        "username": current_user["username"],
        "role": current_user["role"],
        "authenticated": True
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage, current_user: dict = Depends(verify_token)):
    global conversation_chain
    try:
        print(f"💬 Chat ({current_user['role']}): {message.message}")
        if conversation_chain:
            response = conversation_chain.invoke({'question': message.message})
            answer = response.get('answer', 'I could not find relevant information.')
        else:
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
            system_prompts = {
                "patient": "You are a helpful assistant for patients. Respond with compassion and clarity.",
                "visitor": "You are a helpful assistant for visitors. Respond helpfully and politely.",
                "staff":   "You are a helpful assistant for hospital staff. Be precise and professional.",
                "admin":   "You are a helpful assistant for administrators. Give comprehensive, analytical answers."
            }
            system_prompt = system_prompts.get(message.user_role, system_prompts["patient"])
            full_prompt = f"{system_prompt}\n\nUser Question: {message.message}\n\nResponse:"
            response = llm.invoke(full_prompt)
            answer = response.content
        return ChatResponse(response=answer, timestamp=datetime.now().isoformat())
    except Exception as e:
        print(f"❌ Chat error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating response: {str(e)}"
        )

@app.post("/admin/upload-document")
async def upload_document(file: UploadFile = File(...), current_user: dict = Depends(require_admin_role)):
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
                    "message": f"Document uploaded & processed: {message}",
                    "reload_status": reload_message,
                    "filename": file.filename
                }
            else:
                return {
                    "message": f"Uploaded but processing failed: {reload_message}",
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
    documents = list_firebase_files()
    return {
        "documents": documents,
        "count": len(documents),
        "firebase_status": FIREBASE_INITIALIZED
    }

@app.post("/admin/reload-documents")
async def reload_documents_endpoint(current_user: dict = Depends(require_admin_role)):
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
    return {
        "firebase_initialized": FIREBASE_INITIALIZED,
        "documents_loaded": len(loaded_documents),
        "vectorstore_ready": vectorstore is not None,
        "conversation_chain_ready": conversation_chain is not None,
        "groq_api_configured": bool(os.getenv("GROQ_API_KEY")),
        "timestamp": datetime.now().isoformat()
    }

@app.on_event("startup")
async def startup_event():
    print("🚀 KG Hospital Chatbot API backend ready!")
    print(f"🔧 Firebase: {'✅' if FIREBASE_INITIALIZED else '❌'}")
    print("⚡ No document vectorstore loaded at startup (admin triggers reload).")

PORT = int(os.getenv("PORT", 8000))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="info"
    )
