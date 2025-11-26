from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from backend.rag_pipeline import build_vector_db, generate_answer


class ChatRequest(BaseModel):
    query: str


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    
    allow_credentials=True,
    allow_methods=["*"],      
    allow_headers=["*"],      
)



@app.get("/")
def home():
    return {"status": "RAG chatbot running"}


@app.post("/build")
def build_index():
    try:
        status = build_vector_db()
        return {"status": status}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat")
def chat(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    try:
        answer = generate_answer(request.query)
        return {"response": answer}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
