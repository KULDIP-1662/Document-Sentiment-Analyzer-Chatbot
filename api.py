from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from chat import ChatbotLogic
from pdf_processing import load_pdf
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from fastapi import FastAPI, File, UploadFile
import config
import torch
import dill
import torch.nn.functional as F
from transformers import AutoTokenizer

from sentiment.text_extraction import get_sentiment

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.state.retriever = None
embedding_model = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
llm = ChatGoogleGenerativeAI(model=config.GENERATION_MODEL, temperature=0.8)
sentiment_model = torch.load(config.FINE_TUNED_MODEL, pickle_module=dill,map_location=torch.device('cpu'))
tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)
 
class ChatRequest(BaseModel):
    message: str
    history: List[List[str]] = []

class ChatResponse(BaseModel):
    response: str

# @app.post("/process_pdf")
# async def process_pdf(file: UploadFile = File(...)):
#     print(file.filename)

@app.post("/process_pdf")
async def process_pdf(file: UploadFile = File(...)):
    # print(file.filename)
    app.state.retriever  = await load_pdf(file, file.filename, embedding_model)
    # print(app.state.retriever)

    return {"message": "PDF processed successfully!"}


@app.post("/chat/")
async def chat(request: ChatRequest):
    if app.state.retriever  is None:
        raise HTTPException(status_code=400, detail="No PDF uploaded.")
# 
    chatbot = ChatbotLogic(llm, app.state.retriever )
    # print("request.message: ", request.message)
    # print("request.history: ", request.history)
    response = chatbot.query(request.message, request.history)
    # print("response: ", response)
    return ChatResponse(response=response)

@app.post("/sentiment/")
async def sentiment(file: UploadFile = File(...)):

    pdf_sentiment  = await get_sentiment(file, sentiment_model, tokenizer)
    # print(pdf_sentiment)
    return pdf_sentiment

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7000)