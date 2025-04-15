import os
import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader

# Local import
from get_relevant_documents import get_answer_from_llm

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the embeddings model
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Load .txt file documents
with open("data/RAG_document.txt", "r", encoding="utf-8") as file:
    documents = [file.read().strip()]

# Convert documents to embeddings
document_embeddings = embeddings.embed_documents(documents)
document_embeddings = np.array(document_embeddings).astype('float32')

# Initialize FAISS index and add document embeddings
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)

# Retriever class
class SimpleRetriever:
    def __init__(self, index, documents):
        self.index = index
        self.documents = documents

    def retrieve(self, query, top_k=1):
        query_embedding = np.array([embeddings.embed_query(query)]).astype('float32')
        _, indices = self.index.search(query_embedding, top_k)
        return [self.documents[i] for i in indices[0]]

retriever = SimpleRetriever(index, documents)

# RAG class
class SimpleRAG:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def generate(self, query):
        retrieved_docs = self.retriever.retrieve(query)
        augmented_query = f"Context: {' '.join(retrieved_docs)} Query: {query}"
        response = self.llm.invoke(augmented_query)
        return response

llm = OpenAI(api_key=openai_api_key)
rag = SimpleRAG(llm, retriever)

# FastAPI app
app = FastAPI()

# Pydantic schemas
class QueryRequest(BaseModel):
    query: str

@app.post("/generate")
def generate_answer(request: QueryRequest):
    """RAG document response"""
    response = rag.generate(request.query)
    return {"response": response}

@app.post("/csv-query")
def csv_based_answer(request: QueryRequest):
    """CSV-based answer using get_relevant_documents"""
    csv_file_path = "./data/organisation_data.csv"
    loader = CSVLoader(file_path=csv_file_path)
    documents = loader.load()
    response = get_answer_from_llm(
        documents=documents,
        question=request.query,
    )
    return {"response": response}
