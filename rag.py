import os
import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from sqlite3_langchain import db, insert_employee 
from get_relevant_documents import get_answer_from_llm
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from llama_index.llms.openai import OpenAI

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
        response = self.llm.complete(augmented_query)
        return response.text  # ✅ Fixed to return plain string

llm = OpenAI(api_key=openai_api_key)
rag = SimpleRAG(llm, retriever)

# FastAPI app
app = FastAPI()

# Pydantic schemas
class QueryRequest(BaseModel):
    query: str

class EmployeeRequest(BaseModel):
    name: str
    age: int
    department: str

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

@app.post("/db-query")
def db_query(request: QueryRequest):
    """SQL query-based answer using database"""
    llm_for_sql = ChatOpenAI(api_key=openai_api_key)
    chain = create_sql_query_chain(llm_for_sql, db)

    # Generate SQL from LLM
    sql_query = chain.invoke({"question": request.query})

    # Optional: log it for debugging
    print("Generated SQL:", sql_query)

    # ⚠️ Fix for multiple statements — keep only the first
    sanitized_sql = sql_query.strip().split(";")[0]

    # Run the sanitized query
    result = db.run(sanitized_sql)

    return {"response": result}


@app.post("/add-employee")
def add_employee(request: EmployeeRequest):
    """Add new employee to SQLite database, avoiding duplicates"""
    insert_employee(request.name, request.age, request.department)
    return {"status": "Employee inserted (if not duplicate)"}
