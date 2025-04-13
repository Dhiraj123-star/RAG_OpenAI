import os
import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the embeddings model
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Load documents from a .txt file
with open("RAG_document.txt", "r", encoding="utf-8") as file:
    documents = [file.read().strip()]

# Convert documents to embeddings
document_embeddings = embeddings.embed_documents(documents)
document_embeddings = np.array(document_embeddings).astype('float32')

# Initialize FAISS index and add document embeddings
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)

# Define a retriever class
class SimpleRetriever:
    def __init__(self, index, documents):
        self.index = index
        self.documents = documents

    def retrieve(self, query, top_k=1):
        query_embedding = np.array([embeddings.embed_query(query)]).astype('float32')
        _, indices = self.index.search(query_embedding, top_k)
        return [self.documents[i] for i in indices[0]]

retriever = SimpleRetriever(index, documents)

# Combine the retriever with the generator
class SimpleRAG:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def generate(self, query):
        retrieved_docs = self.retriever.retrieve(query)
        augmented_query = f"Context: {' '.join(retrieved_docs)} Query: {query}"
        response = self.llm.invoke(augmented_query)
        return response

llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
rag = SimpleRAG(llm, retriever)

# Example usage
query = input("Enter your Query: ")
response = rag.generate(query)
print(response)
