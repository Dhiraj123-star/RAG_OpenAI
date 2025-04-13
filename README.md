# Simple RAG with FAISS and OpenAI

This project demonstrates a basic Retrieval-Augmented Generation (RAG) setup using:

- **OpenAI** for generating embeddings and answering queries
- **FAISS** for fast similarity-based retrieval
- A `.txt` file as the knowledge source

## Functionality

- Loads context from a `RAG_document.txt` file.
- Converts the document into vector embeddings using OpenAI's Embedding model.
- Stores those embeddings in a FAISS index for fast retrieval.
- Retrieves the most relevant context based on a user's query.
- Feeds the context along with the query to OpenAI's language model to generate an answer.

## Example Flow

1. User provides a query 

2. The system:
   - Retrieves relevant context from the `.txt` file.
   - Augments the query with that context.
   - Sends the augmented prompt to OpenAI.
   - Returns a context-aware response.

## Input File

Make sure `RAG_document.txt` exists in the root directory and contains the content you want to query against.

