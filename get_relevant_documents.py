from dotenv import load_dotenv
import os
import random

load_dotenv()

import openai
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

openai.api_key = os.getenv("OPENAI_API_KEY")

# List of available models
available_models = [
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano-2025-04-14",
    "gpt-4o-mini"
]

def get_k_relevant_documents(documents, question, k=3):
    print(f"Storing {len(documents)} into Vector Store.")
    vector_store = InMemoryVectorStore.from_documents(documents, OpenAIEmbeddings())
    print("Getting relevant documents from in memory vector store.")
    relevant_docs = vector_store.similarity_search(question, k=k)
    print(f"Retrieved similar documents: {len(relevant_docs)}")
    return relevant_docs

def get_answer_from_llm(documents, question):
    print(f"Question: {question}")
    relevant_docs = get_k_relevant_documents(documents, question)

    # Randomly choose a model from the available list
    selected_model = random.choice(available_models)
    print(f"Selected model: {selected_model}")
    model = ChatOpenAI(model=selected_model)

    context_from_docs = "\n\n".join([doc.page_content for doc in relevant_docs])

    messages = [
        SystemMessage(
            content=f"Use the following context to answer my question: {context_from_docs}"
        ),
        HumanMessage(content=f"{question}"),
    ]
    parser = StrOutputParser()

    chain = model | parser
    return chain.invoke(messages)
