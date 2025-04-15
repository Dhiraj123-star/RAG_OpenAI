from dotenv import load_dotenv
import os

load_dotenv()

import openai
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

openai.api_key = os.getenv("OPENAI_API_KEY")


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
    model = ChatOpenAI(model="gpt-4o-mini")

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