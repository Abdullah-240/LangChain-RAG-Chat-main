import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import (
    create_history_aware_retriever,
)
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI

from ingestion import get_vectorstore, vectorstore
from logger import *

load_dotenv()

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model="gpt-4",
            streaming=True,
        )
    return _llm


class LLMProxy:
    @property
    def model_name(self):
        try:
            return get_llm().model_name
        except Exception:
            return "gpt-4"


llm = LLMProxy()

retrieval_qa_chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

rephrase_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just rephrase it if needed and otherwise return it as is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


def run_llm_from_docs(query: str, chat_history: List[Dict[str, Any]] = []):
    active_llm = get_llm()
    combine_docs_chain = create_stuff_documents_chain(active_llm, retrieval_qa_chat_prompt)

    retriever = get_vectorstore().as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 15, "score_threshold": 0.75},
    )
    history_aware_retriever = create_history_aware_retriever(
        llm=active_llm,
        retriever=retriever,
        prompt=rephrase_prompt,
    )
    retrieval_chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=combine_docs_chain,
    )

    result = retrieval_chain.invoke(
        input={"input": query, "chat_history": chat_history}
    )
    return result


def run_general_llm(query: str, chat_history: List[tuple] = []):
    active_llm = get_llm()
    messages = [("system", "You are a helpful AI assistant")]
    messages.extend(chat_history)
    messages.append(("user", query))
    result = active_llm.invoke(messages)
    return result
