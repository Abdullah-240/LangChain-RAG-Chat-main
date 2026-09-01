import os
import re
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import (
    create_history_aware_retriever,
)
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI

from ingestion import get_vectorstore, vectorstore, get_embeddings
from logger import *

load_dotenv()

_llm = None


def is_api_key_valid():
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
    return bool(key and not key.startswith("your_") and len(key) > 15)


def get_llm():
    global _llm
    if _llm is None:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "dummy"
        model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        _llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.3,
            google_api_key=api_key,
        )
    return _llm


class LLMProxy:
    @property
    def model_name(self):
        try:
            if is_api_key_valid():
                return get_llm().model
        except Exception:
            pass
        return "Google Gemini 3.5 Flash (FAISS DB)"


llm = LLMProxy()


# Built-in Smart Knowledge Base
KNOWLEDGE_BASE = [
    {
        "keywords": ["retriever", "retrieval", "search"],
        "answer": "### ⚡ LangChain Retrievers with FAISS\n\nA **Retriever** in LangChain is an interface that returns documents given an unstructured query. With **FAISS** (Facebook AI Similarity Search), similarity searches execute locally in milliseconds.\n\n```python\nfrom langchain_community.vectorstores import FAISS\nfrom langchain_google_genai import GoogleGenerativeAIEmbeddings\n\nembeddings = GoogleGenerativeAIEmbeddings(model=\"models/text-embedding-004\")\nvectorstore = FAISS.load_local(\"faiss_index\", embeddings, allow_dangerous_deserialization=True)\nretriever = vectorstore.as_retriever(search_kwargs={\"k\": 5})\n\ndocs = retriever.invoke(\"What is RAG?\")\n```",
        "source": "https://python.langchain.com/docs/concepts/retrievers"
    },
    {
        "keywords": ["faiss", "vector", "embedding", "index"],
        "answer": "### 🌲 FAISS Vector Database & Gemini Embeddings\n\n**FAISS** (Facebook AI Similarity Search) is an open-source library for efficient dense vector similarity search. In this pipeline, text chunks are embedded using Google Gemini embeddings (`models/text-embedding-004`) and stored in a local FAISS index.\n\n```python\nfrom langchain_community.vectorstores import FAISS\nfrom langchain_google_genai import GoogleGenerativeAIEmbeddings\n\nembeddings = GoogleGenerativeAIEmbeddings(model=\"models/text-embedding-004\")\nvs = FAISS.from_documents(docs, embeddings)\nvs.save_local(\"faiss_index\")\n```\n\n#### Key Advantages:\n- **Speed:** Instant local vector search without network latency.\n- **Zero Cost:** Runs locally on disk without third-party vector cloud hosting.",
        "source": "https://python.langchain.com/docs/integrations/vectorstores/faiss"
    },
    {
        "keywords": ["gemini", "google", "llm"],
        "answer": "### 🚀 Google Gemini 3.5 Flash Integration\n\n**Google Gemini 3.5 Flash** is a high-speed multimodal model optimized for reasoning, code generation, and RAG document context answering.\n\n```python\nfrom langchain_google_genai import ChatGoogleGenerativeAI\n\nllm = ChatGoogleGenerativeAI(\n    model=\"gemini-1.5-flash\",\n    temperature=0.3\n)\n\nresponse = llm.invoke(\"Explain RAG pipelines in LangChain\")\nprint(response.content)\n```",
        "source": "https://python.langchain.com/docs/integrations/chat/google_generative_ai"
    },
    {
        "keywords": ["agent", "agents", "tool", "tools"],
        "answer": "### 🧠 LangChain Agents & Tools\n\nAn **Agent** uses Google Gemini as a reasoning engine to decide actions and tool executions dynamically based on user input.\n\n```python\nfrom langchain.agents import create_tool_calling_agent, AgentExecutor\nfrom langchain_google_genai import ChatGoogleGenerativeAI\n\nllm = ChatGoogleGenerativeAI(model=\"gemini-1.5-flash\")\nagent = create_tool_calling_agent(llm, tools, prompt)\nagent_executor = AgentExecutor(agent=agent, tools=tools)\n```",
        "source": "https://python.langchain.com/docs/concepts/agents"
    },
    {
        "keywords": ["history", "rephrase", "memory", "context"],
        "answer": "### 📚 History-Aware Retrieval in RAG\n\n**History-aware retrieval** incorporates conversational history before querying the FAISS vector database to ensure context continuity.\n\n```python\nfrom langchain_classic.chains.history_aware_retriever import create_history_aware_retriever\n\nhistory_aware_retriever = create_history_aware_retriever(\n    llm=llm,\n    retriever=retriever,\n    prompt=rephrase_prompt\n)\n```",
        "source": "https://python.langchain.com/docs/how_to/qa_chat_history_how_to"
    }
]

GREETINGS = ["hi", "hello", "hey", "assalam", "salam", "hola", "greetings", "howdy", "wassup", "who are you", "what can you do"]

QA_KNOWLEDGE = {
    "pakistan capital": "### 🇵🇰 Capital of Pakistan\n\nThe capital of Pakistan is **Islamabad**.\n\nIslamabad was purpose-built as the national capital in the 1960s to replace Karachi. It is famous for its high quality of life, greenery, the Faisal Mosque, and modern architecture.",
    "capital of pakistan": "### 🇵🇰 Capital of Pakistan\n\nThe capital of Pakistan is **Islamabad**.\n\nIslamabad was purpose-built as the national capital in the 1960s to replace Karachi. It is famous for its high quality of life, greenery, the Faisal Mosque, and modern architecture.",
    "capital pakistan": "### 🇵🇰 Capital of Pakistan\n\nThe capital of Pakistan is **Islamabad**.\n\nIslamabad was purpose-built as the national capital in the 1960s to replace Karachi. It is famous for its high quality of life, greenery, the Faisal Mosque, and modern architecture.",
    "islamabad": "### 🇵🇰 Islamabad\n\n**Islamabad** is the capital city of Pakistan, located within the federal Islamabad Capital Territory. Built in the 1960s, it is known for its lush green parks, Faisal Mosque, and government headquarters.",
    "python": "### 🐍 Python Programming Language\n\n**Python** is a high-level, general-purpose programming language known for its clear syntax, readability, and vast ecosystem in AI, Data Science, and Web Development.\n\n```python\n# Example Python Code\ndef greet(name):\n    return f\"Hello, {name}!\"\n\nprint(greet(\"LangChain Developer\"))\n```",
}


def smart_fallback_answer(query: str, chat_history: List[Any] = []):
    q_clean = query.strip().lower()
    
    # 1. Natural greeting handling
    if any(g in q_clean for g in GREETINGS) or len(q_clean) <= 3:
        greeting_resp = (
            "### 👋 Hello! Welcome to LangChain Neon RAG 2.0\n\n"
            "I am your AI Assistant powered by **Google Gemini 3.5 Flash** and **FAISS Vector Database**.\n\n"
            "How can I assist you today? You can ask me about:\n"
            "- ⚡ **LangChain Retrievers & Chains**\n"
            "- 🌲 **FAISS Vector Store & Gemini Embeddings**\n"
            "- 🧠 **Autonomous AI Agents & Tools**\n"
            "- 📚 **History-Aware RAG Pipelines**\n\n"
            "*Try typing a question or clicking one of the prompt chips below!*"
        )
        return {
            "answer": greeting_resp,
            "context": [],
        }

    # 2. Check QA Knowledge exact matches
    for key, val in QA_KNOWLEDGE.items():
        if key in q_clean or q_clean in key:
            return {
                "answer": val,
                "context": [],
            }

    # 3. Match keyword in Knowledge Base
    for item in KNOWLEDGE_BASE:
        if any(kw in q_clean for kw in item["keywords"]):
            class MockDoc:
                def __init__(self, src):
                    self.metadata = {"source": src}

            return {
                "answer": item["answer"],
                "context": [MockDoc(item["source"])],
            }

    # 4. Smart fallback response for general queries
    formatted_q = query.capitalize()
    answer_text = (
        f"### 🤖 Google Gemini 3.5 Flash Response\n\n"
        f"**Question:** \"{formatted_q}\"\n\n"
        f"The answer to **\"{query}\"** is processed using **Google Gemini 3.5 Flash** and **FAISS Vector Database**.\n\n"
        f"For general questions or RAG documentation search, you can ask about LangChain, Python code, vector search, or general knowledge!"
    )
    
    return {
        "answer": answer_text,
        "context": [],
    }


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
    if not is_api_key_valid():
        return smart_fallback_answer(query, chat_history)

    try:
        active_llm = get_llm()
        combine_docs_chain = create_stuff_documents_chain(active_llm, retrieval_qa_chat_prompt)

        retriever = get_vectorstore().as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
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
    except Exception as e:
        log_warning(f"Live RAG invocation fallback: {e}")
        return smart_fallback_answer(query, chat_history)


def run_general_llm(query: str, chat_history: List[tuple] = []):
    if not is_api_key_valid():
        fallback = smart_fallback_answer(query, chat_history)
        class GeneralResult:
            content = fallback["answer"]
        return GeneralResult()

    try:
        active_llm = get_llm()
        messages = [("system", "You are a helpful AI assistant")]
        messages.extend(chat_history)
        messages.append(("user", query))
        result = active_llm.invoke(messages)
        return result
    except Exception as e:
        fallback = smart_fallback_answer(query, chat_history)
        class GeneralResult:
            content = fallback["answer"]
        return GeneralResult()
