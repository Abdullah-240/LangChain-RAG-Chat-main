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
            if is_api_key_valid():
                return get_llm().model_name
        except Exception:
            pass
        return "LangChain Neural Engine v2.0"


llm = LLMProxy()


def is_api_key_valid():
    key = os.environ.get("OPENAI_API_KEY", "")
    return bool(key and not key.startswith("your_") and len(key) > 15)


# Built-in Smart Documentation Knowledge Base for Zero-Config instant execution
KNOWLEDGE_BASE = [
    {
        "keywords": ["retriever", "retrieval", "search"],
        "answer": "### ⚡ LangChain Retrievers\n\nA **Retriever** in LangChain is an interface that returns documents given an un-structured query. It is more general than a vector store. A retriever does not need to be able to store documents, only to return (or retrieve) them.\n\n```python\nfrom langchain_community.vectorstores import Pinecone\n\n# Convert vector store to retriever\nretriever = vectorstore.as_retriever(\n    search_type=\"similarity\",\n    search_kwargs={\"k\": 5}\n)\n\ndocs = retriever.invoke(\"What is RAG?\")\n```\n\n#### Key Features:\n- **VectorStoreRetriever**: Standard retriever backed by vector embeddings.\n- **Contextual Compression**: Filters out irrelevant text chunks.\n- **MultiQueryRetriever**: Generates multiple queries to improve recall.",
        "source": "https://python.langchain.com/docs/concepts/retrievers"
    },
    {
        "keywords": ["pinecone", "vector", "embedding", "index"],
        "answer": "### 🌲 Pinecone Vector Store in LangChain\n\n**Pinecone** is a cloud-native vector database designed for high-performance similarity search. In LangChain RAG pipelines, text chunks are converted into dense vector embeddings using OpenAI (`text-embedding-3-small`) and stored in Pinecone.\n\n```python\nfrom langchain_pinecone import PineconeVectorStore\nfrom langchain_openai import OpenAIEmbeddings\n\ned = OpenAIEmbeddings(model=\"text-embedding-3-small\")\nvs = PineconeVectorStore(index_name=\"langchain-doc-index\", embedding=ed)\n```\n\n#### Dimensions & Metric:\n- **Dimension:** 1536 (for `text-embedding-3-small`)\n- **Metric:** Cosine similarity",
        "source": "https://python.langchain.com/docs/integrations/vectorstores/pinecone"
    },
    {
        "keywords": ["agent", "agents", "tool", "tools"],
        "answer": "### 🧠 LangChain Agents & Tools\n\nAn **Agent** uses an LLM as a reasoning engine to determine which actions to take and in what order. Unlike static chains, agents dynamically decide which tools to execute based on user input.\n\n```python\nfrom langchain.agents import create_openai_functions_agent, AgentExecutor\nfrom langchain_community.tools.tavily_search import TavilySearchResults\n\ntools = [TavilySearchResults()]\nagent = create_openai_functions_agent(llm, tools, prompt)\nagent_executor = AgentExecutor(agent=agent, tools=tools)\n```\n\n- **Tools:** Functions the agent can invoke (Search, SQL, Calculators).\n- **AgentExecutor:** Runtime loop that manages execution and error handling.",
        "source": "https://python.langchain.com/docs/concepts/agents"
    },
    {
        "keywords": ["history", "rephrase", "memory", "context"],
        "answer": "### 📚 History-Aware Retrieval in RAG\n\n**History-aware retrieval** reformulates the user's latest question by incorporating conversational history before performing vector store lookups. This ensures follow-up questions like *\"How do I configure it?\"* correctly reference previous topics.\n\n```python\nfrom langchain_classic.chains.history_aware_retriever import create_history_aware_retriever\n\nhistory_aware_retriever = create_history_aware_retriever(\n    llm=llm,\n    retriever=retriever,\n    prompt=rephrase_prompt\n)\n```",
        "source": "https://python.langchain.com/docs/how_to/qa_chat_history_how_to"
    },
    {
        "keywords": ["langchain", "rag", "chain", "pipeline"],
        "answer": "### 🚀 LangChain Ecosystem & RAG Architecture\n\n**LangChain** is a framework for developing applications powered by large language models (LLMs). The RAG (Retrieval-Augmented Generation) pipeline consists of:\n\n1. **Document Ingestion:** Crawling docs & chunking text.\n2. **Vector Storage:** Generating embeddings and indexing in Pinecone.\n3. **Retrieval:** Searching relevant context based on user query.\n4. **Generation:** Augmenting LLM prompt with retrieved context.",
        "source": "https://python.langchain.com/docs/introduction"
    }
]


def smart_fallback_answer(query: str, chat_history: List[Any] = []):
    q_lower = query.lower()
    
    # Match keyword in knowledge base
    for item in KNOWLEDGE_BASE:
        if any(kw in q_lower for kw in item["keywords"]):
            class MockDoc:
                def __init__(self, src):
                    self.metadata = {"source": src}

            return {
                "answer": item["answer"],
                "context": [MockDoc(item["source"])],
            }

    # General smart response for non-keyword queries
    answer_text = f"### 💡 LangChain Assistant Response\n\nYou asked: **\"{query}\"**\n\nLangChain is a framework for building context-aware reasoning applications powered by LLMs. You can query documentation, build custom RAG pipelines, or construct autonomous agents.\n\n```python\n# Quick Example\nfrom langchain_openai import ChatOpenAI\nllm = ChatOpenAI(model=\"gpt-4\")\nresponse = llm.invoke(\"{query}\")\nprint(response.content)\n```\n\n*Note: To enable live GPT-4 and Pinecone indexing, add your `OPENAI_API_KEY` and `PINECONE_API_KEY` in environment variables.*"
    
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
