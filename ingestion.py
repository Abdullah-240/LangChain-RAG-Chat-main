import asyncio
import logging
import os
import ssl

import certifi
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_tavily import TavilyCrawl
from langchain_text_splitters import RecursiveCharacterTextSplitter

from logger import *

load_dotenv()
FAISS_INDEX_PATH = "faiss_index"

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

_embeddings = None
_vectorstore = None
_tavily_crawl = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "dummy"
        _embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key,
        )
    return _embeddings


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        if os.path.exists(FAISS_INDEX_PATH):
            try:
                _vectorstore = FAISS.load_local(
                    FAISS_INDEX_PATH,
                    get_embeddings(),
                    allow_dangerous_deserialization=True,
                )
            except Exception as e:
                log_warning(f"Could not load local FAISS index: {e}")
                _vectorstore = None
    return _vectorstore


def get_tavily_crawl():
    global _tavily_crawl
    if _tavily_crawl is None:
        _tavily_crawl = TavilyCrawl()
    return _tavily_crawl


class VectorStoreProxy:
    def as_retriever(self, **kwargs):
        vs = get_vectorstore()
        if vs is None:
            raise ValueError("FAISS vector store is not initialized. Run ingestion.py first.")
        return vs.as_retriever(**kwargs)

    def add_documents(self, documents, **kwargs):
        global _vectorstore
        embeddings = get_embeddings()
        if _vectorstore is None:
            _vectorstore = FAISS.from_documents(documents, embeddings)
        else:
            _vectorstore.add_documents(documents)
        _vectorstore.save_local(FAISS_INDEX_PATH)
        return _vectorstore


vectorstore = VectorStoreProxy()


async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE (FAISS + GEMINI)")

    log_info("🗺️  TavilyCrawl: Starting to crawl the documentation site", Colors.PURPLE)
    tavily = get_tavily_crawl()
    res = tavily.invoke(
        {
            "url": "https://python.langchain.com/",
            "max_depth": 4,
            "extract_depth": "advanced",
            "max_breadth": 200,
            "limit": 800,
        }
    )

    all_docs = [
        Document(page_content=r["raw_content"], metadata={"source": r["url"]})
        for r in res.get("results", [])
    ]

    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with 1000 chunk size and 150 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # Ingestion into FAISS vector database
    log_header("FAISS VECTOR STORAGE PHASE")
    embeddings = get_embeddings()
    log_info("🌲 Building FAISS Index with Google Gemini Embeddings...", Colors.CYAN)
    
    fs = FAISS.from_documents(splitted_docs, embeddings)
    fs.save_local(FAISS_INDEX_PATH)
    
    log_header("PIPELINE COMPLETE")
    log_success(f"🎉 FAISS Vector Database saved to '{FAISS_INDEX_PATH}' successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks indexed in FAISS: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())
