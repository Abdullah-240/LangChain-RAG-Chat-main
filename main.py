import os
from datetime import date

from flask import Flask, jsonify, request
from flask_cors import CORS

from core import llm, run_general_llm, run_llm_from_docs

# create the app
app = Flask(__name__)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-secret")


CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "online",
        "message": "LangChain Chat Backend API is running successfully!",
        "frontend": "http://localhost:5173",
        "endpoints": {
            "health": "GET /health",
            "answer": "POST /answer"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/answer", methods=["POST"])
def answer():
    data = request.get_json(silent=True) or {}
    query = data.get("query")
    chat_history = data.get("chat_history", [])

    if not query:
        return jsonify({"error": "Missing query"}), 400

    try:
        # 1) Try docs pipeline
        docs_result = run_llm_from_docs(query, chat_history)
        sources = [doc.metadata.get("source", "") for doc in docs_result.get("context", []) if hasattr(doc, "metadata")]

        if sources:  # ✅ Docs mode succeeded
            ans = docs_result.get("answer", "")
            provenance = "docs"
        else:  # ❌ Fallback to general LLM
            general_result = run_general_llm(query, chat_history)
            ans = general_result.content
            sources = []
            provenance = "model_only"
        model_name = getattr(llm, "model_name", "gpt-4")
    except Exception as e:
        err_msg = str(e)
        if "API key" in err_msg or "Unauthorized" in err_msg or "401" in err_msg or "api_key" in err_msg or "APIKey" in err_msg or "INVALID_ARGUMENT" in err_msg:
            ans = "⚠️ **Google Gemini API Key Configuration Needed**\n\nPlease set your valid credentials in the `.env` file in the project root:\n\n```env\nGOOGLE_API_KEY=your_gemini_api_key_here\nTAVILY_API_KEY=your_tavily_api_key_here\n```"
            provenance = "model_only"
            sources = []
            model_name = "System"
        else:
            return jsonify({"error": f"Backend Error: {err_msg}"}), 500

    # 2) Update history (same shape for both)
    updated_history = chat_history + [
        {"role": "human", "content": query},
        {"role": "ai", "content": ans},
    ]

    # 3) Unified response
    return jsonify(
        {
            "answer": ans,
            "chat_history": updated_history,
            "sources": sources,
            "provenance": provenance,
            "model_name": model_name,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", debug=True, port=port)
