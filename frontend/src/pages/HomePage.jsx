import { useState, useEffect, useRef } from "react";
import MarkdownMessage from "@/components/markdown-message";

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);

  const canSend = query.trim().length >= 2 && !loading;
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const quickPrompts = [
    { label: "⚡ What is a Retriever in LangChain?", text: "What is a retriever in LangChain?" },
    { label: "🌲 How does Pinecone vector store work?", text: "How does Pinecone vector store work in LangChain RAG?" },
    { label: "🧠 Explain LangChain Agents", text: "What are LangChain Agents and tools?" },
    { label: "📚 What is history-aware retrieval?", text: "Explain history-aware retrieval in documentation chatbot." },
  ];

  async function sendQuery(textToQuery) {
    if (!textToQuery || loading) return;
    const text = textToQuery.trim();
    setQuery("");
    setLoading(true);

    try {
      const apiBase = import.meta.env.VITE_API_BASE_URL || "/api";
      const res = await fetch(`${apiBase}/answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: text,
          chat_history: messages,
        }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || res.statusText);

      const uiHistory = Array.isArray(data.chat_history)
        ? [...data.chat_history]
        : [];
      const cites = Array.isArray(data.sources)
        ? [...new Set(data.sources)]
        : [];
      const modelName = data.model_name ?? "GPT-4";

      for (let i = uiHistory.length - 1; i >= 0; i--) {
        if (uiHistory[i].role === "ai") {
          uiHistory[i] = {
            ...uiHistory[i],
            citations: cites,
            provenance: data.provenance,
            model_name: modelName,
          };
          break;
        }
      }
      setMessages(uiHistory);
    } catch (err) {
      setMessages((m) => [
        ...m,
        {
          role: "human",
          content: text,
        },
        {
          role: "ai",
          content: "⚠️ **Connection Error**: Unable to reach backend server on `http://localhost:5000`. Please make sure the Flask backend is active.",
          provenance: "model_only",
          model_name: "System",
        },
      ]);
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(e) {
    e.preventDefault();
    if (canSend) {
      sendQuery(query);
    }
  }

  function handleClearChat() {
    setMessages([]);
  }

  return (
    <div className="relative min-h-screen bg-[#030611] text-slate-100 flex flex-col font-['Plus_Jakarta_Sans',sans-serif] overflow-x-hidden">
      {/* Background Cyber Orbs & Grid */}
      <div className="fixed inset-0 bg-cyber-grid opacity-30 pointer-events-none z-0"></div>
      <div className="fixed -top-40 -left-40 w-96 h-96 bg-cyan-500/15 rounded-full blur-[120px] pointer-events-none z-0 animate-pulse-glow"></div>
      <div className="fixed top-1/2 -right-40 w-96 h-96 bg-purple-600/15 rounded-full blur-[140px] pointer-events-none z-0 animate-pulse-glow"></div>

      {/* Futuristic Header */}
      <header className="sticky top-0 z-40 w-full glass-panel border-b border-cyan-500/20 shadow-[0_4px_25px_rgba(0,243,255,0.08)]">
        <div className="max-w-5xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-cyan-950/60 border border-cyan-500/40 shadow-[0_0_15px_rgba(0,243,255,0.3)]">
              <img
                src="/langchain_icon.png"
                alt="LangChain logo"
                className="w-6 h-6 object-contain filter drop-shadow-[0_0_8px_rgba(0,243,255,0.8)]"
              />
              <span className="absolute -bottom-0.5 -right-0.5 w-3 h-3 bg-emerald-400 rounded-full border-2 border-[#030611] shadow-[0_0_8px_#00ff9d]"></span>
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-wider font-['Space_Grotesk'] flex items-center gap-2">
                <span className="bg-gradient-to-r from-cyan-400 via-emerald-300 to-purple-400 bg-clip-text text-transparent neon-text-cyan">
                  LANGCHAIN NEON
                </span>
                <span className="text-xs px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 uppercase font-mono tracking-widest">
                  RAG 2.0
                </span>
              </h1>
              <p className="text-[11px] text-cyan-300/70 font-mono tracking-wide">
                DOCUMENTATION GROUNDED AI ASSISTANT
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-950/40 border border-cyan-500/30 text-xs font-mono text-emerald-400 shadow-[0_0_10px_rgba(0,255,157,0.15)]">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              SYSTEM ONLINE
            </div>
            {messages.length > 0 && (
              <button
                onClick={handleClearChat}
                className="px-3 py-1.5 rounded-lg bg-red-950/30 hover:bg-red-900/40 text-red-300 border border-red-500/30 text-xs font-medium transition-all duration-200 hover:shadow-[0_0_10px_rgba(239,68,68,0.3)] cursor-pointer"
              >
                🗑️ Clear Chat
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="relative z-10 flex-1 max-w-4xl w-full mx-auto px-4 py-6 flex flex-col justify-between">
        
        {/* Welcome Cyber Hero Card (When empty) */}
        {messages.length === 0 && (
          <div className="my-auto py-8 text-center space-y-6 animate-fade-in">
            <div className="inline-flex items-center justify-center p-4 rounded-2xl bg-cyan-950/40 border border-cyan-500/40 shadow-[0_0_30px_rgba(0,243,255,0.2)]">
              <span className="text-4xl">🤖</span>
            </div>
            <div className="space-y-2 max-w-lg mx-auto">
              <h2 className="text-2xl sm:text-3xl font-extrabold font-['Space_Grotesk'] bg-gradient-to-r from-cyan-300 via-purple-300 to-emerald-300 bg-clip-text text-transparent">
                How can I assist your LangChain workflow today?
              </h2>
              <p className="text-xs sm:text-sm text-slate-400 leading-relaxed font-light">
                Ask questions about LangChain ecosystem documentation. The RAG pipeline automatically retrieves answers with live source links.
              </p>
            </div>

            {/* Quick Prompts Chips */}
            <div className="pt-4 grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-2xl mx-auto">
              {quickPrompts.map((item, idx) => (
                <button
                  key={idx}
                  onClick={() => sendQuery(item.text)}
                  className="p-3.5 rounded-xl glass-panel text-left text-xs font-medium text-slate-200 hover:text-cyan-300 border border-cyan-500/20 hover:border-cyan-500/50 transition-all duration-300 hover:shadow-[0_0_20px_rgba(0,243,255,0.2)] hover:-translate-y-0.5 group cursor-pointer"
                >
                  <div className="flex items-center justify-between">
                    <span className="group-hover:neon-text-cyan">{item.label}</span>
                    <span className="text-cyan-400 opacity-0 group-hover:opacity-100 transition-opacity">➔</span>
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Message Feed */}
        {messages.length > 0 && (
          <div className="space-y-5 pb-6">
            {messages.map((m, i) => (
              <div
                key={i}
                className={`flex flex-col ${
                  m.role === "human" ? "items-end" : "items-start"
                }`}
              >
                <div
                  className={`max-w-[85%] sm:max-w-[78%] rounded-2xl p-4 transition-all duration-300 ${
                    m.role === "human"
                      ? "bg-gradient-to-r from-cyan-950/80 to-cyan-900/80 border border-cyan-400/50 text-cyan-50 shadow-[0_0_20px_rgba(0,243,255,0.2)] rounded-br-none"
                      : "bg-[#080d1e]/90 border border-purple-500/30 text-slate-100 shadow-[0_0_20px_rgba(188,19,254,0.12)] rounded-bl-none"
                  }`}
                >
                  {/* Sender Header Badge */}
                  <div className="flex items-center gap-2 mb-2 border-b border-white/5 pb-1.5 text-[11px] font-mono">
                    {m.role === "human" ? (
                      <span className="text-cyan-300 font-semibold flex items-center gap-1">
                        👤 USER
                      </span>
                    ) : (
                      <span className="text-purple-300 font-semibold flex items-center gap-1">
                        ⚡ AI ASSISTANT
                      </span>
                    )}
                  </div>

                  {/* Message Content */}
                  <div className="text-sm leading-relaxed whitespace-pre-wrap">
                    <MarkdownMessage content={m.content} />
                  </div>

                  {/* Provenance Badge for AI */}
                  {m.role === "ai" && (
                    <div className="mt-3 pt-2 border-t border-white/5 flex flex-wrap items-center justify-between gap-2 text-[10px] font-mono uppercase">
                      {m.provenance === "docs" ? (
                        <span
                          title="Verified answer retrieved from LangChain official documentation."
                          className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-emerald-950/80 text-emerald-300 border border-emerald-500/50 shadow-[0_0_10px_rgba(0,255,157,0.25)] font-semibold"
                        >
                          📖 GROUNDED IN DOCS
                        </span>
                      ) : (
                        <span
                          title={`Generated by ${m.model_name ?? "GPT-4"} general model.`}
                          className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-purple-950/80 text-purple-300 border border-purple-500/50 shadow-[0_0_10px_rgba(188,19,254,0.25)] font-semibold"
                        >
                          💡 GENERAL LLM ANSWER
                        </span>
                      )}

                      <span className="text-slate-400">
                        MODEL: <span className="text-cyan-300">{m.model_name ?? "GPT-4"}</span>
                      </span>
                    </div>
                  )}

                  {/* Sources Citations Accordion */}
                  {m.role === "ai" &&
                    Array.isArray(m.citations) &&
                    m.citations.length > 0 && (
                      <details className="mt-3 group rounded-xl bg-cyan-950/40 border border-cyan-500/30 overflow-hidden">
                        <summary className="px-3 py-2 text-xs font-mono font-semibold text-cyan-300 cursor-pointer select-none flex items-center justify-between hover:bg-cyan-900/30 transition-colors">
                          <span className="flex items-center gap-1.5">
                            🔗 SOURCES & ATTRIBUTIONS ({m.citations.length})
                          </span>
                          <span className="text-[10px] opacity-70 group-open:rotate-180 transition-transform">▼</span>
                        </summary>
                        <ul className="px-4 py-2 space-y-1.5 text-xs font-mono border-t border-cyan-500/20 bg-[#050914]">
                          {m.citations.map((src, j) => (
                            <li key={j} className="truncate">
                              <a
                                href={src}
                                className="text-cyan-400 hover:text-cyan-300 underline underline-offset-2 break-all hover:neon-text-cyan transition-all"
                                target="_blank"
                                rel="noopener noreferrer"
                              >
                                {j + 1}. {src}
                              </a>
                            </li>
                          ))}
                        </ul>
                      </details>
                    )}
                </div>
              </div>
            ))}

            {/* Glowing Loading Hologram Spinner */}
            {loading && (
              <div className="flex items-center gap-3 p-4 rounded-2xl bg-[#080d1e]/80 border border-cyan-500/40 w-fit shadow-[0_0_20px_rgba(0,243,255,0.2)] animate-pulse">
                <div className="w-5 h-5 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin"></div>
                <span className="text-xs font-mono text-cyan-300 font-semibold tracking-wider neon-text-cyan">
                  🧠 QUERYING VECTOR STORE & GENERATING RESPONSE...
                </span>
              </div>
            )}
            <div ref={endRef} />
          </div>
        )}

        {/* Floating Cyber Input Control Bar */}
        <div className="sticky bottom-4 mt-auto pt-2">
          <form
            onSubmit={handleSubmit}
            className="relative flex items-center gap-2 p-2 rounded-2xl glass-panel border border-cyan-500/40 shadow-[0_0_30px_rgba(0,243,255,0.15)] focus-within:border-cyan-400 focus-within:shadow-[0_0_35px_rgba(0,243,255,0.3)] transition-all duration-300"
          >
            <input
              type="text"
              placeholder="Ask anything about LangChain docs..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={loading}
              className="flex-1 bg-transparent px-4 py-2 text-sm text-slate-100 placeholder-slate-500 focus:outline-none font-medium"
            />
            <button
              type="submit"
              disabled={!canSend}
              className={`px-5 py-2.5 rounded-xl font-mono text-xs font-bold uppercase tracking-wider transition-all duration-300 flex items-center gap-2 cursor-pointer ${
                canSend
                  ? "bg-gradient-to-r from-cyan-500 via-teal-400 to-purple-600 text-black shadow-[0_0_20px_rgba(0,243,255,0.5)] hover:shadow-[0_0_30px_rgba(0,243,255,0.8)] hover:scale-[1.02] active:scale-95"
                  : "bg-slate-800/80 text-slate-500 border border-slate-700/50 cursor-not-allowed"
              }`}
            >
              <span>SEND</span>
              <span className="text-sm">⚡</span>
            </button>
          </form>
          <div className="mt-2 text-center text-[10px] font-mono text-slate-500">
            LANGCHAIN ECOSYSTEM RAG • PINECONE VECTOR SEARCH • OPENAI EMBEDDINGS
          </div>
        </div>

      </main>
    </div>
  );
}
