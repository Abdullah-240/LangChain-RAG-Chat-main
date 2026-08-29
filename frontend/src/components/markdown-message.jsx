import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Highlight, themes } from "prism-react-renderer";

function CodeBlock({ inline, className, children }) {
  const raw = String(children ?? "");
  const code = raw.replace(/^\n+|\n+$/g, "");
  const langRaw = (className || "").replace("language-", "").trim();
  const lang = langRaw || "text";

  const isSingleLine = !code.includes("\n");
  const isPlainLang = ["", "text", "plain", "plaintext"].includes(lang.toLowerCase());
  const isTrivial = isSingleLine && code.trim().length <= 40 && isPlainLang;

  if (inline || isTrivial) {
    return (
      <code className="px-2 py-0.5 rounded bg-cyan-950/80 text-cyan-300 font-mono text-[0.85em] border border-cyan-500/30 shadow-[0_0_8px_rgba(0,243,255,0.15)]">
        {code}
      </code>
    );
  }

  return (
    <div className="my-3 overflow-hidden rounded-xl border border-cyan-500/30 bg-[#070c18] text-slate-100 shadow-[0_0_15px_rgba(0,243,255,0.1)]">
      {!isPlainLang && (
        <div className="flex flex-row items-center justify-between px-3.5 py-2 border-b border-cyan-500/20 bg-cyan-950/40">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-cyan-400 font-['Space_Grotesk']">
            ⚡ {lang}
          </span>
          <button
            className="text-xs px-2.5 py-1 rounded bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 transition-all duration-200 hover:shadow-[0_0_10px_rgba(0,243,255,0.3)] cursor-pointer"
            onClick={(e) => {
              const btn = e.currentTarget;
              const prev = btn.textContent;
              navigator.clipboard
                .writeText(code)
                .then(() => {
                  btn.textContent = "⚡ Copied!";
                  setTimeout(() => (btn.textContent = prev), 1500);
                })
                .catch((err) => console.error("Clipboard copy failed", err));
            }}
            title="Copy code"
          >
            📋 Copy
          </button>
        </div>
      )}
      <div className="p-0">
        <Highlight theme={themes.nightOwl} code={code} language={lang}>
          {({ className, style, tokens, getLineProps, getTokenProps }) => (
            <pre
              className={`${className} m-0 w-full overflow-x-auto p-4 text-xs font-mono leading-relaxed`}
              style={{ ...style, background: "transparent" }}
            >
              {tokens.map((line, i) => (
                <div key={i} {...getLineProps({ line })}>
                  {line.map((token, key) => (
                    <span key={key} {...getTokenProps({ token })} />
                  ))}
                </div>
              ))}
            </pre>
          )}
        </Highlight>
      </div>
    </div>
  );
}

export default function MarkdownMessage({ content }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        code: CodeBlock,
        pre: ({ children }) => <>{children}</>,
        a: ({ href, children }) => (
          <a
            href={href}
            target="_blank"
            rel="noreferrer"
            className="text-cyan-400 hover:text-cyan-300 underline underline-offset-4 decoration-cyan-400/50 hover:decoration-cyan-300 font-medium transition-all duration-200"
          >
            {children}
          </a>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  );
}
