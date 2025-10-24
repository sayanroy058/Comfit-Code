"use client";

import { useEffect, useRef } from "react";
import renderMathInElement from "katex/contrib/auto-render";
import "katex/dist/katex.min.css";
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw"; // 👈 allows raw HTML rendering

interface Props {
  html: string;
  className?: string;
}

// Converts only [ ... ] blocks that begin with a LaTeX backslash (e.g., [ \frac{a}{b} ])
function convertLatexBrackets(input: string) {
  return input.replace(/\[\s*\\(.+?)\s*\]/g, (_, expr) => `$$\\${expr.trim()}$$`);
}

export default function FormattedContent({ html, className = "" }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      const convertedHtml = convertLatexBrackets(html);
      containerRef.current.innerHTML = convertedHtml;

      // Render math content
      renderMathInElement(containerRef.current, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false },
        ],
        throwOnError: false,
        fleqn: false,
        output: "html",
      });
    }
  }, [html]);

  // Check if content has markdown or raw HTML patterns
  const hasMarkdown = /[*_`#\[\]()!]/.test(html) || /<img|<a|<br/.test(html);

  if (hasMarkdown) {
    return (
      <div className={className}>
        <ReactMarkdown
          rehypePlugins={[rehypeRaw]} // 👈 enable raw HTML passthrough
          components={{
            // 👇 Custom rendering overrides
            img: ({ src, alt }) => (
              <img
                src={src || ""}
                alt={alt || "image"}
                className="rounded-lg my-2 max-w-full"
              />
            ),
            a: ({ href, children }) => (
              <a
                href={href}
                className="text-blue-600 hover:underline"
                target="_blank"
                rel="noopener noreferrer"
              >
                {children}
              </a>
            ),
            p: ({ children }) => <p className="mb-2">{children}</p>,
            h1: ({ children }) => (
              <h1 className="text-2xl font-bold mb-3">{children}</h1>
            ),
            h2: ({ children }) => (
              <h2 className="text-xl font-bold mb-2">{children}</h2>
            ),
            h3: ({ children }) => (
              <h3 className="text-lg font-bold mb-2">{children}</h3>
            ),
            h4: ({ children }) => (
              <h4 className="text-base font-bold mb-1">{children}</h4>
            ),
            h5: ({ children }) => (
              <h5 className="text-sm font-bold mb-1">{children}</h5>
            ),
            h6: ({ children }) => (
              <h6 className="text-xs font-bold mb-1">{children}</h6>
            ),
            strong: ({ children }) => (
              <strong className="font-bold">{children}</strong>
            ),
            em: ({ children }) => <em className="italic">{children}</em>,
            code: ({ children }) => (
              <code className="bg-gray-200 px-1 py-0.5 rounded text-sm font-mono">
                {children}
              </code>
            ),
            pre: ({ children }) => (
              <pre className="bg-gray-100 p-3 rounded-lg overflow-x-auto mb-3">
                {children}
              </pre>
            ),
            blockquote: ({ children }) => (
              <blockquote className="border-l-4 border-gray-300 pl-4 italic mb-3">
                {children}
              </blockquote>
            ),
            ul: ({ children }) => (
              <ul className="list-disc list-inside mb-3 space-y-1">
                {children}
              </ul>
            ),
            ol: ({ children }) => (
              <ol className="list-decimal list-inside mb-3 space-y-1">
                {children}
              </ol>
            ),
            li: ({ children }) => <li className="mb-1">{children}</li>,
          }}
        >
          {html}
        </ReactMarkdown>
      </div>
    );
  }

  // Fallback: render raw HTML directly (non-markdown content)
  return <div ref={containerRef} className={className} />;
}
