import { useRef, useState } from "react";
import type { ComponentProps } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./MarkdownContent.css";

function CodeBlock({ children }: ComponentProps<"pre">) {
  const block = useRef<HTMLPreElement>(null);
  const [copiedText, setCopiedText] = useState<string | null>(null);
  const [error, setError] = useState(false);
  const copy = async () => {
    const text = block.current?.textContent ?? "";
    try {
      await navigator.clipboard.writeText(text);
      setCopiedText(text);
      setError(false);
    } catch {
      setError(true);
    }
  };
  return <div className="markdown-code">
    <div className="markdown-code-toolbar"><span role="status">{error ? "复制失败，请手动选择代码" : "代码"}</span><button type="button" onClick={() => void copy()}>{copiedText !== null && copiedText === block.current?.textContent ? "已复制" : "复制代码"}</button></div>
    <pre ref={block}>{children}</pre>
  </div>;
}

const components: ComponentProps<typeof Markdown>["components"] = {
  pre: CodeBlock,
  table: ({ children }) => <div className="markdown-table" role="region" aria-label="表格，可横向滚动" tabIndex={0}><table>{children}</table></div>,
  a: ({ href, children }) => href ? <a href={href} target="_blank" rel="noopener noreferrer">{children}</a> : <span>{children}</span>,
  // 回复中的任意图片地址不自动请求；真正的数据预览走鉴权接口。
  img: ({ src, alt }) => typeof src === "string" && src ? <a href={src} target="_blank" rel="noopener noreferrer">{alt || "查看图片"}</a> : <span>{alt}</span>,
};

export function MarkdownContent({ content }: { content: string }) {
  return <div className="markdown-content"><Markdown remarkPlugins={[remarkGfm]} components={components} skipHtml>{content}</Markdown></div>;
}
