import { useEffect, useState } from "react";
import { Icon } from "./Icon";

export function CopyReplyButton({ content }: { content: string }) {
  const [copied, setCopied] = useState<{ content: string } | null>(null);
  const [failed, setFailed] = useState(false);
  const isCopied = copied?.content === content;
  useEffect(() => {
    if (copied === null) return;
    const timer = window.setTimeout(() => setCopied(null), 1600);
    return () => window.clearTimeout(timer);
  }, [copied]);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied({ content });
      setFailed(false);
    } catch {
      setCopied(null);
      setFailed(true);
    }
  };
  return <div className="reply-copy">
    <button type="button" className="reply-copy-button" aria-label={isCopied ? "已复制回复" : "复制回复"} title={isCopied ? "已复制回复" : "复制回复"} data-copied={isCopied} onClick={() => void copy()}><Icon name={isCopied ? "check" : "copy"} size={16} /></button>
    {failed && <span role="status">复制失败，请手动选择回复文字。</span>}
  </div>;
}
