import { useState } from "react";

export function CopyReplyButton({ content }: { content: string }) {
  const [copied, setCopied] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(content);
      setFailed(false);
    } catch {
      setFailed(true);
    }
  };
  return <div className="reply-copy">
    <button type="button" className="chat-result-link" onClick={() => void copy()}>{copied === content ? "已复制回复" : "复制回复"}</button>
    {failed && <span role="status">复制失败，请手动选择回复文字。</span>}
  </div>;
}
