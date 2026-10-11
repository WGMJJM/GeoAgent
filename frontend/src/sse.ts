type ServerEvent = { event: string; data: string };

// 网络 chunk、UTF-8 字符和 SSE 事件的边界彼此独立。
export async function* readServerEvents(body: ReadableStream<Uint8Array>, onActivity: () => void): AsyncGenerator<ServerEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let event = "message";
  let data: string[] = [];
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (value?.length) onActivity();
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      while (true) {
        const index = buffer.search(/[\r\n]/);
        if (index < 0) break;
        // 等待下一块，避免把跨 chunk 的 CRLF 当成两个换行。
        if (!done && buffer[index] === "\r" && index === buffer.length - 1) break;
        const line = buffer.slice(0, index);
        const width = buffer[index] === "\r" && buffer[index + 1] === "\n" ? 2 : 1;
        buffer = buffer.slice(index + width);
        if (line === "") {
          if (data.length) yield { event, data: data.join("\n") };
          event = "message";
          data = [];
        } else if (!line.startsWith(":")) {
          const colon = line.indexOf(":");
          const field = colon < 0 ? line : line.slice(0, colon);
          const value = colon < 0 ? "" : line.slice(colon + 1).replace(/^ /, "");
          if (field === "event") event = value || "message";
          else if (field === "data") data.push(value);
        }
      }
      // 未用空行结束的事件不发布；调用方必须收到明确的 response 才算完成。
      if (done) break;
    }
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}
