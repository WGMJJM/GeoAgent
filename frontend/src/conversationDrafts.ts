export type ConversationDraft = {
  message: string;
  selectedDatasetIds: string[];
  uploadedFiles: { id: string; name: string }[];
};

export const emptyDraft = (): ConversationDraft => ({ message: "", selectedDatasetIds: [], uploadedFiles: [] });
const storageKey = (userId: string) => `geoagent.drafts.${userId}`;

export function readConversationDrafts(userId: string): Record<string, ConversationDraft> {
  const raw = sessionStorage.getItem(storageKey(userId));
  if (!raw) return {};
  const drafts = JSON.parse(raw) as Record<string, ConversationDraft>;
  // 浏览器存储不是服务端数据；在恢复边界校验结构，避免坏草稿阻断对话。
  if (!drafts || typeof drafts !== "object" || Array.isArray(drafts) || Object.values(drafts).some((draft) =>
    !draft || typeof draft.message !== "string" || !Array.isArray(draft.selectedDatasetIds) ||
    !draft.selectedDatasetIds.every((id) => typeof id === "string") || !Array.isArray(draft.uploadedFiles) ||
    !draft.uploadedFiles.every((file) => file && typeof file.id === "string" && typeof file.name === "string")
  )) throw new Error("Invalid conversation draft");
  return drafts;
}

export function saveConversationDrafts(userId: string, drafts: Record<string, ConversationDraft>) {
  const pending = Object.fromEntries(Object.entries(drafts).filter(([, draft]) =>
    draft.message || draft.selectedDatasetIds.length || draft.uploadedFiles.length
  ));
  if (Object.keys(pending).length) sessionStorage.setItem(storageKey(userId), JSON.stringify(pending));
  else sessionStorage.removeItem(storageKey(userId));
}
