import type { ReactNode } from "react";

// 所有图标使用同一视框与线宽；不改变承载它们的按钮和布局。
const shapes = {
  compass: <><circle cx="12" cy="12" r="9" /><path d="m16 8-2.5 5.5L8 16l2.5-5.5L16 8Z" /></>,
  layers: <><path d="m12 3 9 5-9 5-9-5 9-5Z" /><path d="m3 12 9 5 9-5M3 16l9 5 9-5" /></>,
  agent: <><rect x="4" y="7" width="16" height="13" rx="3" /><path d="M12 3v4M2 12v4m20-4v4M9 16h6M9 11v1m6-1v1" /></>,
  activity: <path d="M3 12h4l3-8 4 16 3-8h4" />,
  plus: <path d="M12 5v14M5 12h14" />,
  close: <path d="m6 6 12 12M6 18 18 6" />,
  trash: <path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7m4-7v7" />,
  refresh: <path d="M20 10a8 8 0 1 0-2 8M20 4v6h-6" />,
  check: <path d="m5 12 4 4L19 6" />,
  clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
  attachment: <path d="m8 12 7-7a4 4 0 0 1 6 6L10 22a6 6 0 0 1-8-8L13 3m-7 13 9-9" />,
  send: <><path d="m21 3-7 18-4-7-7-4 18-7Z" /><path d="m10 14 11-11" /></>,
  branch: <path d="M6 3v10a4 4 0 0 0 4 4h10m-4-4 4 4-4 4" />,
  external: <><path d="M14 3h7v7m0-7L10 14" /><path d="M10 3H3v18h18v-7" /></>,
} satisfies Record<string, ReactNode>;

export type IconName = keyof typeof shapes;

export function Icon({ name, size = 16 }: { name: IconName; size?: number }) {
  return <svg className="ui-icon" data-icon={name} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">{shapes[name]}</svg>;
}
