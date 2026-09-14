// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { AssistantRuntimeProvider, useLocalRuntime, type ChatModelAdapter } from "@assistant-ui/react";
import { XProvider } from "@ant-design/x";
import { App as AntApp } from "antd";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Composer } from "../src/components/chat/composer";

const onRun = vi.fn();
const adapter: ChatModelAdapter = {
  async *run({ messages }) {
    onRun(messages.at(-1)?.content);
    yield { content: [{ type: "text", text: "test response" }] };
  },
};
function Harness() {
  const runtime = useLocalRuntime(adapter);
  return <AssistantRuntimeProvider runtime={runtime}>
    <XProvider><AntApp><Composer /></AntApp></XProvider>
  </AssistantRuntimeProvider>;
}

let host: HTMLDivElement;
let root: Root;
let input: HTMLTextAreaElement;
beforeEach(async () => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("ResizeObserver", class { observe() {} unobserve() {} disconnect() {} });
  window.matchMedia = vi.fn().mockImplementation((media: string) => ({
    matches: false, media, onchange: null, addListener() {}, removeListener() {},
    addEventListener() {}, removeEventListener() {}, dispatchEvent: () => true,
  }));
  onRun.mockClear();
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
  await act(async () => { root.render(<Harness />); });
  input = host.querySelector<HTMLTextAreaElement>("textarea.ant-sender-input")!;
  expect(input).not.toBeNull();
  act(() => input.focus());
});
afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
  vi.unstubAllGlobals();
});

function composition(type: "compositionstart" | "compositionend", data = "") {
  act(() => input.dispatchEvent(new CompositionEvent(type, { bubbles: true, data })));
}
function edit(value: string, isComposing: boolean, cursor = value.length) {
  let valueAtEventEnd = "";
  let cursorAtEventEnd: number | null = null;
  act(() => {
    // Model the browser's native edit, bypassing React's value tracker.
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")!.set!.call(input, value);
    input.setSelectionRange(cursor, cursor);
    input.dispatchEvent(new InputEvent("input", {
      bubbles: true, isComposing, inputType: isComposing ? "insertCompositionText" : "insertText",
    }));
    // Assert before timers/act flushes can conceal a controlled-input rollback.
    valueAtEventEnd = input.value;
    cursorAtEventEnd = input.selectionStart;
  });
  expect(valueAtEventEnd).toBe(value);
  expect(cursorAtEventEnd).toBe(cursor);
}
function enter(options: { isComposing?: boolean; shiftKey?: boolean } = {}) {
  const event = new KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true, ...options });
  act(() => input.dispatchEvent(event));
  return event;
}

describe("Composer input synchronization", () => {
  it("keeps each IME edit and commits selected Chinese without stale pinyin", async () => {
    composition("compositionstart");
    for (const value of ["n", "ni", "nih", "niha", "nihao"]) edit(value, true);
    expect(enter({ isComposing: true }).defaultPrevented).toBe(false);
    expect(onRun).not.toHaveBeenCalled();
    edit("你好", true);
    composition("compositionend", "你好");
    edit("你好", false);
    composition("compositionstart");
    for (const value of ["你好m", "你好ma", "你好吗"]) edit(value, true);
    composition("compositionend", "吗");
    edit("你好吗", false);
    await act(async () => {});
    expect(input.value).toBe("你好吗");
    expect(onRun).not.toHaveBeenCalled();
  });

  it("does not duplicate letters or reset the caret during rapid edits", () => {
    for (const value of ["a", "aa", "aaa", "aaaa", "aaaaa"]) edit(value, false);
    edit("aaXaaa", false, 3);
    edit("aaXYaaa", false, 4);
    expect(input.value).toBe("aaXYaaa");
    expect(input.selectionStart).toBe(4);
  });

  it("allows Shift+Enter and sends the committed text only once on Enter", async () => {
    edit("你好", false);
    expect(enter({ shiftKey: true }).defaultPrevented).toBe(false);
    expect(onRun).not.toHaveBeenCalled();
    edit("你好\n测试", false);
    enter();
    await act(async () => { await new Promise(resolve => setTimeout(resolve, 20)); });
    expect(onRun).toHaveBeenCalledTimes(1);
    expect(onRun).toHaveBeenCalledWith([{ type: "text", text: "你好\n测试" }]);
    expect(input.value).toBe("");
  });
});
