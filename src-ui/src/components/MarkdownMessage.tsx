import React, { useEffect, useRef, useState } from "react";

export interface MarkdownMessageProps {
  content: string;
  isStreaming?: boolean;
  onCopyCode?: (code: string) => void;
}

const COPY_FEEDBACK_DURATION_MS = 1800;
const MAX_BLOCKQUOTE_DEPTH = 6;

export async function copyTextToClipboard(text: string): Promise<boolean> {
  if (
    typeof navigator !== "undefined" &&
    navigator.clipboard &&
    typeof navigator.clipboard.writeText === "function"
  ) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // Fall through to execCommand fallback when Clipboard API is unavailable or denied.
    }
  }

  if (typeof document === "undefined" || !document.body) {
    return false;
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.className = "chat-clipboard-fallback";
  document.body.appendChild(textarea);

  try {
    textarea.select();
    textarea.setSelectionRange(0, text.length);
    return typeof document.execCommand === "function"
      ? Boolean(document.execCommand("copy"))
      : false;
  } catch {
    return false;
  } finally {
    document.body.removeChild(textarea);
  }
}

export type TableAlignment = "left" | "center" | "right" | null;

export type HeadingLevel = 1 | 2 | 3 | 4 | 5 | 6;

export interface ParsedHeading {
  level: HeadingLevel;
  text: string;
}

export type MarkdownBlock =
  | { type: "code"; language: string; code: string }
  | { type: "heading"; level: HeadingLevel; text: string }
  | { type: "hr" }
  | { type: "blockquote"; blocks: MarkdownBlock[] }
  | {
      type: "table";
      headers: string[];
      alignments: TableAlignment[];
      rows: string[][];
    }
  | { type: "list"; ordered: boolean; start: number; items: string[] }
  | { type: "paragraph"; lines: string[] };

const FENCE_OPEN_RE = /^\s{0,3}```(.*)$/;
const FENCE_CLOSE_RE = /^\s{0,3}```\s*$/;
const HR_RE = /^\s{0,3}([-*_])(?:\s*\1){2,}\s*$/;
const HEADING_PREFIX_RE = /^[ \t]{0,3}(#{1,6})[ \t]+/;
const BLOCKQUOTE_RE = /^\s{0,3}>\s?(.*)$/;
const UL_ITEM_RE = /^\s*[-*+]\s+(.*)$/;
const OL_ITEM_RE = /^\s*(\d+)\.\s+(.*)$/;
const TABLE_SEP_CELL_RE = /^:?-{1,}:?$/;
const WORD_CHAR_RE = /^[A-Za-z0-9]$/;
const ESCAPABLE_PUNCT_RE = /^[\\`*_{}[\]()#+\-.!~<>|]$/;

function isHorizontalWhitespace(ch: string | undefined): boolean {
  return ch === " " || ch === "\t" || ch === "\r";
}

export function parseHeadingLine(line: string): ParsedHeading | null {
  if (line.includes("\n")) {
    return null;
  }

  const prefixMatch = HEADING_PREFIX_RE.exec(line);
  if (!prefixMatch) {
    return null;
  }

  const start = prefixMatch[0].length;
  let end = line.length;

  while (end > start && isHorizontalWhitespace(line[end - 1])) {
    end -= 1;
  }
  if (end <= start) {
    return null;
  }

  let hashStart = end;
  while (hashStart > start && line[hashStart - 1] === "#") {
    hashStart -= 1;
  }

  if (hashStart < end) {
    if (hashStart === start) {
      return null;
    }
    if (isHorizontalWhitespace(line[hashStart - 1])) {
      end = hashStart;
      while (end > start && isHorizontalWhitespace(line[end - 1])) {
        end -= 1;
      }
      if (end <= start) {
        return null;
      }
    }
  }

  const level = Math.min(6, Math.max(1, prefixMatch[1].length)) as HeadingLevel;
  return {
    level,
    text: line.slice(start, end),
  };
}

function isWordChar(ch: string | undefined): boolean {
  return ch !== undefined && WORD_CHAR_RE.test(ch);
}

function isWhitespaceChar(ch: string | undefined): boolean {
  return ch === undefined || /\s/.test(ch);
}

function splitTableRow(line: string): string[] {
  const cells: string[] = [];
  let current = "";
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === "\\" && i + 1 < line.length && line[i + 1] === "|") {
      current += "|";
      i += 1;
      continue;
    }
    if (ch === "|") {
      cells.push(current);
      current = "";
      continue;
    }
    current += ch;
  }
  cells.push(current);

  if (cells.length > 0 && cells[0].trim() === "") {
    cells.shift();
  }
  if (cells.length > 0 && cells[cells.length - 1].trim() === "") {
    cells.pop();
  }
  return cells.map((cell) => cell.trim());
}

function parseTableSeparator(line: string): TableAlignment[] | null {
  if (!line.includes("|")) {
    return null;
  }
  const cells = splitTableRow(line);
  if (cells.length === 0) {
    return null;
  }
  const alignments: TableAlignment[] = [];
  for (const cell of cells) {
    if (!TABLE_SEP_CELL_RE.test(cell)) {
      return null;
    }
    const startsColon = cell.startsWith(":");
    const endsColon = cell.endsWith(":");
    if (startsColon && endsColon) {
      alignments.push("center");
    } else if (endsColon) {
      alignments.push("right");
    } else if (startsColon) {
      alignments.push("left");
    } else {
      alignments.push(null);
    }
  }
  return alignments;
}

function isBlockStarter(lines: string[], index: number): boolean {
  const line = lines[index];
  if (line.trim() === "") {
    return true;
  }
  if (
    FENCE_OPEN_RE.test(line) ||
    HR_RE.test(line) ||
    parseHeadingLine(line) !== null ||
    BLOCKQUOTE_RE.test(line) ||
    UL_ITEM_RE.test(line) ||
    OL_ITEM_RE.test(line)
  ) {
    return true;
  }
  if (
    index + 1 < lines.length &&
    line.includes("|") &&
    parseTableSeparator(lines[index + 1]) !== null
  ) {
    return true;
  }
  return false;
}

export function parseBlocks(raw: string, depth = 0): MarkdownBlock[] {
  const normalized = raw.replace(/\r\n?/g, "\n");
  const lines = normalized.split("\n");
  const blocks: MarkdownBlock[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (line.trim() === "") {
      i += 1;
      continue;
    }

    const fenceMatch = FENCE_OPEN_RE.exec(line);
    if (fenceMatch) {
      const rawLang = fenceMatch[1].trim();
      const language = rawLang ? rawLang.split(/\s+/)[0] : "code";
      const codeLines: string[] = [];
      i += 1;
      while (i < lines.length) {
        if (FENCE_CLOSE_RE.test(lines[i])) {
          i += 1;
          break;
        }
        codeLines.push(lines[i]);
        i += 1;
      }
      blocks.push({
        type: "code",
        language,
        code: codeLines.join("\n"),
      });
      continue;
    }

    if (HR_RE.test(line)) {
      blocks.push({ type: "hr" });
      i += 1;
      continue;
    }

    const parsedHeading = parseHeadingLine(line);
    if (parsedHeading !== null) {
      blocks.push({
        type: "heading",
        level: parsedHeading.level,
        text: parsedHeading.text,
      });
      i += 1;
      continue;
    }

    if (BLOCKQUOTE_RE.test(line) && depth < MAX_BLOCKQUOTE_DEPTH) {
      const quoteLines: string[] = [];
      while (i < lines.length) {
        const quoteMatch = BLOCKQUOTE_RE.exec(lines[i]);
        if (!quoteMatch) {
          break;
        }
        quoteLines.push(quoteMatch[1]);
        i += 1;
      }
      blocks.push({
        type: "blockquote",
        blocks: parseBlocks(quoteLines.join("\n"), depth + 1),
      });
      continue;
    }

    if (i + 1 < lines.length && line.includes("|")) {
      const alignments = parseTableSeparator(lines[i + 1]);
      if (alignments !== null) {
        const headers = splitTableRow(line);
        const colCount = Math.max(headers.length, alignments.length);
        const normalizedHeaders = Array.from(
          { length: colCount },
          (_, colIdx) => headers[colIdx] ?? ""
        );
        const normalizedAlignments = Array.from(
          { length: colCount },
          (_, colIdx) => alignments[colIdx] ?? null
        );
        const rows: string[][] = [];
        i += 2;
        while (i < lines.length) {
          const rowLine = lines[i];
          if (rowLine.trim() === "" || !rowLine.includes("|")) {
            break;
          }
          if (
            FENCE_OPEN_RE.test(rowLine) ||
            parseHeadingLine(rowLine) !== null ||
            BLOCKQUOTE_RE.test(rowLine)
          ) {
            break;
          }
          const cells = splitTableRow(rowLine);
          rows.push(
            Array.from({ length: colCount }, (_, colIdx) => cells[colIdx] ?? "")
          );
          i += 1;
        }
        blocks.push({
          type: "table",
          headers: normalizedHeaders,
          alignments: normalizedAlignments,
          rows,
        });
        continue;
      }
    }

    const ulMatch = UL_ITEM_RE.exec(line);
    if (ulMatch) {
      const items: string[] = [];
      while (i < lines.length) {
        const match = UL_ITEM_RE.exec(lines[i]);
        if (!match || HR_RE.test(lines[i])) {
          break;
        }
        items.push(match[1]);
        i += 1;
      }
      blocks.push({ type: "list", ordered: false, start: 1, items });
      continue;
    }

    const olMatch = OL_ITEM_RE.exec(line);
    if (olMatch) {
      const start = Number.parseInt(olMatch[1], 10) || 1;
      const items: string[] = [];
      while (i < lines.length) {
        const match = OL_ITEM_RE.exec(lines[i]);
        if (!match) {
          break;
        }
        items.push(match[2]);
        i += 1;
      }
      blocks.push({ type: "list", ordered: true, start, items });
      continue;
    }

    const paragraphLines: string[] = [line];
    i += 1;
    while (i < lines.length && !isBlockStarter(lines, i)) {
      paragraphLines.push(lines[i]);
      i += 1;
    }
    blocks.push({ type: "paragraph", lines: paragraphLines });
  }

  return blocks;
}

function isSafeLinkUrl(rawUrl: string): boolean {
  const normalized = rawUrl.trim().toLowerCase();
  return (
    normalized.startsWith("https://") ||
    normalized.startsWith("http://") ||
    normalized.startsWith("mailto:")
  );
}

function findClosingInlineCode(text: string, openEnd: number, tickCount: number): number {
  let i = openEnd;
  while (i < text.length) {
    if (text[i] === "`") {
      let run = 0;
      while (i + run < text.length && text[i + run] === "`") {
        run += 1;
      }
      if (run === tickCount) {
        return i;
      }
      i += run;
      continue;
    }
    i += 1;
  }
  return -1;
}

function findMatchingBracket(text: string, openIndex: number): number {
  let depth = 0;
  for (let i = openIndex; i < text.length; i += 1) {
    const ch = text[i];
    if (ch === "\\" && i + 1 < text.length) {
      i += 1;
      continue;
    }
    if (ch === "[") {
      depth += 1;
    } else if (ch === "]") {
      depth -= 1;
      if (depth === 0) {
        return i;
      }
    }
  }
  return -1;
}

function findMatchingParen(text: string, openIndex: number): number {
  let depth = 0;
  for (let i = openIndex; i < text.length; i += 1) {
    const ch = text[i];
    if (ch === "\\" && i + 1 < text.length) {
      i += 1;
      continue;
    }
    if (ch === "(") {
      depth += 1;
    } else if (ch === ")") {
      depth -= 1;
      if (depth === 0) {
        return i;
      }
    }
  }
  return -1;
}

function findClosingDelimiter(
  text: string,
  fromIndex: number,
  delimiter: "**" | "__" | "~~" | "*" | "_"
): number {
  const len = delimiter.length;
  let i = fromIndex;
  while (i <= text.length - len) {
    if (text[i] === "\\" && i + 1 < text.length) {
      i += 2;
      continue;
    }
    if (text[i] === "`") {
      let tickCount = 0;
      while (i + tickCount < text.length && text[i + tickCount] === "`") {
        tickCount += 1;
      }
      const closeCode = findClosingInlineCode(text, i + tickCount, tickCount);
      if (closeCode !== -1) {
        i = closeCode + tickCount;
        continue;
      }
    }
    if (text.startsWith(delimiter, i)) {
      const prevChar = i > 0 ? text[i - 1] : undefined;
      const nextChar = i + len < text.length ? text[i + len] : undefined;
      if (delimiter === "*" || delimiter === "**") {
        if (!isWhitespaceChar(prevChar)) {
          if (delimiter === "*" && nextChar === "*") {
            // Skip '**' pair when searching for single '*' closing delimiter.
            i += 2;
            continue;
          }
          return i;
        }
      } else if (delimiter === "_" || delimiter === "__") {
        if (!isWhitespaceChar(prevChar) && !isWordChar(nextChar)) {
          if (delimiter === "_" && nextChar === "_") {
            i += 2;
            continue;
          }
          return i;
        }
      } else if (i > fromIndex) {
        return i;
      }
    }
    i += 1;
  }
  return -1;
}

function parseInline(text: string, keyPrefix = "in", depth = 0): React.ReactNode[] {
  if (depth > 16) {
    return [text];
  }

  const nodes: React.ReactNode[] = [];
  let plainBuffer = "";
  let keyCounter = 0;
  let i = 0;

  const flushPlain = (): void => {
    if (plainBuffer.length > 0) {
      nodes.push(plainBuffer);
      plainBuffer = "";
    }
  };

  const nextKey = (): string => {
    keyCounter += 1;
    return `${keyPrefix}-${keyCounter}`;
  };

  while (i < text.length) {
    const ch = text[i];

    // Escaped punctuation
    if (ch === "\\" && i + 1 < text.length && ESCAPABLE_PUNCT_RE.test(text[i + 1])) {
      plainBuffer += text[i + 1];
      i += 2;
      continue;
    }

    // Inline code
    if (ch === "`") {
      let tickCount = 0;
      while (i + tickCount < text.length && text[i + tickCount] === "`") {
        tickCount += 1;
      }
      const closeIdx = findClosingInlineCode(text, i + tickCount, tickCount);
      if (closeIdx !== -1) {
        flushPlain();
        let codeContent = text.slice(i + tickCount, closeIdx);
        if (
          codeContent.length >= 2 &&
          codeContent.startsWith(" ") &&
          codeContent.endsWith(" ") &&
          codeContent.trim().length > 0
        ) {
          codeContent = codeContent.slice(1, -1);
        }
        nodes.push(
          <code key={nextKey()} className="chat-md-inline-code">
            {codeContent}
          </code>
        );
        i = closeIdx + tickCount;
        continue;
      }
      plainBuffer += text.slice(i, i + tickCount);
      i += tickCount;
      continue;
    }

    // Markdown link [label](url)
    if (ch === "[") {
      const closeBracket = findMatchingBracket(text, i);
      if (closeBracket !== -1 && text[closeBracket + 1] === "(") {
        const closeParen = findMatchingParen(text, closeBracket + 1);
        if (closeParen !== -1) {
          const label = text.slice(i + 1, closeBracket);
          const rawUrl = text.slice(closeBracket + 2, closeParen).trim();
          flushPlain();
          if (isSafeLinkUrl(rawUrl)) {
            const key = nextKey();
            nodes.push(
              <a
                key={key}
                href={rawUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="chat-md-link"
              >
                {parseInline(label, `${key}-a`, depth + 1)}
              </a>
            );
          } else {
            // Render unsafe protocols (e.g., javascript:, data:) strictly as plain text.
            nodes.push(text.slice(i, closeParen + 1));
          }
          i = closeParen + 1;
          continue;
        }
      }
    }

    // Strikethrough ~~text~~
    if (text.startsWith("~~", i) && !isWhitespaceChar(text[i + 2])) {
      const closeStrike = findClosingDelimiter(text, i + 2, "~~");
      if (closeStrike !== -1 && closeStrike > i + 2) {
        flushPlain();
        const key = nextKey();
        nodes.push(
          <del key={key}>
            {parseInline(text.slice(i + 2, closeStrike), `${key}-del`, depth + 1)}
          </del>
        );
        i = closeStrike + 2;
        continue;
      }
    }

    // Bold **text**
    if (text.startsWith("**", i) && !isWhitespaceChar(text[i + 2])) {
      const closeBold = findClosingDelimiter(text, i + 2, "**");
      if (closeBold !== -1 && closeBold > i + 2) {
        flushPlain();
        const key = nextKey();
        nodes.push(
          <strong key={key}>
            {parseInline(text.slice(i + 2, closeBold), `${key}-b`, depth + 1)}
          </strong>
        );
        i = closeBold + 2;
        continue;
      }
    }

    // Bold __text__ (only on word boundaries)
    if (
      text.startsWith("__", i) &&
      !isWordChar(i > 0 ? text[i - 1] : undefined) &&
      !isWhitespaceChar(text[i + 2])
    ) {
      const closeBold = findClosingDelimiter(text, i + 2, "__");
      if (closeBold !== -1 && closeBold > i + 2) {
        flushPlain();
        const key = nextKey();
        nodes.push(
          <strong key={key}>
            {parseInline(text.slice(i + 2, closeBold), `${key}-ub`, depth + 1)}
          </strong>
        );
        i = closeBold + 2;
        continue;
      }
    }

    // Italic *text*
    if (ch === "*" && text[i + 1] !== "*" && !isWhitespaceChar(text[i + 1])) {
      const closeItalic = findClosingDelimiter(text, i + 1, "*");
      if (closeItalic !== -1 && closeItalic > i + 1) {
        flushPlain();
        const key = nextKey();
        nodes.push(
          <em key={key}>
            {parseInline(text.slice(i + 1, closeItalic), `${key}-em`, depth + 1)}
          </em>
        );
        i = closeItalic + 1;
        continue;
      }
    }

    // Italic _text_ (preserve snake_case_words)
    if (
      ch === "_" &&
      text[i + 1] !== "_" &&
      !isWordChar(i > 0 ? text[i - 1] : undefined) &&
      !isWhitespaceChar(text[i + 1])
    ) {
      const closeItalic = findClosingDelimiter(text, i + 1, "_");
      if (closeItalic !== -1 && closeItalic > i + 1) {
        flushPlain();
        const key = nextKey();
        nodes.push(
          <em key={key}>
            {parseInline(text.slice(i + 1, closeItalic), `${key}-uem`, depth + 1)}
          </em>
        );
        i = closeItalic + 1;
        continue;
      }
    }

    plainBuffer += ch;
    i += 1;
  }

  flushPlain();
  return nodes;
}

function getAlignmentClass(alignment: TableAlignment): string {
  if (alignment === "left") return "chat-md-align-left";
  if (alignment === "center") return "chat-md-align-center";
  if (alignment === "right") return "chat-md-align-right";
  return "";
}

export const MarkdownMessage: React.FC<MarkdownMessageProps> = ({
  content,
  isStreaming = false,
  onCopyCode,
}) => {
  const [copiedBlockKey, setCopiedBlockKey] = useState<string | null>(null);
  const copyTimerRef = useRef<number | null>(null);
  const isMountedRef = useRef<boolean>(true);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      if (copyTimerRef.current !== null) {
        window.clearTimeout(copyTimerRef.current);
        copyTimerRef.current = null;
      }
    };
  }, []);

  const handleCopyCodeBlock = async (blockKey: string, code: string): Promise<void> => {
    onCopyCode?.(code);
    await copyTextToClipboard(code);
    if (!isMountedRef.current) {
      return;
    }
    if (copyTimerRef.current !== null) {
      window.clearTimeout(copyTimerRef.current);
    }
    setCopiedBlockKey(blockKey);
    copyTimerRef.current = window.setTimeout(() => {
      if (isMountedRef.current) {
        setCopiedBlockKey(null);
      }
      copyTimerRef.current = null;
    }, COPY_FEEDBACK_DURATION_MS);
  };

  const renderBlocks = (blocks: MarkdownBlock[], prefix: string): React.ReactNode[] =>
    blocks.map((block, idx) => {
      const key = `${prefix}-${idx}`;
      switch (block.type) {
        case "code": {
          const isCopied = copiedBlockKey === key;
          return (
            <div key={key} className="chat-md-code-block">
              <div className="chat-md-code-header">
                <span className="chat-md-code-lang">{block.language}</span>
                <button
                  type="button"
                  data-testid="chat-code-copy-btn"
                  className={`chat-md-code-copy-btn${isCopied ? " copied" : ""}`}
                  aria-label={isCopied ? "Copied code" : "Copy code"}
                  onClick={() => void handleCopyCodeBlock(key, block.code)}
                >
                  {isCopied ? "Copied" : "Copy code"}
                </button>
              </div>
              <pre className="chat-md-pre">
                <code>{block.code}</code>
              </pre>
            </div>
          );
        }
        case "heading": {
          const HeadingTag = `h${block.level}` as
            | "h1"
            | "h2"
            | "h3"
            | "h4"
            | "h5"
            | "h6";
          return (
            <HeadingTag key={key} className={`chat-md-heading chat-md-h${block.level}`}>
              {parseInline(block.text, `${key}-h`)}
            </HeadingTag>
          );
        }
        case "hr":
          return <hr key={key} className="chat-md-hr" />;
        case "blockquote":
          return (
            <blockquote key={key} className="chat-md-blockquote">
              {renderBlocks(block.blocks, `${key}-bq`)}
            </blockquote>
          );
        case "table":
          return (
            <div key={key} className="chat-md-table-wrap">
              <table className="chat-md-table">
                <thead>
                  <tr>
                    {block.headers.map((header, colIdx) => (
                      <th
                        key={`${key}-th-${colIdx}`}
                        className={getAlignmentClass(block.alignments[colIdx] ?? null)}
                      >
                        {parseInline(header, `${key}-th-${colIdx}`)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {block.rows.map((row, rowIdx) => (
                    <tr key={`${key}-tr-${rowIdx}`}>
                      {row.map((cell, colIdx) => (
                        <td
                          key={`${key}-td-${rowIdx}-${colIdx}`}
                          className={getAlignmentClass(block.alignments[colIdx] ?? null)}
                        >
                          {parseInline(cell, `${key}-td-${rowIdx}-${colIdx}`)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        case "list": {
          const ListTag = block.ordered ? "ol" : "ul";
          return (
            <ListTag
              key={key}
              className={`chat-md-list ${block.ordered ? "chat-md-ol" : "chat-md-ul"}`}
              {...(block.ordered && block.start !== 1 ? { start: block.start } : {})}
            >
              {block.items.map((item, itemIdx) => (
                <li key={`${key}-li-${itemIdx}`} className="chat-md-li">
                  {parseInline(item, `${key}-li-${itemIdx}`)}
                </li>
              ))}
            </ListTag>
          );
        }
        case "paragraph":
          return (
            <p key={key} className="chat-md-paragraph">
              {block.lines.map((pLine, lineIdx) => (
                <React.Fragment key={`${key}-pl-${lineIdx}`}>
                  {lineIdx > 0 && <br />}
                  {parseInline(pLine, `${key}-pl-${lineIdx}`)}
                </React.Fragment>
              ))}
            </p>
          );
      }
    });

  const blocks = parseBlocks(content);

  return (
    <div className="chat-md-root">
      {renderBlocks(blocks, "root")}
      {isStreaming && content.length > 0 && (
        <span
          className="chat-streaming-cursor"
          data-testid="chat-streaming-cursor"
          aria-hidden="true"
        >
          ▍
        </span>
      )}
    </div>
  );
};

export default MarkdownMessage;
