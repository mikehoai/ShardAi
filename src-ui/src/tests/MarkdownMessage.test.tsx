import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
  MarkdownMessage,
  parseBlocks,
  parseHeadingLine,
} from "../components/MarkdownMessage";

describe("MarkdownMessage linear O(N) ATX heading parser (Issue #84)", () => {
  it("parses and renders standard # through ###### headings", () => {
    const markdown = [
      "# Heading Level 1",
      " ## Heading Level 2",
      "  ### Heading Level 3",
      "   #### Heading Level 4",
      "#####\tHeading Level 5",
      "######   Heading Level 6   ",
    ].join("\n");

    const blocks = parseBlocks(markdown);
    expect(blocks).toEqual([
      { type: "heading", level: 1, text: "Heading Level 1" },
      { type: "heading", level: 2, text: "Heading Level 2" },
      { type: "heading", level: 3, text: "Heading Level 3" },
      { type: "heading", level: 4, text: "Heading Level 4" },
      { type: "heading", level: 5, text: "Heading Level 5" },
      { type: "heading", level: 6, text: "Heading Level 6" },
    ]);

    const { container } = render(<MarkdownMessage content={markdown} />);
    for (let level = 1; level <= 6; level += 1) {
      const el = container.querySelector(`h${level}.chat-md-heading.chat-md-h${level}`);
      expect(el).not.toBeNull();
      expect(el?.textContent).toBe(`Heading Level ${level}`);
    }
  });

  it("strips optional closing ### sequences while preserving literal # in '# C#' and '# Issue #84'", () => {
    expect(parseHeadingLine("# Heading ###   ")).toEqual({
      level: 1,
      text: "Heading",
    });
    expect(parseHeadingLine("## Sub Heading\t##\t\r")).toEqual({
      level: 2,
      text: "Sub Heading",
    });
    expect(parseHeadingLine("# C#")).toEqual({
      level: 1,
      text: "C#",
    });
    expect(parseHeadingLine("# C#   ")).toEqual({
      level: 1,
      text: "C#",
    });
    expect(parseHeadingLine("# C# ###   ")).toEqual({
      level: 1,
      text: "C#",
    });
    expect(parseHeadingLine("# Issue #84")).toEqual({
      level: 1,
      text: "Issue #84",
    });
    expect(parseHeadingLine("### Escaped \\#")).toEqual({
      level: 3,
      text: "Escaped \\#",
    });

    const { container } = render(
      <MarkdownMessage content={"# C#\n## Issue #84\n### Escaped \\# ###   "} />
    );
    expect(container.querySelector("h1")?.textContent).toBe("C#");
    expect(container.querySelector("h2")?.textContent).toBe("Issue #84");
    expect(container.querySelector("h3")?.textContent).toBe("Escaped #");
  });

  it("renders inline formatting inside headings and handles table/blockquote boundaries", () => {
    const markdown = [
      "## Release **v1.2**, *linear* `O(N)` & [Guide](https://shardai.online/docs) ###   ",
      "",
      "> ### Quoted **Heading** ##",
      "",
      "| Col A | Col B |",
      "| :--- | :--- |",
      "| 1 | 2 |",
      "# Heading | After Table",
    ].join("\n");

    const { container } = render(<MarkdownMessage content={markdown} />);

    const h2 = container.querySelector("h2.chat-md-heading");
    expect(h2).not.toBeNull();
    expect(h2?.querySelector("strong")?.textContent).toBe("v1.2");
    expect(h2?.querySelector("em")?.textContent).toBe("linear");
    expect(h2?.querySelector("code.chat-md-inline-code")?.textContent).toBe("O(N)");
    expect(h2?.querySelector("a.chat-md-link")?.getAttribute("href")).toBe(
      "https://shardai.online/docs"
    );
    expect(h2?.textContent).toBe("Release v1.2, linear O(N) & Guide");

    const quotedH3 = container.querySelector("blockquote h3.chat-md-heading");
    expect(quotedH3?.textContent).toBe("Quoted Heading");

    expect(container.querySelectorAll("table.chat-md-table tbody tr")).toHaveLength(1);
    const trailingH1 = container.querySelector("h1.chat-md-heading");
    expect(trailingH1?.textContent).toBe("Heading | After Table");
  });

  it("treats '#   ' and empty-after-stripping lines consistently as non-headings across isBlockStarter and parseBlocks", () => {
    expect(parseHeadingLine("#   ")).toBeNull();
    expect(parseHeadingLine("#\t\t\r")).toBeNull();
    expect(parseHeadingLine("# ###")).toBeNull();
    expect(parseHeadingLine("# ###   ")).toBeNull();
    expect(parseHeadingLine("    # Four leading spaces")).toBeNull();
    expect(parseHeadingLine("####### Seven hashes")).toBeNull();
    expect(parseHeadingLine("#NoSpaceAfterHash")).toBeNull();
    expect(parseHeadingLine("# Multi\nLine")).toBeNull();

    expect(parseBlocks("#   ")).toEqual([{ type: "paragraph", lines: ["#   "] }]);
    expect(parseBlocks("First paragraph line\n#   \n# ###   ")).toEqual([
      {
        type: "paragraph",
        lines: ["First paragraph line", "#   ", "# ###   "],
      },
    ]);

    const { container } = render(<MarkdownMessage content="#   " />);
    expect(container.querySelector("h1")).toBeNull();
    expect(container.querySelector("p.chat-md-paragraph")?.textContent).toBe("#   ");
  });

  it("parses and renders pathological ReDoS heading input in linear time (< 50 ms)", () => {
    const line = "# " + " ".repeat(15000) + "x\t" + "#".repeat(15000) + "!";
    const expectedText = "x\t" + "#".repeat(15000) + "!";

    const parseStart = performance.now();
    const blocks = parseBlocks(line);
    const parseElapsedMs = performance.now() - parseStart;

    expect(parseElapsedMs).toBeLessThan(50);
    expect(blocks).toEqual([
      {
        type: "heading",
        level: 1,
        text: expectedText,
      },
    ]);

    const renderStart = performance.now();
    const { container } = render(<MarkdownMessage content={line} isStreaming={true} />);
    const renderElapsedMs = performance.now() - renderStart;

    expect(renderElapsedMs).toBeLessThan(50);
    const h1 = container.querySelector("h1.chat-md-heading");
    expect(h1?.textContent).toBe(expectedText);
    expect(screen.getByTestId("chat-streaming-cursor")).toBeTruthy();
  });

  it("handles trailing-hash pathological input in linear time (< 50 ms)", () => {
    const closingLine = "# " + " ".repeat(15000) + "Streaming Title " + "#".repeat(15000) + "   ";

    const start = performance.now();
    const parsed = parseHeadingLine(closingLine);
    const elapsedMs = performance.now() - start;

    expect(elapsedMs).toBeLessThan(50);
    expect(parsed).toEqual({
      level: 1,
      text: "Streaming Title",
    });
  });
});
