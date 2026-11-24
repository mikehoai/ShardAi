import mammoth from "mammoth";
import * as pdfjsLib from "pdfjs-dist";
import pdfWorkerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";

export const MAX_DOC_TEXT_CHARS = 6000;

let workerConfigured = false;

function ensurePdfWorker(): void {
  if (!workerConfigured) {
    pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;
    workerConfigured = true;
  }
}

function getFileExtension(name: string): string {
  const dotIdx = name.lastIndexOf(".");
  return dotIdx >= 0 ? name.slice(dotIdx).toLowerCase() : "";
}

async function readFileAsArrayBuffer(file: File): Promise<ArrayBuffer> {
  if (typeof file.arrayBuffer === "function") {
    return file.arrayBuffer();
  }
  return new Promise<ArrayBuffer>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (reader.result instanceof ArrayBuffer) {
        resolve(reader.result);
      } else {
        reject(new Error(`Unable to read "${file.name}" as binary data.`));
      }
    };
    reader.onerror = () => {
      reject(new Error(`Failed to read "${file.name}".`));
    };
    reader.readAsArrayBuffer(file);
  });
}

async function readPlainText(file: File): Promise<string> {
  if (typeof file.text === "function") {
    return file.text();
  }
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      resolve(typeof reader.result === "string" ? reader.result : "");
    };
    reader.onerror = () => {
      reject(new Error(`Failed to read text from "${file.name}".`));
    };
    reader.readAsText(file);
  });
}

function truncateText(text: string, maxChars: number): string {
  if (text.length > maxChars) {
    return `${text.slice(0, maxChars)}\n...[truncated to fit model context]`;
  }
  return text;
}

async function extractDocxText(file: File): Promise<string> {
  const arrayBuffer = await readFileAsArrayBuffer(file);
  const result = await mammoth.extractRawText({ arrayBuffer });
  const cleaned = result.value.trim();
  if (!cleaned) {
    throw new Error(`No readable text found in "${file.name}".`);
  }
  return cleaned;
}

async function extractPdfText(file: File): Promise<string> {
  ensurePdfWorker();
  const arrayBuffer = await readFileAsArrayBuffer(file);
  const loadingTask = pdfjsLib.getDocument({
    data: new Uint8Array(arrayBuffer),
  });
  try {
    const pdfDoc = await loadingTask.promise;
    try {
      const pageTexts: string[] = [];
      for (let pageNum = 1; pageNum <= pdfDoc.numPages; pageNum += 1) {
        const page = await pdfDoc.getPage(pageNum);
        const textContent = await page.getTextContent();
        const parts: string[] = [];
        for (const item of textContent.items) {
          if (typeof item === "object" && item !== null && "str" in item && typeof item.str === "string") {
            if (item.str) {
              parts.push(item.str);
            }
          }
        }
        const joined = parts.join(" ").trim();
        if (joined) {
          pageTexts.push(joined);
        }
      }
      const cleaned = pageTexts.join("\n\n").trim();
      if (!cleaned) {
        throw new Error(
          `No readable text could be extracted from "${file.name}" (it may be a scanned/image-only PDF).`
        );
      }
      return cleaned;
    } finally {
      await pdfDoc.cleanup();
    }
  } finally {
    await loadingTask.destroy();
  }
}

export async function extractDocumentText(
  file: File,
  maxChars: number = MAX_DOC_TEXT_CHARS
): Promise<string> {
  const mime = (file.type || "").toLowerCase();
  const ext = getFileExtension(file.name);

  if (ext === ".doc" || mime === "application/msword") {
    throw new Error(
      `Cannot read legacy binary .doc file "${file.name}". Please convert it to .docx, .pdf, or .txt.`
    );
  }

  let extracted: string;
  if (
    ext === ".docx" ||
    mime === "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
  ) {
    extracted = await extractDocxText(file);
  } else if (ext === ".pdf" || mime === "application/pdf") {
    extracted = await extractPdfText(file);
  } else {
    extracted = (await readPlainText(file)).trim();
    if (!extracted) {
      throw new Error(`File "${file.name}" is empty.`);
    }
  }

  return truncateText(extracted, maxChars);
}
