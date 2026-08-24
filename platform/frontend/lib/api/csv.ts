import type {
  AnalysisKind,
  CsvPreviewRow,
  CsvValidationRequest,
  CsvValidationResult,
} from "./types";

const normaliseColumn = (value: string) =>
  value.trim().toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");

export function parseCsv(content: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;

  for (let index = 0; index < content.length; index += 1) {
    const character = content[index];
    if (quoted) {
      if (character === '"' && content[index + 1] === '"') {
        field += '"';
        index += 1;
      } else if (character === '"') {
        quoted = false;
      } else {
        field += character;
      }
      continue;
    }
    if (character === '"') {
      quoted = true;
    } else if (character === ",") {
      row.push(field.trim());
      field = "";
    } else if (character === "\n") {
      row.push(field.trim());
      if (row.some(Boolean)) rows.push(row);
      row = [];
      field = "";
    } else if (character !== "\r") {
      field += character;
    }
  }
  if (quoted) throw new Error("The CSV contains an unclosed quoted field.");
  row.push(field.trim());
  if (row.some(Boolean)) rows.push(row);
  return rows;
}

const requiredColumns: Record<AnalysisKind, string[]> = {
  "event-counts": ["cell_id", "count"],
  trajectory: ["cell_id", "history"],
};

function isBinaryHistory(value: string) {
  const compact = value.replace(/[\s,;|\[\]()]/g, "");
  return compact === "" || /^[01]+$/.test(compact);
}

export function validateCsvLocally(request: CsvValidationRequest): CsvValidationResult {
  const warnings: string[] = [];
  const errors: string[] = [];
  let rows: string[][] = [];
  try {
    rows = parseCsv(request.content);
  } catch (error) {
    errors.push(error instanceof Error ? error.message : "The CSV could not be parsed.");
  }

  const rawColumns = rows[0] ?? [];
  const columns = rawColumns.map(normaliseColumn);
  const body = rows.slice(1).filter((row) => row.some(Boolean));
  const duplicateColumns = columns.filter((column, index) => columns.indexOf(column) !== index);
  if (duplicateColumns.length) errors.push(`Duplicate column: ${duplicateColumns[0]}.`);
  for (const required of requiredColumns[request.kind]) {
    if (!columns.includes(required)) errors.push(`Missing required column: ${required}.`);
  }
  if (!body.length) errors.push("Add at least one data row.");

  const conditionIndex = columns.indexOf("condition");
  const conditions = Array.from(
    new Set(
      body.map((row) => row[conditionIndex]?.trim() || "Condition 1"),
    ),
  );
  if (conditions.length > 4) errors.push("At most four experimental conditions are supported.");
  if (conditionIndex < 0) warnings.push("No condition column was found; all rows will use Condition 1.");

  const countIndex = columns.indexOf("count");
  if (request.kind === "event-counts" && countIndex >= 0) {
    const invalid = body.findIndex((row) => !/^\d+$/.test(row[countIndex]?.trim() ?? ""));
    if (invalid >= 0) errors.push(`Count in data row ${invalid + 1} must be a non-negative integer.`);
  }
  const historyIndex = columns.indexOf("history");
  if (request.kind === "trajectory" && historyIndex >= 0) {
    const invalid = body.findIndex((row) => !isBinaryHistory(row[historyIndex]?.trim() ?? ""));
    if (invalid >= 0) errors.push(`History in data row ${invalid + 1} must contain only 0 and 1.`);
  }

  const preview: CsvPreviewRow[] = body.slice(0, 5).map((row) =>
    Object.fromEntries(columns.map((column, index) => [column, row[index] ?? ""])),
  );

  return {
    valid: errors.length === 0,
    kind: request.kind,
    filename: request.filename,
    columns,
    rowCount: body.length,
    conditionCount: conditions.length,
    conditions,
    donorAware: columns.includes("donor_id"),
    preview,
    warnings,
    errors,
  };
}
