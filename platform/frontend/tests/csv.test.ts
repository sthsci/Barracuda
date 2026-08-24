import { describe, expect, it } from "vitest";
import { parseCsv, validateCsvLocally } from "@/lib/api";

describe("CSV contract", () => {
  it("parses quoted ordered histories without splitting them", () => {
    expect(parseCsv('cell_id,history\ncell_1,"0,1,0"\n')).toEqual([
      ["cell_id", "history"],
      ["cell_1", "0,1,0"],
    ]);
  });

  it("recognises event-count conditions and donor-aware input", () => {
    const result = validateCsvLocally({
      kind: "event-counts",
      filename: "counts.csv",
      content: "cell_id,donor_id,condition,count\na,d1,Control,0\nb,d2,Treatment,3\n",
    });
    expect(result.valid).toBe(true);
    expect(result.donorAware).toBe(true);
    expect(result.conditions).toEqual(["Control", "Treatment"]);
    expect(result.rowCount).toBe(2);
  });

  it("reports schema and value errors before analysis creation", () => {
    const result = validateCsvLocally({
      kind: "trajectory",
      filename: "bad.csv",
      content: "cell_id,history\na,0102\n",
    });
    expect(result.valid).toBe(false);
    expect(result.errors).toContain("History in data row 1 must contain only 0 and 1.");
  });
});
