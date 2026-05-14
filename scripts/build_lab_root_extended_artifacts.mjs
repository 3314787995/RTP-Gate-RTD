import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const __filename = fileURLToPath(import.meta.url);
const repoRoot = path.resolve(path.dirname(__filename), "..");
const artifactDir = path.join(repoRoot, "artifacts", "lab_root_2026_05_14_extended");
const baseArtifactDir = path.join(repoRoot, "artifacts", "lab_root_2026_05_14");
const reportDir = path.join(repoRoot, "reports", "day12_rtp_gate");
const summaryPath = path.join(artifactDir, "lab_root_extended_summary.json");
const baseSummaryPath = path.join(baseArtifactDir, "lab_root_summary.json");

const kCurveCsvPath = path.join(reportDir, "gsm8k_k_curve_2026_05_14.csv");
const kCurveMdPath = path.join(reportDir, "gsm8k_k_curve_2026_05_14.md");
const extendedMdPath = path.join(reportDir, "lab_root_rtp_gate_extended_report_2026_05_14.md");
const extendedXlsxPath = path.join(reportDir, "lab_root_rtp_gate_extended_results_2026_05_14.xlsx");

const summary = JSON.parse((await fs.readFile(summaryPath, "utf8")).replace(/^\uFEFF/, ""));
const baseSummary = summary.base_summary || JSON.parse((await fs.readFile(baseSummaryPath, "utf8")).replace(/^\uFEFF/, ""));

function asNumber(value) {
  if (value === "" || value === null || value === undefined) return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function round(value, digits = 4) {
  const number = asNumber(value);
  return number === null ? "" : Number(number.toFixed(digits));
}

function layerCsv(layers) {
  return Array.isArray(layers) ? layers.join(",") : String(layers || "");
}

function colName(n) {
  let s = "";
  while (n > 0) {
    const m = (n - 1) % 26;
    s = String.fromCharCode(65 + m) + s;
    n = Math.floor((n - 1) / 26);
  }
  return s;
}

function findGeneration(runId, task = "gsm8k") {
  return baseSummary.generation.find((row) => row.run_id === runId && row.task === task);
}

function byRunId(rows) {
  const result = new Map();
  for (const row of rows) result.set(row.run_id, row);
  return result;
}

function familyOrder(family) {
  return { dense: 0, "RTP-Gate": 1, "Iterative proxy": 2, "Reverse tail": 3 }[family] ?? 99;
}

function parseSelectionRows() {
  const rows = [];
  for (const [file, entries] of Object.entries(baseSummary.selection || {})) {
    const family = file.includes("structure") ? "structure-aware" : "pure";
    for (const row of entries) rows.push({ family, ...row });
  }
  return rows;
}

function addSheet(workbook, name, headers, rows, options = {}) {
  const sheet = workbook.worksheets.add(name);
  const data = [headers, ...rows];
  if (data.length) {
    sheet.getRange(`A1:${colName(headers.length)}${data.length}`).values = data;
    const used = sheet.getRange(`A1:${colName(headers.length)}${data.length}`);
    used.format = {
      font: { name: "Calibri", size: 11, color: "tx1" },
      borders: { preset: "outside", style: "thin", color: "#D1D5DB" },
      verticalAlignment: "center",
      wrapText: true,
    };
    const header = sheet.getRange(`A1:${colName(headers.length)}1`);
    header.format = {
      fill: "#1F4E79",
      font: { name: "Calibri", size: 11, color: "#FFFFFF", bold: true },
      horizontalAlignment: "center",
      verticalAlignment: "center",
      wrapText: true,
    };
    used.format.autofitColumns();
    used.format.autofitRows();
  }
  if (options.numberFormats) {
    for (const [range, format] of Object.entries(options.numberFormats)) {
      sheet.getRange(range).format.numberFormat = format;
    }
  }
  if (options.widths) {
    for (const [range, width] of Object.entries(options.widths)) {
      sheet.getRange(range).format.columnWidthPx = width;
    }
  }
  return sheet;
}

async function writeCsv(pathname, headers, rows) {
  const escape = (value) => {
    const text = String(value ?? "");
    if (/[",\n\r]/.test(text)) return `"${text.replaceAll('"', '""')}"`;
    return text;
  };
  const lines = [headers.join(","), ...rows.map((row) => headers.map((header) => escape(row[header])).join(","))];
  await fs.writeFile(pathname, `${lines.join("\n")}\n`, "utf8");
}

await fs.mkdir(reportDir, { recursive: true });

const denseScore = asNumber(findGeneration("dense_base")?.score) ?? 0;
const newByRunId = byRunId(summary.k_curve?.new_results || []);
const kCurveRows = [
  {
    family: "dense",
    candidate: "dense_base",
    k: 0,
    layers: [],
    gsm8k_score: denseScore,
    source: "phase1_raw_eval",
    status: findGeneration("dense_base")?.status || "done",
    file: findGeneration("dense_base")?.file || "",
    note: "Dense baseline from phase-1 raw eval.",
  },
  {
    family: "RTP-Gate",
    candidate: "rtp_gate_k2",
    k: 2,
    layers: [1, 24],
    gsm8k_score: asNumber(newByRunId.get("rtp_gate_k2")?.score),
    source: "phase2_k_curve",
    status: newByRunId.get("rtp_gate_k2")?.status || "",
    file: newByRunId.get("rtp_gate_k2")?.file || "",
    note: "Unique layer set shared by pure and structure-aware k=2 selection.",
  },
  {
    family: "RTP-Gate",
    candidate: "rtp_gate_k3",
    k: 3,
    layers: [1, 9, 24],
    gsm8k_score: asNumber(findGeneration("rtp_gate_structure_k3")?.score),
    source: "phase1_raw_eval",
    status: findGeneration("rtp_gate_structure_k3")?.status || "done",
    file: findGeneration("rtp_gate_structure_k3")?.file || "",
    note: "Phase-1 k=3 result; pure and structure-aware selected the same layers.",
  },
  {
    family: "RTP-Gate",
    candidate: "rtp_gate_k5",
    k: 5,
    layers: [1, 9, 10, 19, 24],
    gsm8k_score: asNumber(newByRunId.get("rtp_gate_k5")?.score),
    source: "phase2_k_curve",
    status: newByRunId.get("rtp_gate_k5")?.status || "",
    file: newByRunId.get("rtp_gate_k5")?.file || "",
    note: "Unique layer set shared by pure and structure-aware k=5 selection.",
  },
  {
    family: "Iterative proxy",
    candidate: "iterative_proxy_k2",
    k: 2,
    layers: [24, 25],
    gsm8k_score: asNumber(newByRunId.get("iterative_proxy_k2")?.score),
    source: "phase2_k_curve",
    status: newByRunId.get("iterative_proxy_k2")?.status || "",
    file: newByRunId.get("iterative_proxy_k2")?.file || "",
    note: "Existing proxy baseline.",
  },
  {
    family: "Iterative proxy",
    candidate: "iterative_proxy_k3",
    k: 3,
    layers: [1, 24, 25],
    gsm8k_score: asNumber(findGeneration("iterative_proxy_k3")?.score),
    source: "phase1_raw_eval",
    status: findGeneration("iterative_proxy_k3")?.status || "done",
    file: findGeneration("iterative_proxy_k3")?.file || "",
    note: "Phase-1 k=3 baseline.",
  },
  {
    family: "Iterative proxy",
    candidate: "iterative_proxy_k5",
    k: 5,
    layers: [1, 21, 22, 24, 25],
    gsm8k_score: asNumber(newByRunId.get("iterative_proxy_k5")?.score),
    source: "phase2_k_curve",
    status: newByRunId.get("iterative_proxy_k5")?.status || "",
    file: newByRunId.get("iterative_proxy_k5")?.file || "",
    note: "Existing proxy baseline.",
  },
  {
    family: "Reverse tail",
    candidate: "reverse_2",
    k: 2,
    layers: [24, 25],
    gsm8k_score: asNumber(newByRunId.get("reverse_2")?.score),
    source: "phase2_k_curve",
    status: newByRunId.get("reverse_2")?.status || "",
    file: newByRunId.get("reverse_2")?.file || "",
    note: "Naive tail-pruning baseline.",
  },
  {
    family: "Reverse tail",
    candidate: "reverse_3",
    k: 3,
    layers: [23, 24, 25],
    gsm8k_score: asNumber(findGeneration("reverse_3")?.score),
    source: "phase1_raw_eval",
    status: findGeneration("reverse_3")?.status || "done",
    file: findGeneration("reverse_3")?.file || "",
    note: "Phase-1 catastrophic tail-pruning baseline.",
  },
  {
    family: "Reverse tail",
    candidate: "reverse_5",
    k: 5,
    layers: [21, 22, 23, 24, 25],
    gsm8k_score: asNumber(newByRunId.get("reverse_5")?.score),
    source: "phase2_k_curve",
    status: newByRunId.get("reverse_5")?.status || "",
    file: newByRunId.get("reverse_5")?.file || "",
    note: "Naive tail-pruning baseline.",
  },
]
  .map((row) => ({
    ...row,
    layers_csv: layerCsv(row.layers),
    retention_vs_dense: denseScore ? asNumber(row.gsm8k_score) / denseScore : null,
  }))
  .sort((a, b) => familyOrder(a.family) - familyOrder(b.family) || a.k - b.k);

await writeCsv(
  kCurveCsvPath,
  ["family", "candidate", "k", "layers_csv", "gsm8k_score", "retention_vs_dense", "source", "status", "note", "file"],
  kCurveRows.map((row) => ({
    ...row,
    gsm8k_score: round(row.gsm8k_score, 6),
    retention_vs_dense: round(row.retention_vs_dense, 6),
  })),
);

const kCurveTable = kCurveRows
  .map((row) => `| ${row.family} | ${row.candidate} | ${row.k} | ${row.layers_csv || "-"} | ${round(row.gsm8k_score, 3)} | ${round(row.retention_vs_dense, 3)} | ${row.source} |`)
  .join("\n");
const scoreFor = (candidate) => kCurveRows.find((row) => row.candidate === candidate)?.gsm8k_score;
const rtpK2 = scoreFor("rtp_gate_k2");
const rtpK3 = scoreFor("rtp_gate_k3");
const rtpK5 = scoreFor("rtp_gate_k5");
const reverseK2 = scoreFor("reverse_2");
const reverseK5 = scoreFor("reverse_5");
const iterativeK5 = scoreFor("iterative_proxy_k5");

const stabilitySummary = summary.stability?.summary?.[0] || {};
const stabilitySelection = summary.stability?.selection_overlap || [];
const stabilityLayerRows = summary.stability?.single_layer_rtd || [];
const selectedSourceIndices = summary.stability?.trace_manifest?.selected_source_indices || [];
const stabilitySelectionTable = stabilitySelection
  .map((row) => `| ${row.family} | ${row.k} | ${row.base_layers} | ${row.seed_layers} | ${round(row.jaccard, 3)} |`)
  .join("\n");

await fs.writeFile(
  kCurveMdPath,
  `# GSM8K k-curve snapshot

Generated: ${summary.snapshot?.snapshot_time_beijing || ""}

This table combines phase-1 k=3 raw-eval results with phase-2 k=2/k=5 GSM8K 500 runs.

| family | candidate | k | layers | GSM8K | retention vs dense | source |
|---|---|---:|---|---:|---:|---|
${kCurveTable}

RTD/RTP-Gate remains a pruning-risk diagnostic. The k-curve is a downstream validation view, not the score used to select layers.
`,
  "utf8",
);

const workbook = Workbook.create();

addSheet(workbook, "Summary", ["Field", "Value"], [
  ["Snapshot", summary.snapshot?.snapshot_time_beijing || ""],
  ["Server", baseSummary.snapshot?.server || "lab-root"],
  ["Remote root", baseSummary.snapshot?.root || "/root/hs/paper2_layer_pruning"],
  ["Phase 1", "k=3 full raw eval completed earlier and preserved in the first final report."],
  ["Phase 2", "GSM8K k=2/k=5 curve plus seed_5678 RTD/selection stability."],
  ["Dense GSM8K", round(denseScore, 4)],
  ["Stability seed", summary.stability?.seed || "5678"],
  ["Trace counts", JSON.stringify(summary.stability?.trace_manifest?.counts || {})],
  ["Shuffle enabled", String(summary.stability?.trace_manifest?.shuffle_correct_items ?? "")],
  ["Selected source indices head", selectedSourceIndices.slice(0, 20).join(",")],
]);

addSheet(
  workbook,
  "GSM8K_K_Curve",
  ["family", "candidate", "k", "layers", "GSM8K score", "retention vs dense", "source", "status", "note"],
  kCurveRows.map((row) => [row.family, row.candidate, row.k, row.layers_csv, round(row.gsm8k_score, 4), round(row.retention_vs_dense, 4), row.source, row.status, row.note]),
  { numberFormats: { "E2:F100": "0.0000" }, widths: { "A:B": 160, "D:D": 160, "I:I": 420 } },
);

addSheet(
  workbook,
  "Stability_Seed_5678",
  ["metric", "value"],
  [
    ["single_layer_count", stabilitySummary.single_layer_count || ""],
    ["single_layer_spearman", round(stabilitySummary.single_layer_spearman, 6)],
    ["base_top5_layers", stabilitySummary.base_top5_layers || ""],
    ["seed_top5_layers", stabilitySummary.seed_top5_layers || ""],
    ["top5_overlap", stabilitySummary.top5_overlap || ""],
  ],
  { numberFormats: { "B2:B20": "0.000000" }, widths: { "A:A": 220, "B:B": 360 } },
);

addSheet(
  workbook,
  "Stability_Selection",
  ["family", "k", "base_layers", "seed_layers", "Jaccard"],
  stabilitySelection.map((row) => [row.family, Number(row.k), row.base_layers, row.seed_layers, round(row.jaccard, 6)]),
  { numberFormats: { "E2:E100": "0.000000" }, widths: { "A:A": 180, "C:D": 180 } },
);

addSheet(
  workbook,
  "Stability_Single_Layer",
  ["layer", "base_calibration_rtd", "seed_calibration_rtd"],
  stabilityLayerRows.map((row) => [Number(row.layer), round(row.base_calibration_rtd, 6), round(row.seed_calibration_rtd, 6)]),
  { numberFormats: { "B2:C100": "0.000000" } },
);

addSheet(
  workbook,
  "RTP_Gate_Selection_Base",
  ["family", "label", "k", "layers", "ordered_layers", "best_rtd", "best_selection_score", "best_candidate_layer"],
  parseSelectionRows().map((row) => [row.family, row.label, Number(row.k), row.layers, row.ordered_layers, round(row.best_rtd, 6), round(row.best_selection_score, 6), row.best_candidate_layer]),
  { numberFormats: { "F2:G100": "0.000000" }, widths: { "B:B": 220, "D:E": 160 } },
);

const errorScan = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "formula error scan",
});
console.log(errorScan.ndjson);

for (const sheetName of ["Summary", "GSM8K_K_Curve", "Stability_Seed_5678", "Stability_Selection"]) {
  await workbook.render({ sheetName, range: "A1:H12", scale: 1 });
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(extendedXlsxPath);

await fs.writeFile(
  extendedMdPath,
  `# lab-root RTP-Gate/RTD extended report

Snapshot time: ${summary.snapshot?.snapshot_time_beijing || ""} Beijing time  
Remote root: \`${baseSummary.snapshot?.root || "/root/hs/paper2_layer_pruning"}\`  
Workbook: \`reports/day12_rtp_gate/${path.basename(extendedXlsxPath)}\`

## What this second phase adds

- GSM8K 500 k-curve points for RTP-Gate, reverse-tail pruning, and iterative-proxy baselines at k=2 and k=5.
- The existing phase-1 k=3 GSM8K results are reused as the middle point of the curve.
- A seed_5678 trace stability run with true shuffled dense-correct source items.

## GSM8K k-curve

| family | candidate | k | layers | GSM8K | retention vs dense | source |
|---|---|---:|---|---:|---:|---|
${kCurveTable}

## k-curve readout

- RTP-Gate is monotonic across the measured strengths: ${round(rtpK2, 3)} at k=2, ${round(rtpK3, 3)} at k=3, and ${round(rtpK5, 3)} at k=5.
- At k=2, RTP-Gate beats the tail/prior proxy layer set by ${round(rtpK2 - reverseK2, 3)} GSM8K absolute points.
- At k=5, all methods are weak; RTP-Gate avoids the complete reverse-tail collapse (${round(reverseK5, 3)}) but is slightly below iterative_proxy_k5 (${round(iterativeK5, 3)}).
- This supports the intended use of RTD as a pruning-risk gate, not a claim that RTP-Gate is globally optimal for every k.

## Stability summary

- Trace seed: ${summary.stability?.seed || "5678"}
- Trace counts: \`${JSON.stringify(summary.stability?.trace_manifest?.counts || {})}\`
- Shuffle enabled: \`${summary.stability?.trace_manifest?.shuffle_correct_items ?? ""}\`
- Single-layer calibration RTD Spearman: ${round(stabilitySummary.single_layer_spearman, 4)}
- Base top-5 risky layers: \`${stabilitySummary.base_top5_layers || ""}\`
- Seed top-5 risky layers: \`${stabilitySummary.seed_top5_layers || ""}\`
- Top-5 risky overlap: ${stabilitySummary.top5_overlap || ""}/5

| family | k | base layers | seed layers | Jaccard |
|---|---:|---|---|---:|
${stabilitySelectionTable}

## Interpretation

This extended phase is meant to strengthen the main claim, not replace the first final report. RTP-Gate/RTD is still framed as a pruning-risk gate: it estimates which layer-removal candidates deserve expensive downstream evaluation. The seed stability check tests whether the RTD ranking and selected layer sets are robust to changing the dense-correct trace sample order.

Here, the RTP-Gate k-curve degrades smoothly as k increases and the seed stability overlap is high. That strengthens the diagnostic story, while still leaving the honest limitation that RTP-Gate is a gate for candidate choice rather than a proof of globally optimal pruning.
`,
  "utf8",
);

console.log(JSON.stringify({ kCurveCsvPath, kCurveMdPath, extendedMdPath, extendedXlsxPath }, null, 2));
