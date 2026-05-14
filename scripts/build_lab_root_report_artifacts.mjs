import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const __filename = fileURLToPath(import.meta.url);
const repoRoot = path.resolve(path.dirname(__filename), "..");
const artifactDir = path.join(repoRoot, "artifacts", "lab_root_2026_05_14");
const reportDir = path.join(repoRoot, "reports", "day12_rtp_gate");
const summaryPath = path.join(artifactDir, "lab_root_summary.json");
const xlsxPath = path.join(reportDir, "lab_root_rtp_gate_final_results_2026_05_14.xlsx");
const mdPath = path.join(reportDir, "lab_root_rtp_gate_final_report_2026_05_14.md");

const summary = JSON.parse((await fs.readFile(summaryPath, "utf8")).replace(/^\uFEFF/, ""));

function pct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "";
  return Number(value);
}

function round(value, digits = 4) {
  if (value === "" || value === null || value === undefined || Number.isNaN(Number(value))) return "";
  return Number(Number(value).toFixed(digits));
}

function layerCsv(layers) {
  return Array.isArray(layers) ? layers.join(",") : "";
}

function findGen(runId, task) {
  return summary.generation.find((row) => row.run_id === runId && row.task === task);
}

function rtdForLayers(layers) {
  const key = JSON.stringify([...layers].sort((a, b) => a - b));
  const rows = summary.rtd_scores.filter((row) => JSON.stringify([...(row.runtime_skip_layers || [])].sort((a, b) => a - b)) === key);
  if (!rows.length) return null;
  const preferred = rows.find((row) => row.candidate_name === "rtp_gate_structure_drop_" + layers.join("_"));
  return preferred || rows[0];
}

function classRowsFor(runId) {
  return summary.classification_fixed.filter((row) => row.run_id === runId && row.status === "done");
}

function avg(rows, field = "score") {
  const values = rows.map((row) => Number(row[field])).filter((value) => Number.isFinite(value));
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
}

function denseScoreByTask(task) {
  const row = summary.classification_fixed.find((item) => item.run_id === "dense_base" && item.task === task);
  return row ? Number(row.score) : null;
}

const candidateLayers = new Map();
candidateLayers.set("dense_base", []);
candidateLayers.set("reverse_3", [23, 24, 25]);
candidateLayers.set("iterative_proxy_k3", [1, 24, 25]);
candidateLayers.set("rtp_gate_pure_k3", [1, 9, 24]);
candidateLayers.set("rtp_gate_structure_k3", [1, 9, 24]);
candidateLayers.set("risky_k5", [21, 22, 23, 24, 25]);

const candidateLabels = {
  dense_base: "Dense baseline",
  reverse_3: "Tail pruning baseline",
  iterative_proxy_k3: "Existing iterative proxy",
  rtp_gate_pure_k3: "RTP-Gate pure selection",
  rtp_gate_structure_k3: "RTP-Gate structure-aware selection",
  risky_k5: "Highest-risk k=5 control",
};

const candidateOrder = [
  "dense_base",
  "iterative_proxy_k3",
  "rtp_gate_pure_k3",
  "rtp_gate_structure_k3",
  "reverse_3",
  "risky_k5",
];

const denseGsm8k = Number(findGen("dense_base", "gsm8k")?.score ?? 0);
const denseXsum = Number(findGen("dense_base", "xsum")?.score ?? 0);
const denseClassAvg = avg(classRowsFor("dense_base"));

const candidateSummary = candidateOrder.map((runId) => {
  const layers = candidateLayers.get(runId) || [];
  const gsm8k = findGen(runId, "gsm8k");
  const xsum = findGen(runId, "xsum");
  const classRows = classRowsFor(runId);
  const classAvg = avg(classRows);
  const rtd = layers.length ? rtdForLayers(layers) : null;
  return {
    run_id: runId,
    label: candidateLabels[runId] || runId,
    layers,
    k: layers.length,
    gsm8k_score: pct(gsm8k?.score),
    gsm8k_retention: denseGsm8k ? pct(gsm8k?.score) / denseGsm8k : "",
    xsum_score: pct(xsum?.score),
    xsum_retention: denseXsum ? pct(xsum?.score) / denseXsum : "",
    classification_avg: classAvg,
    classification_retention: denseClassAvg ? classAvg / denseClassAvg : "",
    calibration_rtd: rtd?.calibration_rtd ?? "",
    holdout_rtd: rtd?.holdout_rtd ?? "",
    overall_rtd: rtd?.overall_rtd ?? "",
    note:
      runId === "risky_k5"
        ? "Danger control selected by highest calibration RTD among k=5 candidates."
        : runId.includes("rtp_gate")
          ? "Formal raw-eval k=3 RTP-Gate candidate."
          : "Comparison baseline.",
  };
});

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

function colName(n) {
  let s = "";
  while (n > 0) {
    const m = (n - 1) % 26;
    s = String.fromCharCode(65 + m) + s;
    n = Math.floor((n - 1) / 26);
  }
  return s;
}

await fs.mkdir(reportDir, { recursive: true });
const workbook = Workbook.create();

const summaryRows = [
  ["Snapshot", summary.snapshot.snapshot_time_beijing],
  ["Server", summary.snapshot.server],
  ["Remote root", summary.snapshot.root],
  ["Trace counts", `smoke=${summary.trace_manifest.counts.smoke}, calibration=${summary.trace_manifest.counts.calibration}, holdout=${summary.trace_manifest.counts.holdout}`],
  ["Formal raw eval scope", "Dense, reverse_3, iterative_proxy_k3, RTP-Gate pure/structure k3, risky_k5"],
  ["Important scope note", "Only k=3 RTP-Gate candidates have full raw GSM8K/XSUM/classification in this snapshot; k2/k5 are selected/scored by RTD but not raw-evaluated."],
  ["Classification note", "Old classification outputs are invalid because runtime skip was not applied; this workbook uses classification_runtime_skip_fixed_20260514."],
  ["Main take-away", "RTP-Gate k3 avoids the catastrophic collapse seen in reverse_3/risky_k5, but does not retain dense GSM8K performance."],
];
addSheet(workbook, "Summary", ["Field", "Value"], summaryRows, { widths: { "A:A": 180, "B:B": 760 } });

addSheet(
  workbook,
  "Candidate_Summary",
  [
    "run_id",
    "label",
    "layers",
    "k",
    "GSM8K score",
    "GSM8K retention",
    "XSUM score",
    "XSUM retention",
    "Classification avg",
    "Classification retention",
    "Calibration RTD",
    "Holdout RTD",
    "Overall RTD",
    "note",
  ],
  candidateSummary.map((row) => [
    row.run_id,
    row.label,
    layerCsv(row.layers),
    row.k,
    round(row.gsm8k_score, 4),
    round(row.gsm8k_retention, 4),
    round(row.xsum_score, 4),
    round(row.xsum_retention, 4),
    round(row.classification_avg, 4),
    round(row.classification_retention, 4),
    round(row.calibration_rtd, 4),
    round(row.holdout_rtd, 4),
    round(row.overall_rtd, 4),
    row.note,
  ]),
  {
    numberFormats: {
      "E2:M100": "0.0000",
    },
    widths: { "A:A": 170, "B:B": 210, "C:C": 120, "N:N": 430 },
  },
);

addSheet(
  workbook,
  "Raw_Generation",
  ["run_id", "task", "samples", "score", "dense_score", "retention_vs_dense", "status"],
  summary.generation.map((row) => {
    const dense = Number(findGen("dense_base", row.task)?.score ?? 0);
    return [row.run_id, row.task, row.samples, round(row.score, 4), round(dense, 4), dense ? round(Number(row.score) / dense, 4) : "", row.status];
  }),
  { numberFormats: { "D2:F100": "0.0000" }, widths: { "A:A": 190, "B:B": 90 } },
);

addSheet(
  workbook,
  "Classification_Fixed",
  ["run_id", "task", "samples", "score", "dense_score", "retention_vs_dense", "skip_layers", "runtime_skip_enabled", "status"],
  summary.classification_fixed.map((row) => {
    const dense = denseScoreByTask(row.task);
    return [
      row.run_id,
      row.task,
      row.samples,
      round(row.score, 4),
      round(dense, 4),
      dense ? round(Number(row.score) / dense, 4) : "",
      layerCsv(row.skip_layers),
      row.runtime_skip_enabled,
      row.status,
    ];
  }),
  { numberFormats: { "D2:F200": "0.0000" }, widths: { "A:A": 190, "B:B": 130, "G:G": 120 } },
);

addSheet(
  workbook,
  "RTD_Scores",
  ["candidate_name", "layers", "k", "trace_count", "overall_rtd", "calibration_rtd", "holdout_rtd", "status", "file"],
  summary.rtd_scores
    .filter((row) => row.status === "done")
    .sort((a, b) => (a.k - b.k) || String(a.candidate_name).localeCompare(String(b.candidate_name)))
    .map((row) => [
      row.candidate_name,
      layerCsv(row.runtime_skip_layers),
      row.k,
      row.trace_count,
      round(row.overall_rtd, 4),
      round(row.calibration_rtd, 4),
      round(row.holdout_rtd, 4),
      row.status,
      row.file,
    ]),
  { numberFormats: { "E2:G1000": "0.0000" }, widths: { "A:A": 300, "B:B": 160, "I:I": 300 } },
);

const selectionRows = [];
for (const [file, rows] of Object.entries(summary.selection)) {
  const family = file.includes("structure") ? "structure" : "pure";
  for (const row of rows) {
    selectionRows.push([
      family,
      row.label,
      Number(row.k),
      row.layers,
      row.ordered_layers,
      round(row.best_rtd, 4),
      round(row.best_selection_score, 4),
      row.best_candidate_layer,
    ]);
  }
}
addSheet(
  workbook,
  "RTP_Gate_Selection",
  ["family", "label", "k", "layers", "ordered_layers", "best_rtd", "best_selection_score", "best_candidate_layer"],
  selectionRows,
  { numberFormats: { "F2:G100": "0.0000" }, widths: { "B:B": 220, "D:E": 160 } },
);

addSheet(
  workbook,
  "Saved_Consistency",
  ["candidate", "trace_count", "overall_rtd", "calibration_rtd", "holdout_rtd", "status", "file"],
  summary.consistency.map((row) => [
    row.candidate_name,
    row.trace_count,
    round(row.overall_rtd, 4),
    round(row.calibration_rtd, 4),
    round(row.holdout_rtd, 4),
    row.status,
    row.file,
  ]),
  { numberFormats: { "C2:E50": "0.0000" }, widths: { "A:A": 260, "G:G": 270 } },
);

addSheet(
  workbook,
  "Old_Classification_Audit",
  ["file", "run_id", "task", "score", "runtime_skip_raw", "formal_use"],
  summary.classification_old.map((row) => [row.file, row.run_id, row.task, round(row.score, 4), JSON.stringify(row.runtime_skip), "INVALID - runtime skip wrapper bug"]),
  { numberFormats: { "D2:D200": "0.0000" }, widths: { "A:A": 320, "E:F": 380 } },
);

const notesRows = [
  ["RTD interpretation", "RTD is a pruning-risk diagnostic, not a downstream task score."],
  ["RTP-Gate interpretation", "RTP-Gate is a gating/selection method for prioritizing candidates before expensive raw evaluation; this snapshot does not claim global optimal pruning."],
  ["k=3 scope", "The complete raw-evaluated RTP-Gate comparison in this snapshot is k=3. Pure and structure-aware selected the same layers: [1,9,24]."],
  ["k2/k5 status", "RTP-Gate k2/k5 selections exist from RTD, but full raw GSM8K/XSUM/classification curves were not run yet."],
  ["Old classification", summary.classification_old_invalid_note],
  ["Reference correlations", `Day8 runtime-skip reference approx Spearman abs=${summary.correlation_reference.day8_runtime_skip_reference.spearman_abs_approx}, AUROC=${summary.correlation_reference.day8_runtime_skip_reference.auroc_approx}; Day11 saved-model reference approx Spearman abs=${summary.correlation_reference.day11_saved_model_reference.spearman_abs_approx}, AUROC=${summary.correlation_reference.day11_saved_model_reference.auroc_approx}.`],
];
addSheet(workbook, "Notes", ["Topic", "Note"], notesRows, { widths: { "A:A": 210, "B:B": 850 } });

const inspect = await workbook.inspect({
  kind: "table",
  range: "Candidate_Summary!A1:N8",
  include: "values",
  tableMaxRows: 10,
  tableMaxCols: 14,
});
console.log(inspect.ndjson.split("\n").slice(0, 5).join("\n"));

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "formula error scan",
});
console.log(errors.ndjson);

for (const sheetName of ["Summary", "Candidate_Summary", "Raw_Generation", "Classification_Fixed", "RTD_Scores"]) {
  await workbook.render({ sheetName, range: "A1:H12", scale: 1 });
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(xlsxPath);

const classAvgLines = candidateSummary
  .map((row) => `| ${row.run_id} | ${layerCsv(row.layers) || "-"} | ${round(row.gsm8k_score, 3)} | ${round(row.gsm8k_retention, 3)} | ${round(row.xsum_score, 3)} | ${round(row.classification_avg, 3)} | ${round(row.calibration_rtd, 3)} |`)
  .join("\n");

const md = `# lab-root RTP-Gate/RTD final snapshot report

Snapshot time: ${summary.snapshot.snapshot_time_beijing} Beijing time  
Remote root: \`${summary.snapshot.root}\`  
Workbook: \`reports/day12_rtp_gate/${path.basename(xlsxPath)}\`

## One-line conclusion

This lab-root rerun supports the RTP-Gate/RTD idea as a pruning-risk diagnostic: RTP-Gate k=3 does not recover dense performance, but it avoids the catastrophic GSM8K collapse seen in naive tail pruning and in the deliberately risky k=5 control.

## What has been run

- Dense traces: \`smoke=${summary.trace_manifest.counts.smoke}\`, \`calibration=${summary.trace_manifest.counts.calibration}\`, \`holdout=${summary.trace_manifest.counts.holdout}\`.
- Single-layer RTD: 26/26 layers completed.
- Multi-layer RTD and selection: known baselines plus RTP-Gate pure/structure selections completed.
- Raw evaluation: dense, \`reverse_3\`, \`iterative_proxy_k3\`, \`rtp_gate_pure_k3\`, \`rtp_gate_structure_k3\`, and \`risky_k5\`.
- Saved-model consistency: \`saved_single_layer_24\` and \`saved_rtp_gate_structure_k3\` completed.
- Classification controls were rerun after fixing the runtime-skip wrapper bug; the fixed outputs are the formal classification results.

## Scope clarification

The current full raw-eval comparison is mainly a **three-layer pruning** comparison. RTP-Gate selected k=2/k=3/k=5 candidates at the RTD stage, but the expensive downstream raw evaluations in this snapshot were run for k=3 RTP-Gate candidates plus the k=5 risky control. Therefore, this is not yet a full k2/k3/k5 raw-eval curve.

## Main raw-eval table

| candidate | layers | GSM8K | GSM8K retention | XSUM | classification avg | calibration RTD |
|---|---:|---:|---:|---:|---:|---:|
${classAvgLines}

## Interpretation

Dense remains the upper baseline at GSM8K 0.624. The RTP-Gate k=3 candidates score 0.332 on GSM8K, so they lose substantial reasoning ability relative to dense. But the comparison that matters for the gate is the failure mode: \`reverse_3\` falls to 0.006 and \`risky_k5\` falls to 0.000. In that sense, RTP-Gate is doing the intended diagnostic job: it avoids layer sets that RTD flags as extremely risky.

\`rtp_gate_pure_k3\` and \`rtp_gate_structure_k3\` are identical here because both selected the same layers, \`[1,9,24]\`. Their raw results therefore match exactly.

The fixed classification controls no longer show the impossible all-candidates-equal pattern. Average classification scores fall from dense 0.624 to about 0.560 for RTP-Gate k=3, 0.544 for reverse_3, 0.534 for iterative_proxy_k3, and 0.514 for risky_k5. These controls suggest broad language capability degrades but does not collapse in the same way as GSM8K reasoning.

## RTD and selection notes

- RTP-Gate pure k2: \`[1,24]\`
- RTP-Gate pure k3: \`[1,9,24]\`
- RTP-Gate pure k5: \`[1,9,10,19,24]\`
- Structure-aware selected the same layer sets for k2/k3/k5 in this run.
- Saved-model consistency produced RTD 4.738 for \`saved_single_layer_24\` and 19.776 for \`saved_rtp_gate_structure_k3\`.

## Correct use of the results

RTD should be described as a risk diagnostic or gate, not as a downstream benchmark score. RTP-Gate should be described as a candidate-selection method for reducing expensive raw evaluations, not as a proof of globally optimal pruning.

The old classification directory is retained only for audit purposes. Those outputs are invalid for formal analysis because the wrapper did not actually apply runtime layer skipping before the fix.

## Recommended next experiments

1. Run raw GSM8K/XSUM/classification for RTP-Gate k2 and k5 to produce a real k-curve.
2. Repeat trace construction with another seed or calibration/holdout split to test selection stability.
3. Add an explicit classification evaluator regression test so runtime skip cannot silently fail again.
`;

await fs.writeFile(mdPath, md, "utf8");
console.log(JSON.stringify({ xlsxPath, mdPath }, null, 2));
