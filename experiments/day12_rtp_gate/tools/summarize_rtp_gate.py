#!/usr/bin/env python
"""Summarize RTP-Gate RTD scores against existing retention evidence."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


COMPONENTS = [
    "normalized_delta_nll",
    "topk_union_jsd",
    "late_jsd_spike",
    "late_entropy_spike",
    "entropy_monotonicity_violation",
    "math_token_delta_nll",
    "answer_token_delta_nll",
    "overconfidence_collapse",
    "rtd",
]
BASELINE_SCORE_FIELDS = ["layer_index", "reverse_tail_priority", "bi_prune_priority"]
BI_ORDER = [23, 24, 2, 20, 21, 11, 18]


def fnum(value):
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_layers(raw) -> tuple[int, ...]:
    if raw is None:
        return tuple()
    text = str(raw).strip()
    if not text:
        return tuple()
    return tuple(sorted(int(x.strip()) for x in text.split(",") if x.strip()))


def load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_file"] = str(path)
        return data
    except Exception as exc:
        return {"status": "bad_json", "error": repr(exc), "_file": str(path)}


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg
        i = j
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return pearson(rankdata(xs), rankdata(ys))


def auroc(scores: list[float], labels: list[int]) -> float | None:
    positives = [(s, y) for s, y in zip(scores, labels) if y == 1]
    negatives = [(s, y) for s, y in zip(scores, labels) if y == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0
    for ps, _ in positives:
        for ns, _ in negatives:
            total += 1
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / total if total else None


def precision_at_k(scores: list[float], labels: list[int], k: int) -> float | None:
    if not scores:
        return None
    k = min(k, len(scores))
    ordered = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)[:k]
    return sum(label for _, label in ordered) / k if k else None


def layer_from_run_id(run_id: str):
    marker = "layer_"
    if marker in run_id:
        try:
            return int(run_id.split(marker)[-1].split("_")[0])
        except ValueError:
            return None
    if "drop_" in run_id:
        tail = run_id.split("drop_")[-1]
        layers = parse_layers(tail.replace("_", ","))
        if len(layers) == 1:
            return layers[0]
    return None


def build_retention_maps(root: Path) -> dict[tuple[int, ...], dict]:
    out: dict[tuple[int, ...], dict] = {}
    saved = read_csv(root / "reports" / "day11_remaining_small_model_completion" / "saved_layer_full_sweep.csv")
    for row in saved:
        if row.get("task") not in ("gsm8k", "xsum") or row.get("status") != "done":
            continue
        layer = layer_from_run_id(row.get("run_id", ""))
        if layer is None:
            continue
        key = (layer,)
        out.setdefault(key, {"layers": ",".join(str(x) for x in key)})
        out[key][f"{row['task']}_retention"] = fnum(row.get("retention"))
        out[key][f"{row['task']}_score"] = fnum(row.get("score"))

    ratio = read_csv(root / "reports" / "day8_small_model_gap_completion" / "pruning_ratio_sweep.csv")
    for row in ratio:
        layers = parse_layers(row.get("runtime_skip_layers"))
        if not layers or row.get("task") not in ("gsm8k", "xsum") or row.get("status") != "done":
            continue
        out.setdefault(layers, {"layers": ",".join(str(x) for x in layers)})
        out[layers][f"{row['task']}_retention"] = fnum(row.get("retention"))
        out[layers][f"{row['task']}_score"] = fnum(row.get("score"))
        out[layers]["known_route"] = row.get("run_id")

    iterative_plan = read_csv(root / "reports" / "day11_remaining_small_model_completion" / "iterative_pruning_plan.csv")
    iterative_layers = {row.get("label"): parse_layers(row.get("layers")) for row in iterative_plan}
    iterative = read_csv(root / "reports" / "day11_remaining_small_model_completion" / "iterative_pruning_mini.csv")
    for row in iterative:
        run_id = row.get("run_id")
        layers = iterative_layers.get(run_id)
        if not layers or row.get("task") not in ("gsm8k", "xsum") or row.get("status") != "done":
            continue
        out.setdefault(layers, {"layers": ",".join(str(x) for x in layers)})
        out[layers][f"{row['task']}_retention"] = fnum(row.get("retention"))
        out[layers][f"{row['task']}_score"] = fnum(row.get("score"))
        out[layers]["known_route"] = run_id
    return out


def collect_rtd_rows(score_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(score_dir.glob("*.json")):
        data = load_json(path)
        if data.get("status") != "done":
            continue
        layers = tuple(data.get("runtime_skip_layers") or [])
        overall = data.get("overall") or {}
        by_partition = data.get("by_partition") or {}
        row = {
            "candidate_name": data.get("candidate_name", path.stem),
            "layers": ",".join(str(x) for x in layers),
            "k": len(layers),
            "trace_count": data.get("trace_count"),
            "top_k": data.get("top_k"),
            "score_file": str(path),
        }
        for field in COMPONENTS:
            row[field] = overall.get(field)
        for partition, prefix in (("smoke", "smoke"), ("calibration", "calibration"), ("holdout", "holdout")):
            part = by_partition.get(partition) or {}
            row[f"{prefix}_rtd"] = part.get("rtd")
        rows.append(row)
    return rows


def enrich_with_retention(rows: list[dict], retention: dict[tuple[int, ...], dict]) -> None:
    for row in rows:
        layers = parse_layers(row.get("layers"))
        info = retention.get(layers, {})
        row["known_route"] = info.get("known_route", "")
        row["gsm8k_retention"] = info.get("gsm8k_retention")
        row["xsum_retention"] = info.get("xsum_retention")
        risk = None
        if info.get("gsm8k_retention") is not None:
            risk = 1 if float(info["gsm8k_retention"]) < 0.60 else 0
        row["risky_gsm8k_lt_0_60"] = risk
        if len(layers) == 1:
            layer = layers[0]
            row["layer_index"] = layer
            row["reverse_tail_priority"] = layer
            row["bi_prune_priority"] = (len(BI_ORDER) - BI_ORDER.index(layer)) if layer in BI_ORDER else 0
        else:
            row["layer_index"] = ""
            row["reverse_tail_priority"] = ""
            row["bi_prune_priority"] = ""


def metrics_for(rows: list[dict], score_field: str, target_field: str = "gsm8k_retention") -> dict:
    pairs = []
    for row in rows:
        score = fnum(row.get(score_field))
        target = fnum(row.get(target_field))
        if score is not None and target is not None:
            pairs.append((score, target))
    if len(pairs) < 3:
        return {"n": len(pairs)}
    scores = [p[0] for p in pairs]
    targets = [p[1] for p in pairs]
    labels = [1 if y < 0.60 else 0 for y in targets]
    return {
        "n": len(pairs),
        "spearman_vs_retention": spearman(scores, targets),
        "spearman_abs": abs(spearman(scores, targets) or 0.0),
        "auroc_risky": auroc(scores, labels),
        "precision_at_5": precision_at_k(scores, labels, 5),
        "precision_at_10": precision_at_k(scores, labels, 10),
    }


def calibrated_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    single = [r for r in rows if int(r.get("k") or 0) == 1 and fnum(r.get("gsm8k_retention")) is not None]
    fields = [f for f in COMPONENTS if f != "rtd"]
    result_rows = []
    try:
        import numpy as np
        from sklearn.linear_model import LassoCV, LogisticRegression
        from sklearn.model_selection import LeaveOneOut
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        return [], {"status": "skipped", "reason": f"sklearn unavailable: {exc!r}"}
    if len(single) < 8:
        return [], {"status": "skipped", "reason": "not enough single-layer rows"}
    x = np.array([[float(r.get(f) or 0.0) for f in fields] for r in single], dtype=float)
    y_ret = np.array([float(r["gsm8k_retention"]) for r in single], dtype=float)
    y_risk = np.array([1 if y < 0.60 else 0 for y in y_ret], dtype=int)
    loo = LeaveOneOut()
    pred_ret = []
    pred_risk = []
    for train, test in loo.split(x):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train])
        x_test = scaler.transform(x[test])
        lasso = LassoCV(cv=min(5, len(train)), random_state=1234).fit(x_train, y_ret[train])
        pred_ret.append(float(lasso.predict(x_test)[0]))
        if len(set(y_risk[train])) > 1:
            clf = LogisticRegression(penalty="l1", solver="liblinear", random_state=1234).fit(x_train, y_risk[train])
            pred_risk.append(float(clf.predict_proba(x_test)[0, 1]))
        else:
            pred_risk.append(float(y_risk[train][0]))
    for row, yhat, phat in zip(single, pred_ret, pred_risk):
        result_rows.append(
            {
                "candidate_name": row["candidate_name"],
                "layers": row["layers"],
                "gsm8k_retention": row["gsm8k_retention"],
                "predicted_retention_loo_lasso": yhat,
                "predicted_risk_loo_logistic": phat,
            }
        )
    summary = {
        "status": "done",
        "n": len(single),
        "lasso_spearman": spearman(pred_ret, y_ret.tolist()),
        "logistic_auroc": auroc(pred_risk, y_risk.tolist()),
        "feature_fields": fields,
    }
    return result_rows, summary


def write_status(report_dir: Path, summary: dict, component_rows: list[dict], multi_rows: list[dict]) -> None:
    rtd_metric = next((r for r in component_rows if r.get("score_field") == "rtd"), {})
    lines = [
        "# Day12 RTP-Gate Status",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Key Metrics",
        "",
        f"- RTD single-layer Spearman abs: {rtd_metric.get('spearman_abs', '')}",
        f"- RTD risky AUROC: {rtd_metric.get('auroc_risky', '')}",
        f"- scored candidates: {summary.get('rtd_rows')}",
        f"- scored single-layer candidates with retention: {summary.get('single_layer_with_retention')}",
        f"- scored multi-layer candidates with retention: {len(multi_rows)}",
        "",
        "## Outputs",
        "",
        "- `rtd_scores.csv`",
        "- `single_layer_component_metrics.csv`",
        "- `top_risky_layers_by_rtd.csv`",
        "- `multi_layer_comparison.csv`",
        "- `rtd_calibrated_leave_one_out.csv`",
        "- `day12_rtp_gate_summary.json`",
        "",
        "## Caveats",
        "",
        "- Fixed-weight RTD is the primary result; calibrated scores are exploratory.",
        "- XSUM/classification are controls, not evidence that reasoning is healthy.",
        "- Runtime-skip vs saved-model consistency should be checked before final claims.",
    ]
    (report_dir / "day12_rtp_gate_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/cike/hs/paper2_layer_pruning")
    parser.add_argument("--score-dir", default=None)
    parser.add_argument("--report-dir", default=None)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    score_dir = Path(args.score_dir).resolve() if args.score_dir else root / "results" / "day12_rtp_gate" / "rtd_scores"
    report_dir = Path(args.report_dir).resolve() if args.report_dir else root / "reports" / "day12_rtp_gate"
    report_dir.mkdir(parents=True, exist_ok=True)

    rtd_rows = collect_rtd_rows(score_dir)
    retention = build_retention_maps(root)
    enrich_with_retention(rtd_rows, retention)
    single = [r for r in rtd_rows if int(r.get("k") or 0) == 1]
    multi = [r for r in rtd_rows if int(r.get("k") or 0) > 1 and fnum(r.get("gsm8k_retention")) is not None]

    component_rows = []
    for field in BASELINE_SCORE_FIELDS + COMPONENTS:
        metric = metrics_for(single, field)
        component_rows.append({"score_field": field, **metric})
    top_risky = sorted(single, key=lambda r: fnum(r.get("rtd")) if fnum(r.get("rtd")) is not None else -1, reverse=True)
    calibrated, calibration_summary = calibrated_rows(rtd_rows)

    score_fields = [
        "candidate_name",
        "layers",
        "k",
        "known_route",
        "trace_count",
        "top_k",
        *BASELINE_SCORE_FIELDS,
        *COMPONENTS,
        "smoke_rtd",
        "calibration_rtd",
        "holdout_rtd",
        "gsm8k_retention",
        "xsum_retention",
        "risky_gsm8k_lt_0_60",
        "score_file",
    ]
    write_csv(report_dir / "rtd_scores.csv", rtd_rows, score_fields)
    write_csv(
        report_dir / "single_layer_component_metrics.csv",
        component_rows,
        ["score_field", "n", "spearman_vs_retention", "spearman_abs", "auroc_risky", "precision_at_5", "precision_at_10"],
    )
    write_csv(report_dir / "top_risky_layers_by_rtd.csv", top_risky[:15], score_fields)
    write_csv(report_dir / "multi_layer_comparison.csv", sorted(multi, key=lambda r: fnum(r.get("rtd")) or 0.0), score_fields)
    write_csv(
        report_dir / "rtd_calibrated_leave_one_out.csv",
        calibrated,
        ["candidate_name", "layers", "gsm8k_retention", "predicted_retention_loo_lasso", "predicted_risk_loo_logistic"],
    )

    summary = {
        "status": "done",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rtd_rows": len(rtd_rows),
        "status_counts": dict(Counter("done" for _ in rtd_rows)),
        "single_layer_with_retention": sum(1 for r in single if fnum(r.get("gsm8k_retention")) is not None),
        "component_metrics": component_rows,
        "calibration_summary": calibration_summary,
        "outputs": {
            "rtd_scores": str(report_dir / "rtd_scores.csv"),
            "single_layer_component_metrics": str(report_dir / "single_layer_component_metrics.csv"),
            "top_risky_layers_by_rtd": str(report_dir / "top_risky_layers_by_rtd.csv"),
            "multi_layer_comparison": str(report_dir / "multi_layer_comparison.csv"),
            "rtd_calibrated_leave_one_out": str(report_dir / "rtd_calibrated_leave_one_out.csv"),
        },
    }
    (report_dir / "day12_rtp_gate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_status(report_dir, summary, component_rows, multi)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
