"""Audit saved BEIR ranks against trec_eval, without running the models.

Install pytrec-eval-terrier==0.5.11 in a separate environment. Run from the
repository root after `npm run prepare --prefix benchmarks`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import pytrec_eval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("evaluations/results/scifact-v2.json"))
    parser.add_argument("--qrels", type=Path, default=Path("data/scifact/qrels/test.tsv"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report_bytes = args.report.read_bytes()
    report = json.loads(report_bytes)
    qrels_bytes = args.qrels.read_bytes()
    qrels_hash = hashlib.sha256(qrels_bytes).hexdigest()
    expected_hash = report["protocol"]["dataset"]["files"]["qrels/test.tsv"]
    if qrels_hash != expected_hash:
        raise ValueError("Qrels checksum does not match the pinned dataset")
    qrels: dict[str, dict[str, int]] = {}
    for row in csv.DictReader(qrels_bytes.decode().splitlines(), delimiter="\t"):
        qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    queries = report["queries"]
    if len(queries) != len(qrels) or {q["id"] for q in queries} != set(qrels):
        raise ValueError("Expected every official test query exactly once")
    for query in queries:
        if query["judgments"] != qrels[query["id"]]:
            raise ValueError(f"Judgment mismatch for query {query['id']}")
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"P.1", "ndcg_cut.10", "recall.10,100"})
    mrr_evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"})
    names = {
        "P_1": "hit_rate@1",
        "ndcg_cut_10": "ndcg@10",
        "recall_10": "recall@10",
        "recall_100": "recall@100",
        "recip_rank": "mrr@10",
    }
    max_error = 0.0
    comparisons = 0
    for mode in report["protocol"]["modes"]:
        run: dict[str, dict[str, float]] = {}
        for query in queries:
            ids = [doc_id for doc_id, _ in query["modes"][mode]["ranked"]]
            if report["protocol"]["dataset"].get("excludeSelf") and query["id"] in ids:
                raise ValueError(f"Self match for {query['id']}/{mode}")
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate ranked document for {query['id']}/{mode}")
            # Tuple order is the saved ranking. Reranked logits and the RRF tail
            # are different score scales, so never re-sort them by raw score.
            run[query["id"]] = {doc_id: float(len(ids) - i) for i, doc_id in enumerate(ids)}
        scores = evaluator.evaluate(run)
        top_ten = {qid: dict(list(ranks.items())[:10]) for qid, ranks in run.items()}
        mrr = mrr_evaluator.evaluate(top_ten)
        for query in queries:
            measured = scores[query["id"]] | mrr[query["id"]]
            for trec_name, saved_name in names.items():
                actual = measured[trec_name]
                expected = query["modes"][mode]["metrics"][saved_name]
                error = abs(actual - expected)
                if not math.isfinite(expected) or error > 1e-12:
                    raise ValueError(f"Metric mismatch: {query['id']}/{mode}/{saved_name}")
                max_error = max(max_error, error)
                comparisons += 1
    result = {
        "status": "passed",
        "tool": "pytrec-eval-terrier",
        "version": importlib.metadata.version("pytrec-eval-terrier"),
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "test_qrels_sha256": qrels_hash,
        "queries": len(queries),
        "modes": report["protocol"]["modes"],
        "metrics": list(names.values()),
        "comparisons": comparisons,
        "max_absolute_error": max_error,
        "tolerance": 1e-12,
        "scope": "Per-query metric values; excludes bootstrap intervals and model inference.",
    }
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
