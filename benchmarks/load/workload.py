"""Prepare a deterministic, relevance-independent SciFact workload through the API."""

import csv
import json
import sqlite3
from hashlib import sha256
from pathlib import Path
from time import perf_counter

FILES = {
    "corpus.jsonl": "dec31c8182f3d744c7d2c09423756fd1d17cbef75808db13ba01cc0aab4d1ac6",
    "queries.jsonl": "8ff84a7c903f722981cd8d595c022660140c51867b27608a6d4910db86080313",
    "qrels/train.tsv": "a53f2114831916c096b6c37d9e54da68cef4efdcdbd5ed46533601af972acf1d",
}
LOGICAL = "scifact-load-v1"
SEARCH = dict(collection=LOGICAL, mode="hybrid", candidate_limit=100, result_limit=10)
MODEL = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
REVISION = "f3cb857cba53019a20df283396bcca179cf051a4"


def digest(data):
    return sha256(data).hexdigest()


def load_workload(directory):
    for name, expected in FILES.items():
        if digest((directory / name).read_bytes()) != expected:
            raise ValueError("SciFact input digest mismatch: " + name)

    def read_rows(name):
        return [json.loads(line) for line in (directory / name).read_text().splitlines()]

    def order(row):
        return digest(row["_id"].encode()), row["_id"]

    documents = sorted(read_rows("corpus.jsonl"), key=order)[:1000]
    with (directory / "qrels/train.tsv").open() as stream:
        train = {r["query-id"] for r in csv.DictReader(stream, delimiter="\t")}
    queries = []
    used = set()
    for row in sorted(read_rows("queries.jsonl"), key=order):
        text = row["text"].strip()
        if row["_id"] in train and text and text.casefold() not in used:
            queries.append((row["_id"], text))
            used.add(text.casefold())
            if len(queries) == 32:
                break
    if len({r["_id"] for r in documents}) != 1000 or len(queries) != 32:
        raise ValueError("Incomplete workload")
    return documents, queries


async def prepare(client, qdrant, physical, documents, catalog_path: Path):
    started = perf_counter()
    sources = {}
    expected_chunks = 0
    for row in documents:
        source = "scifact:" + row["_id"]
        response = await client.post(
            "/v1/documents",
            json={
                "external_id": row["_id"],
                "source": source,
                "title": row["title"],
                "content": row["text"],
                "collection": LOGICAL,
            },
        )
        response.raise_for_status()
        item = response.json()
        if item["decision"] != "new" or item["chunks"] <= 0 or item["document_id"] in sources:
            raise ValueError("Invalid ingest response for " + source)
        sources[item["document_id"]] = source
        expected_chunks += item["chunks"]
        if len(sources) % 200 == 0:
            print(f"Ingested {len(sources)} documents", flush=True)
    ingestion_s = perf_counter() - started
    with sqlite3.connect(f"{catalog_path.as_uri()}?mode=ro", uri=True) as db:
        active = dict(
            db.execute(
                "SELECT document_id,source FROM documents WHERE collection=? AND status='active'",
                (LOGICAL,),
            )
        )
        chunks = [
            c[0]
            for c in db.execute(
                "SELECT c.id FROM chunks c JOIN documents d ON c.document_id=d.document_id "
                "AND c.version=d.active_version WHERE d.collection=? ORDER BY c.id",
                (LOGICAL,),
            )
        ]
    verified = await qdrant.post(f"/collections/{physical}/points/count", json={"exact": True})
    verified.raise_for_status()
    points = verified.json()["result"]["count"]
    if active != sources or not len(chunks) == expected_chunks == points:
        raise ValueError("Active catalog/Qdrant counts or source mapping differ from ingestion")
    return sources, dict(
        ingestion_s=ingestion_s,
        verification_s=perf_counter() - started - ingestion_s,
        documents=len(active),
        chunks=len(chunks),
        qdrant_points=points,
        index_id=digest("\n".join(chunks).encode()),
    )
