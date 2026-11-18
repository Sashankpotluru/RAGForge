# Security boundaries

Run this unauthenticated API on loopback or within an equivalently trusted environment. Any caller can ingest, search, delete, select collections, request evaluation or trigger configured provider costs. Logical collection filtering is query scoping. Public deployment needs authentication, authorization, TLS, resource quotas and operational monitoring.

The service accepts structured text or uploaded bytes. Server-side input paths are unavailable via HTTP. Documents are capped at 20 MiB and request bodies at 28 MiB. Parsers run in-process. PDFs require extractable text. Run untrusted document workloads in an isolated environment with OS resource limits.

URL loading requires `RAGFORGE_URL_INGESTION_ENABLED=true` and a JSON array in `RAGFORGE_URL_ALLOWED_HOSTS`. Only explicitly allowed HTTPS hostnames on port 443 qualify. DNS results are checked for nonpublic and multicast addresses. The connection uses a numeric address and validates TLS against the original hostname. Each redirect repeats those checks. The client ignores proxy settings, rejects compressed and unsupported responses and permits at most three redirects. Use network-level egress controls alongside application checks.

Ingested text, prompts, citations, SQLite and Qdrant payloads may contain sensitive information. A configured external generation endpoint receives selected evidence on answer requests. The citation validator checks source identifiers and inline references. Factual entailment and completeness require independent assessment.

