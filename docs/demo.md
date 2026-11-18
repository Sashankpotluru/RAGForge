# Sample evidence workbench

This guide runs the Python-backed demo on your own machine. The
[hosted browser edition](https://sashankpotluru.github.io/RAGForge/) runs on GitHub
Pages and has a separate [setup guide](browser-demo.md).

The workbench serves six original synthetic policy documents. Open Workbench to search, Corpus to read each full source, Results to compare the recorded retrieval measurements, and Architecture to follow the data flow. Clicking a sample question fills the question field. Find evidence runs retrieval. Generate answer runs retrieval followed by local inference.

## Start on loopback

Use Python 3.12 or 3.13 and uv. The standard demo uses deterministic hash embeddings plus BM25 and requires no network once dependencies are installed.

```bash
uv sync --locked
uv run --no-sync ragforge demo --port 8787 --data-dir data/public-demo
```

For neural retrieval, reranking and local generation, install the optional ML dependencies and provision the pinned public model files. The command below uses Hugging Face Hub's downloader from the installed dependencies.

```bash
uv sync --locked --extra ml
uv run --no-sync python - <<'PY'
from huggingface_hub import snapshot_download
for name, revision in [
    ("sentence-transformers-testing/stsb-bert-tiny-safetensors", "f3cb857cba53019a20df283396bcca179cf051a4"),
    ("cross-encoder/ms-marco-TinyBERT-L2-v2", "81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
]:
    snapshot_download(name, revision=revision, allow_patterns=["*.json","*.safetensors","*.txt","*.jinja","*.model"])
PY
HF_HUB_OFFLINE=1 uv run --no-sync ragforge demo --port 8787 --data-dir data/public-demo --neural --generate
```

Visit [the local workbench](http://127.0.0.1:8787). This command binds to loopback and runs a single worker. A clean shutdown removes its temporary sample index. If an abrupt shutdown leaves files behind, use another empty dedicated directory.

## Temporary public URL

Install the [official Cloudflare tunnel client](https://developers.cloudflare.com/tunnel/downloads/)
if `cloudflared` is not already available. With the demo running, start a Quick
Tunnel in a second terminal.

```bash
cloudflared tunnel --url http://127.0.0.1:8787
```

Share the printed HTTPS URL. The free temporary URL depends on the host staying awake and both processes staying online. Restarting the tunnel can assign a different URL. There is no uptime guarantee. Press Ctrl+C in the tunnel terminal to close public access, then Ctrl+C in the workbench terminal to stop the application.

## Query and evidence boundaries

The public service indexes its packaged sample into a new temporary directory. It exposes fixed read-only sample data, five or fewer search hits, and optional cited answers. The private administration API runs in a separate application. The public query routes share a thirty-request sliding minute and three concurrent slots across all visitors. One slot at a time may generate. Requests accept up to 500 question characters, 8 KiB of JSON, and a five-second body-read deadline.

The pinned Qwen model runs on the host, with a 1,024-token context limit and at most 72 generated tokens. The question and evidence stay on the host during inference. The model is prompted to quote evidence. The service attaches a citation only when the response contains at least two words and the complete generated phrase matches a single retrieved passage, ignoring case and terminal punctuation. A failed match returns the retrieved evidence with a generation status. A match establishes textual provenance. Readers should still check whether the selected passage answers the question.

## Recorded local generation

Three sequential requests used hybrid reranking and the pinned local generator on Apple Silicon with MPS. Hardware and model cache state affect timing. The backup question returned “Database backups run every six hours. [C1]” in 325 ms of generation time. The credential expiry and audit retention questions returned “Four hours [C1]” and “One year [C1]”, each in about 77 ms. These three successful samples illustrate constrained extractive generation. In an initial free-form trial the same three questions yielded no answers passing the requested inline citation format. [Recorded outputs, passages and timings](../evaluations/results/local-generation-v1.json)
