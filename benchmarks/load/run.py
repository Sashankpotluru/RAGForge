"""Run the backend-load-v1 protocol on loopback, using a fresh private runtime."""

import asyncio
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import uuid
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import httpx
from huggingface_hub import hf_hub_download

from benchmarks.load.collector import collect, diagnostics, metrics, summarize, validate
from benchmarks.load.workload import FILES, MODEL, REVISION, SEARCH, digest, load_workload, prepare

ROOT = Path(__file__).resolve().parents[2]
# Rotate level order between repetitions to reduce systematic order effects.
ORDER = (1, 4, 8, 16, 8, 16, 1, 4, 16, 8, 4, 1)
SCOPE = (
    "Warm loopback HTTP hybrid search on one host, one Uvicorn worker, SQLite and native Qdrant."
)
MEASURED = ["benchmarks/load", ":(glob)src/ragforge/**/*.py", "pyproject.toml", "uv.lock"]
MAC_ARCHIVE = "531904e822050790bb8c1da6705f801ee48b91eb4c229d0d605ff06f6331b20a"
MAC_BINARY = "830bd61d0efdba90e0737e174cefe32d4f9e0161b0e2b8684d655fb774544fd3"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def source():
    if git("status", "--porcelain", "--", *MEASURED):
        raise RuntimeError("Commit measured source before running")
    files = git("ls-files", "--", *MEASURED).splitlines()
    return dict(
        commit=git("rev-parse", "HEAD"),
        measured_paths_clean=True,
        checkout_dirty=bool(git("status", "--porcelain")),
        files_sha256={name: digest((ROOT / name).read_bytes()) for name in files},
    )


def ports(count):
    with ExitStack() as stack:
        sockets = [stack.enter_context(socket.socket()) for _ in range(count)]
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        return [s.getsockname()[1] for s in sockets]


@contextmanager
def process(command, cwd, env, log):
    with log.open("wb") as stream:
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
        try:
            yield proc
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


async def wait_ready(client, path, process):
    for _ in range(600):
        if process.poll() is not None:
            raise RuntimeError(f"Service exited {process.returncode}; inspect ignored runtime logs")
        try:
            response = await client.get(path)
            if response.is_success:
                return response
        except httpx.RequestError:
            pass
        await asyncio.sleep(0.2)
    raise TimeoutError("Service readiness exceeded budget")


def hardware():
    result = dict(
        os=platform.system(),
        release=platform.release(),
        architecture=platform.machine(),
        python=platform.python_version(),
        logical_cpus=os.cpu_count(),
    )
    if platform.system() == "Darwin":

        def sysctl(name):
            return subprocess.check_output(["sysctl", "-n", name], text=True).strip()

        result.update(
            model=sysctl("machdep.cpu.brand_string"),
            physical_cpus=int(sysctl("hw.physicalcpu")),
            installed_ram_bytes=int(sysctl("hw.memsize")),
            macos=platform.mac_ver()[0],
        )
    return result


async def main():
    started = datetime.now(UTC).isoformat()
    code = source()
    binary = Path(os.environ["QDRANT_BINARY"]).resolve(strict=True)
    binary_hash = digest(binary.read_bytes())
    archive = Path(os.environ["QDRANT_ARCHIVE"]).resolve(strict=True)
    archive_hash = digest(archive.read_bytes())
    server_version = subprocess.check_output([str(binary), "--version"], text=True).strip()
    if server_version != "qdrant 1.20.0":
        raise ValueError("Qdrant v1.20.0 required")
    if (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and (archive_hash != MAC_ARCHIVE or binary_hash != MAC_BINARY)
    ):
        raise ValueError("Qdrant native release digest mismatch")
    model_path = hf_hub_download(
        MODEL, "model.safetensors", revision=REVISION, local_files_only=True
    )
    weights = digest(Path(model_path).read_bytes())
    if weights != "b57380d8465cb456819716ab92ba12933f8e9142ae5f930ba18ca830e9333af2":
        raise ValueError("Unexpected cached weights")
    t0 = perf_counter()
    documents, queries = load_workload(ROOT / "data/scifact")
    selection_s = perf_counter() - t0
    api_port, http_port, grpc_port = ports(3)
    api_url, q_url = f"http://127.0.0.1:{api_port}", f"http://127.0.0.1:{http_port}"
    physical = "load_" + uuid.uuid4().hex
    log_dir = ROOT / "data/runtime-load-logs" / physical
    log_dir.mkdir(parents=True)
    base_env = {k: v for k, v in os.environ.items() if not k.startswith(("QDRANT__", "RAGFORGE_"))}
    api_env = dict(
        base_env,
        HF_HUB_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="6",
        MKL_NUM_THREADS="6",
        RAGFORGE_EMBEDDING_MODEL=MODEL,
        RAGFORGE_EMBEDDING_REVISION=REVISION,
        RAGFORGE_EMBEDDING_DEVICE="cpu",
        RAGFORGE_QDRANT_URL=q_url,
        RAGFORGE_QDRANT_COLLECTION=physical,
    )
    q_env = dict(
        base_env,
        QDRANT__TELEMETRY_DISABLED="true",
        QDRANT__SERVICE__HOST="127.0.0.1",
        QDRANT__SERVICE__HTTP_PORT=str(http_port),
        QDRANT__SERVICE__GRPC_PORT=str(grpc_port),
    )
    trials = []
    warmups = []
    with tempfile.TemporaryDirectory(prefix="ragforge-load-") as tmp, ExitStack() as stack:
        runtime = Path(tmp).resolve()
        api_env["RAGFORGE_DATA_DIR"] = str(runtime / "api")
        launch = perf_counter()
        q_proc = stack.enter_context(
            process([str(binary), "--disable-telemetry"], runtime, q_env, log_dir / "qdrant.log")
        )
        async with (
            httpx.AsyncClient(base_url=q_url, trust_env=False) as qdrant,
            httpx.AsyncClient(
                base_url=api_url,
                trust_env=False,
                timeout=30,
                limits=httpx.Limits(max_connections=16, max_keepalive_connections=16),
            ) as client,
        ):
            await wait_ready(qdrant, "/readyz", q_proc)
            api_proc = stack.enter_context(
                process(
                    [sys.executable, str(ROOT / "benchmarks/load/serve.py"), str(api_port)],
                    runtime,
                    api_env,
                    log_dir / "api.log",
                )
            )
            ready = (await wait_ready(client, "/readyz", api_proc)).json()
            startup_s = perf_counter() - launch
            sources, prep = await prepare(
                client, qdrant, physical, documents, runtime / "api/catalog.sqlite3"
            )
            execution = json.loads((runtime / "execution.json").read_text())
            for trial_no, clients in enumerate(ORDER, 1):
                # Warm all 32 query texts before each trial; validation must pass.
                warm = perf_counter()
                for _, query in queries:
                    payload = {**SEARCH, "query": query}
                    response = await client.post("/v1/search", json=payload)
                    response.raise_for_status()
                    validate(response, payload, sources, ready["config_id"], prep["index_id"])
                warmups.append(perf_counter() - warm)
                trial = await collect(
                    client,
                    queries,
                    SEARCH,
                    sources,
                    ready["config_id"],
                    prep["index_id"],
                    clients,
                    30,
                )
                trial["trial"] = trial_no
                trial["summary"] = {
                    **metrics(trial["requests"], trial["elapsed_s"]),
                    **diagnostics(trial["requests"]),
                }
                trials.append(trial)
                (log_dir / f"trial-{trial_no}.json").write_text(json.dumps(trial))
                print(f"Trial {trial_no}/12, clients={clients}: {trial['summary']}", flush=True)
    scenarios = summarize(trials)
    ids = [d["_id"] for d in documents]
    report = dict(
        schema_version=1,
        scope=SCOPE,
        started_utc=started,
        source=code,
        hardware=hardware(),
        packages={
            n: version(n)
            for n in (
                "ragforge",
                "fastapi",
                "starlette",
                "anyio",
                "uvicorn",
                "httpx",
                "qdrant-client",
                "sentence-transformers",
                "torch",
                "transformers",
                "huggingface-hub",
            )
        },
        execution=execution,
        qdrant=dict(
            version=server_version,
            binary_sha256=binary_hash,
            archive_sha256=archive_hash,
            collection=physical,
            telemetry=False,
            http_port=http_port,
            grpc_port=grpc_port,
        ),
        api_port=api_port,
        config_id=ready["config_id"],
        model=dict(name=MODEL, revision=REVISION, weights_sha256=weights),
        search=SEARCH,
        preparation=dict(selection_s=selection_s, startup_s=startup_s, **prep),
        workload=dict(
            dataset="BEIR SciFact",
            input_sha256=FILES,
            document_ids=ids,
            document_ids_sha256=digest("\n".join(ids).encode()),
            queries=queries,
            ordering=(
                "Ascending SHA-256 of UTF-8 ID; first 1000 documents, first 32 distinct "
                "trimmed casefolded train texts. ID digest joins ordered IDs with LF and no "
                "trailing LF."
            ),
        ),
        protocol=dict(
            clients=list(ORDER),
            scheduled_seconds=30,
            timeout_s=30,
            warmup="Each trial preceded by all 32 validated queries sequentially.",
            warmup_s=warmups,
            client=(
                "Co-located async HTTP/1.1 keep-alive; zero think time, closed loop, no retries."
            ),
            scheduling=(
                "Each worker cycles query index (worker + local sequence * clients) modulo "
                "32, starting over each trial. Trial stops issuing at deadline and drains in "
                "flight."
            ),
            quantile=(
                "Nearest rank over successful end-to-end request times, including JSON "
                "parsing and validation."
            ),
            resources="Process CPU and RSS unmeasured.",
            limits=(
                "Closed-loop clients self-throttle during stalls (coordinated omission). "
                "Single-host observations do not establish open-loop capacity, WAN latency or "
                "an SLA. Tiny-BERT differs from the browser MiniLM q8 model. No generation, "
                "rewriting or reranking."
            ),
        ),
        trials=trials,
        scenarios=scenarios,
    )
    report_bytes = (json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    summary = dict(
        schema_version=1, report_sha256=digest(report_bytes), scope=SCOPE, scenarios=scenarios
    )
    out = ROOT / "benchmarks/data/backend-load-v1"
    out.mkdir(exist_ok=True)
    (ROOT / "evaluations/results/backend-load-v1.json").write_bytes(report_bytes)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
