"""Start an ordinary single-worker API with a reproducible Torch execution profile."""

import asyncio
import json
import sqlite3
import sys
from pathlib import Path

import torch
import uvicorn
from anyio.to_thread import current_default_thread_limiter, run_sync

from ragforge.api.app import create_app


async def main():
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    Path("execution.json").write_text(
        json.dumps(
            {
                "sqlite": sqlite3.sqlite_version,
                "torch_intraop": await run_sync(torch.get_num_threads),
                "torch_interop": await run_sync(torch.get_num_interop_threads),
                "anyio_thread_tokens": current_default_thread_limiter().total_tokens,
                "uvicorn_workers": 1,
                "device": "cpu",
            }
        )
    )
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(), host="127.0.0.1", port=int(sys.argv[1]), workers=1, access_log=False
        )
    )
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
