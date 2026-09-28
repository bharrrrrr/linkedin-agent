#!/usr/bin/env python3
"""Run the OpenAI LinkedIn Agent locally.

Examples:
    python3 scripts/run_agent.py "Write a LinkedIn post about my cybersecurity internship."
    python3 scripts/run_agent.py --session kani "Draft a comment for https://www.linkedin.com/..."
    python3 scripts/run_agent.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.cli import main_async  # noqa: E402


if __name__ == "__main__":
    import asyncio

    raise SystemExit(asyncio.run(main_async()))
