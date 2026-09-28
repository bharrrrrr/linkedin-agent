#!/usr/bin/env python3
"""Verify the local OpenAI LinkedIn agent graph without making an API call."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    try:
        import agents  # noqa: F401
    except ImportError:
        print("FAIL: openai-agents is not installed. Run: pip install -r requirements.txt")
        return 1

    from agent.runtime import LinkedInAgentRuntime
    from agent.skill_loader import discover_skills

    runtime = LinkedInAgentRuntime()
    skills = discover_skills()
    specialist_names = {
        getattr(getattr(handoff, "agent", handoff), "name", str(handoff))
        for handoff in runtime.agent.handoffs
    }

    print(f"Model: {runtime.model}")
    print(f"Skills discovered: {len(skills)}")
    print(f"Specialist handoffs: {len(specialist_names)}")
    print(f"OPENAI_API_KEY: {'set' if os.getenv('OPENAI_API_KEY') else 'not set (offline smoke test)'}")
    print(f"State DB: {runtime.session_db}")

    if len(skills) != 12:
        print("FAIL: expected exactly 12 LinkedIn skills")
        return 1
    if len(specialist_names) != 12:
        print("FAIL: expected exactly 12 specialist handoffs")
        return 1

    print("PASS: OpenAI LinkedIn agent graph is wired correctly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
