"""Interactive command-line entry point for the OpenAI LinkedIn agent."""
from __future__ import annotations

import argparse
import asyncio
import sys

from .runtime import LinkedInAgentRuntime


def _approval_lines(run) -> str:
    rows = ["\nApproval required before the agent can continue:"]
    for approval in run.approvals():
        rows.append(
            f"\nTool: {approval.tool_name}\nArguments: {approval.arguments}"
        )
    return "\n".join(rows)


async def main_async() -> int:
    parser = argparse.ArgumentParser(
        description="Run the OpenAI LinkedIn Agent locally."
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="One request. Omit for interactive mode.",
    )
    parser.add_argument(
        "--session",
        default="local",
        help="Persistent conversation id.",
    )
    args = parser.parse_args()

    runtime = LinkedInAgentRuntime()

    async def handle(prompt: str) -> None:
        run = await runtime.run(prompt, session_id=args.session)
        while True:
            if run.interrupted:
                print(_approval_lines(run))
                answer = input("Approve these actions? [y/N]: ").strip().lower()
                run = await runtime.resume_with_approvals(
                    run,
                    approve=answer in {"y", "yes"},
                )
                continue
            print(run.result.final_output)
            break

    if args.prompt:
        await handle(" ".join(args.prompt))
        return 0

    print("LinkedIn Agent (OpenAI). Type 'exit' to quit.")
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if prompt.lower() in {"exit", "quit"}:
            return 0
        if not prompt:
            continue
        try:
            await handle(prompt)
        except Exception as exc:
            print(
                f"Agent error: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
