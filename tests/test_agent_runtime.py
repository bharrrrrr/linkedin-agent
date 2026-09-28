"""Offline contracts for the OpenAI agent adapter.

These tests never call OpenAI. They verify that the runtime sees the same 12
skills and that the approval/write boundary is represented in the adapter.
"""
from __future__ import annotations

import importlib.util
import unittest

AGENTS_AVAILABLE = importlib.util.find_spec("agents") is not None


@unittest.skipUnless(AGENTS_AVAILABLE, "openai-agents is not installed")
class AgentRuntimeContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from agent.skill_loader import discover_skills, skill_catalog_text
        from agent import tools

        cls.discover_skills = discover_skills
        cls.skill_catalog_text = skill_catalog_text
        cls.tools = tools

    def test_exactly_12_skills_are_discoverable(self):
        names = {spec.name for spec in self.discover_skills()}
        self.assertEqual(len(names), 12)

    def test_catalog_mentions_all_skills(self):
        catalog = self.skill_catalog_text()
        for spec in self.discover_skills():
            self.assertIn(spec.name, catalog)

    def test_all_write_tools_require_approval(self):
        for tool in self.tools.WRITE_TOOLS:
            self.assertTrue(
                tool.needs_approval,
                f"{tool.name} is write/cost bearing but not approval-gated",
            )

    def test_read_tools_are_not_approval_gated(self):
        for tool in self.tools.READ_TOOLS:
            self.assertFalse(
                tool.needs_approval,
                f"{tool.name} is read-only but unexpectedly requires approval",
            )


if __name__ == "__main__":
    unittest.main()
