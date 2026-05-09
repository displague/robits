import tempfile
import unittest
from pathlib import Path

from robits.runtime.tool_proposals import ToolProposalStore


class ToolProposalStoreTests(unittest.TestCase):
    def test_json_store_persists_proposals(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tool_proposals.json"
            store = ToolProposalStore(path)
            proposal = store.create(
                requested_by="SE",
                tool_name="weather.lookup",
                description="Look up weather.",
            )

            reloaded = ToolProposalStore(path)

        self.assertEqual(reloaded.get(proposal["proposal_id"])["tool_name"], "weather.lookup")

    def test_constructor_defers_directory_creation_until_save(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "tool_proposals.json"

            ToolProposalStore(path)

            self.assertFalse(path.parent.exists())

    def test_returned_records_do_not_mutate_store_state(self):
        store = ToolProposalStore()
        proposal = store.create(
            requested_by="SE",
            tool_name="weather.lookup",
            description="Look up weather.",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        )

        proposal["parameters"]["required"].append("unit")
        listed = store.list()
        listed[0]["parameters"]["required"].append("country")
        fetched = store.get(proposal["proposal_id"])

        self.assertEqual(fetched["parameters"]["required"], ["location"])


if __name__ == "__main__":
    unittest.main()
