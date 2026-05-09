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


if __name__ == "__main__":
    unittest.main()
