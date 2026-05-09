import tempfile
import unittest

from robits.runtime.workspace import AgentWorkspaceStore, WorkspacePathError


class AgentWorkspaceStoreTests(unittest.TestCase):
    def test_workspace_persists_files_under_agent_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = AgentWorkspaceStore(temp_dir)
            store.write("SE", "TODO.md", "ship it")
            reloaded = AgentWorkspaceStore(temp_dir)

            result = reloaded.read("SE", "TODO.md")

        self.assertEqual(result["content"], "ship it")
        self.assertEqual(result["path"], "TODO.md")

    def test_workspace_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = AgentWorkspaceStore(temp_dir)

            with self.assertRaises(WorkspacePathError):
                store.write("SE", "../escape.txt", "nope")


if __name__ == "__main__":
    unittest.main()
