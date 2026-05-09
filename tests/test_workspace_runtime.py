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

            with self.assertRaises(WorkspacePathError):
                store.write("SE", "..\\escape.txt", "nope")

            with self.assertRaises(WorkspacePathError):
                store.write("SE", "C:/escape.txt", "nope")

    def test_workspace_read_streams_to_limit_and_validates_max_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = AgentWorkspaceStore(temp_dir)
            store.write("SE", "LOG.txt", "abcdef")

            result = store.read("SE", "LOG.txt", max_bytes=3)

            self.assertEqual(result["content"], "abc")
            self.assertTrue(result["truncated"])
            self.assertEqual(result["size"], 6)
            with self.assertRaises(WorkspacePathError):
                store.read("SE", "LOG.txt", max_bytes=-1)

    def test_workspace_delete_sanitizes_non_empty_directory_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = AgentWorkspaceStore(temp_dir)
            store.write("SE", "notes/TODO.md", "ship it")

            with self.assertRaisesRegex(WorkspacePathError, "Could not delete directory 'notes'"):
                store.delete("SE", "notes")


if __name__ == "__main__":
    unittest.main()
