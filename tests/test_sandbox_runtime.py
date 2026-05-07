import unittest

import main
from robits.runtime.sandbox import FakeSandboxBackend, SandboxMetadata, SandboxRuntime


class SandboxRuntimeTests(unittest.TestCase):
    def test_default_agents_have_disabled_sandbox_metadata(self):
        employees = main.build_employee_dict()

        self.assertFalse(employees["SE"].sandbox_metadata.enabled)
        self.assertEqual(employees["SE"].sandbox_metadata.backend, "none")
        self.assertFalse(employees["CEO"].sandbox_metadata.enabled)

    def test_disabled_sandbox_runtime_skips_execution(self):
        runtime = SandboxRuntime()
        metadata = SandboxMetadata.disabled("SE")

        result = runtime.execute_tool(metadata, "org.create_role", {"role_name": "QA"})

        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.backend, "none")

    def test_fake_backend_receives_private_and_shared_workspace_metadata(self):
        backend = FakeSandboxBackend(output="done", name="local_process")
        runtime = SandboxRuntime(backend=backend)
        metadata = SandboxMetadata.local_process(
            "SE",
            private_workspace="/agents/SE",
            shared_organization_workspace="/organization",
        )

        result = runtime.execute_tool(metadata, "project.build", {"target": "runtime"})

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.output, "done")
        self.assertEqual(backend.requests[0].agent_name, "SE")
        self.assertEqual(backend.requests[0].private_workspace, "/agents/SE")
        self.assertEqual(backend.requests[0].shared_organization_workspace, "/organization")
        self.assertEqual(backend.requests[0].arguments, {"target": "runtime"})

    def test_enabled_sandbox_without_backend_fails_explicitly(self):
        runtime = SandboxRuntime()
        metadata = SandboxMetadata.local_process(
            "SE",
            private_workspace="/agents/SE",
            shared_organization_workspace="/organization",
        )

        result = runtime.execute_tool(metadata, "project.build")

        self.assertEqual(result.status, "error")
        self.assertIn("no backend", result.output)

    def test_enabled_sandbox_requires_matching_backend(self):
        runtime = SandboxRuntime(backend=FakeSandboxBackend(name="container"))
        metadata = SandboxMetadata.local_process(
            "SE",
            private_workspace="/agents/SE",
            shared_organization_workspace="/organization",
        )

        result = runtime.execute_tool(metadata, "project.build")

        self.assertEqual(result.status, "error")
        self.assertIn("backend mismatch", result.output)


if __name__ == "__main__":
    unittest.main()
