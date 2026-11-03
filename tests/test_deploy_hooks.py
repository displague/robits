"""Deployment hooks (#141): heartbeat file, health check and a mounted tool catalogue."""
import os
import tempfile
import time
import unittest
from datetime import timedelta
from io import StringIO
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import main
from robits import health
from robits.core.config import _config
from tests.test_headless import FakeRole, HeadlessTestCase


class HeartbeatTests(HeadlessTestCase):
    def test_headless_loop_touches_heartbeat_on_every_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            beat = Path(tmp) / "heartbeat"
            touches = []
            a = FakeRole("A", ["ok"])
            a.waiting_until = self.clock.now + timedelta(minutes=10)

            def on_sleep(_n):
                touches.append(beat.exists())
                os.utime(beat, (0, 0))  # age it, so the next pass must touch it again

            with patch.object(_config, "heartbeat_file", str(beat)):
                session = self.session({"A": a}, on_sleep=on_sleep)
                self.run_quiet(session, initial_message="status?", max_turns=1)
            self.assertEqual(touches, [True, True])
            self.assertGreater(beat.stat().st_mtime, time.time() - 60, "touched after the last sleep")


class HealthCheckTests(unittest.TestCase):
    def test_fresh_stale_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            beat = Path(tmp) / "heartbeat"
            self.assertFalse(health.check(str(beat), 900)[0])
            beat.touch()
            self.assertTrue(health.check(str(beat), 900)[0])
            os.utime(beat, (time.time() - 1000,) * 2)
            ok, message = health.check(str(beat), 900)
            self.assertFalse(ok)
            self.assertIn("old", message)
        self.assertFalse(health.check("", 900)[0])

    def test_main_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            beat = Path(tmp) / "heartbeat"
            beat.touch()
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(health.main(["--file", str(beat), "--max-age", "60"]), 0)
                self.assertEqual(health.main(["--file", str(beat) + ".missing"]), 1)


class ToolsFileTests(unittest.TestCase):
    def test_session_loads_tools_from_configured_file(self):
        loaded = []
        with patch.object(_config, "tools_file", "/config/tools.yaml"), \
                patch.object(main, "load_tools", lambda system, path=None: loaded.append(path)), \
                patch.object(main, "Session") as session_cls:
            session_cls.return_value.headless = False
            with redirect_stdout(StringIO()):
                main.run_simulation(initial_message=None, max_turns=0)
        self.assertEqual(loaded, ["/config/tools.yaml"])


class KubernetesExampleTests(unittest.TestCase):
    """The example roster in deploy/kubernetes must load with every persona intact."""

    def test_example_personas_load(self):
        from robits.core.persona import load_personas
        path = Path(__file__).parents[1] / "deploy" / "kubernetes" / "config" / "personas.yaml"
        with patch.object(_config, "memory_store", None):
            employee_dict = main.build_employee_dict(load_personas(str(path)), headless=True)
        self.assertEqual(sorted(employee_dict), ["ops_lead", "sam_writer"])
        self.assertEqual(employee_dict["ops_lead"].allowed_tools,
                         {"agent.think", "agent.wait", "memory.search"})


if __name__ == "__main__":
    unittest.main()
