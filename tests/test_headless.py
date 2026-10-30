"""Headless mode (#135) and idle scheduling (#136) with fake models and a fake clock."""
import os
import signal
import threading
import time
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from io import StringIO
from unittest.mock import patch

import main
from robits.core.config import _config
from robits.core.session import ExternalInput


class FakeRole:
    def __init__(self, name, responses=None):
        self.name = name
        self.responses = list(responses or [])
        self.received = []
        self.alarms = []
        self.waiting_until = None

    def interact(self, sender=None, prompt=None):
        self.received.append((sender, prompt))
        if self.responses:
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response
        return ""


class FakeClock:
    def __init__(self):
        self.now = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now


class FakeSleeper:
    """Advances the fake clock instead of sleeping, unless woken first.

    on_sleep(call_index) runs before the sleep, so a test can deliver an event
    or request a stop the way another thread would.
    """

    def __init__(self, clock, on_sleep=None):
        self.clock = clock
        self.on_sleep = on_sleep
        self.calls = []

    def __call__(self, seconds, wake_event):
        self.calls.append(seconds)
        if self.on_sleep is not None:
            self.on_sleep(len(self.calls))
        if wake_event.is_set():
            return
        if seconds is None:
            raise AssertionError("unbounded sleep with nothing to wake it")
        self.clock.now += timedelta(seconds=seconds)


class HeadlessTestCase(unittest.TestCase):
    def setUp(self):
        for name, value in {
            "memory_store": None,
            "agent_workspace_store": None,
            "_org_workspace": None,
            "break_schedule": [],
            "headless": False,
            "min_turn_interval_seconds": 0.0,
            "max_turns_per_hour": 0,
            "idle_max_sleep_seconds": 300.0,
            "error_backoff_seconds": 5.0,
        }.items():
            patcher = patch.object(_config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.clock = FakeClock()

    def session(self, participants, on_sleep=None, **kwargs):
        self.sleeper = FakeSleeper(self.clock, on_sleep)
        session = main.Session(
            participants=participants,
            run_id="headless-test",
            headless=True,
            clock=self.clock,
            sleeper=self.sleeper,
            **kwargs,
        )
        self.addCleanup(session.close)
        return session

    def run_quiet(self, session, **kwargs):
        with redirect_stdout(StringIO()):
            return session.run(**kwargs)

    def events(self, session, event_type):
        return [e for e in session.event_stream.events() if e.event_type == event_type]


class HeadlessRosterTests(HeadlessTestCase):
    def test_default_cast_has_no_human_when_headless(self):
        roster = main.build_employee_dict(headless=True)
        self.assertNotIn("CEO", roster)
        self.assertFalse(any(isinstance(p, main.Human) for p in roster.values()))
        self.assertIn("Ops", roster)

    def test_persona_roster_skips_humans_and_adds_no_ceo(self):
        personas = {
            "boss": {"role": "CEO", "full_name": "The Boss"},
            "alex": {"role": "SE", "full_name": "Alex Smith"},
        }
        with self.assertLogs(level="WARNING"):
            roster = main.build_employee_dict(personas, headless=True)
        self.assertEqual(list(roster), ["alex"])

    def test_interactive_roster_still_adds_ceo(self):
        roster = main.build_employee_dict({"alex": {"role": "SE", "full_name": "Alex Smith"}})
        self.assertTrue(any(isinstance(p, main.Human) for p in roster.values()))

    def test_headless_session_drops_human_participants(self):
        session = self.session({"CEO": main.Human(), "A": FakeRole("A")})
        self.assertEqual(list(session.participants), ["A"])
        self.assertIsInstance(session.last_receiver, ExternalInput)

    def test_ceo_name_has_no_blanket_tool_access_when_headless(self):
        tool = main.ToolDefinition(
            name="org.secret", description="", parameters={}, args=[], required_args=[], func=None,
        )
        role = FakeRole("CEO")
        role.capabilities = set()
        role.allowed_tools = set()
        self.assertTrue(main.role_can_use_tool(role, tool))
        role.runtime_headless = True
        self.assertFalse(main.role_can_use_tool(role, tool))
        del role.runtime_headless
        with patch.object(_config, "headless", True):
            self.assertFalse(main.role_can_use_tool(role, tool))

    def test_parse_args_accepts_headless(self):
        self.assertTrue(main.parse_args(["--headless", "--turns", "2"]).headless)
        self.assertFalse(main.parse_args([]).headless)


class HeadlessRunTests(HeadlessTestCase):
    def test_prompt_seed_comes_from_external_sender_without_stdin(self):
        a = FakeRole("A", ["B, hi there"])
        b = FakeRole("B", ["thanks"])
        session = self.session({"A": a, "B": b})
        with patch("builtins.input", side_effect=AssertionError("stdin read")):
            self.run_quiet(session, initial_message="hello", max_turns=2)
        self.assertEqual(session.turns_completed, 2)
        self.assertEqual(a.received[0][0], "external")
        self.assertIn("hello", a.received[0][1])
        self.assertEqual(b.received[0][0], "A")
        self.assertEqual(self.sleeper.calls, [])

    def test_bounded_run_with_nothing_to_do_ends_without_model_calls(self):
        a = FakeRole("A", ["unused"])
        session = self.session({"A": a})
        self.run_quiet(session, max_turns=3)
        self.assertEqual(a.received, [])
        self.assertEqual(session.turns_completed, 0)
        self.assertEqual(len(self.events(session, "session.idle_exit")), 1)

    def test_silence_does_not_force_another_turn(self):
        a = FakeRole("A", [""])
        b = FakeRole("B", ["unused"])
        session = self.session({"A": a, "B": b})
        self.run_quiet(session, initial_message="A: rest if you like", max_turns=5)
        self.assertEqual(session.turns_completed, 1)
        self.assertEqual(b.received, [])

    def test_loop_detector_remains_a_safety_net(self):
        a = FakeRole("A", ["same"] * 5)
        b = FakeRole("B", ["same"] * 5)
        session = self.session({"A": a, "B": b})
        self.run_quiet(session, initial_message="go", max_turns=10)
        self.assertEqual(len(self.events(session, "session.loop_detected")), 1)
        self.assertEqual(session.turns_completed, 3)


class IdleSchedulingTests(HeadlessTestCase):
    def test_all_waiting_sleeps_until_earliest_wake_without_model_calls(self):
        a = FakeRole("A", ["ok"])
        b = FakeRole("B", ["unused"])
        start = self.clock.now
        a.waiting_until = start + timedelta(minutes=10)
        b.waiting_until = start + timedelta(minutes=20)
        calls_while_asleep = []
        session = self.session(
            {"A": a, "B": b},
            on_sleep=lambda _n: calls_while_asleep.append(len(a.received) + len(b.received)),
        )
        self.run_quiet(session, initial_message="status?", max_turns=1)

        self.assertEqual(self.sleeper.calls, [300.0, 300.0])  # capped slices of the 10 min wait
        self.assertEqual(calls_while_asleep, [0, 0])
        sleeping = self.events(session, "session.sleeping")[0].payload
        self.assertEqual(sleeping["reason"], "no_persona_due")
        self.assertEqual(sleeping["wake_at"], (start + timedelta(minutes=10)).isoformat(timespec="seconds"))
        self.assertEqual(sleeping["personas"]["B"], b.waiting_until.isoformat(timespec="seconds"))
        self.assertEqual(len(a.received), 1)
        self.assertIn("Your wait has ended", a.received[0][1])
        self.assertIn("status?", a.received[0][1])
        self.assertEqual(b.received, [])

    def test_external_event_wakes_session_early(self):
        a = FakeRole("A", [""])
        a.waiting_until = self.clock.now + timedelta(hours=2)
        session = None

        def on_sleep(n):
            if n == 1:
                session.deliver("A: ping")
            else:
                session.request_stop("test")

        session = self.session({"A": a}, on_sleep=on_sleep)
        start = self.clock.now
        self.run_quiet(session)

        self.assertEqual(self.clock.now, start)  # woken before any time passed
        self.assertEqual(len(a.received), 1)
        self.assertIn("ping", a.received[0][1])
        completed = self.events(session, "session.completed")[0].payload
        self.assertEqual(completed["stop_reason"], "test")

    def test_due_alarm_wakes_its_persona(self):
        a = FakeRole("A", ["noted"])
        due = self.clock.now + timedelta(minutes=3)
        a.alarms.append(main.Alarm("alarm-1", "A", "stand-up", due.isoformat()))
        session = self.session({"A": a})
        self.run_quiet(session, max_turns=1)
        self.assertEqual(self.sleeper.calls, [180.0])
        self.assertIn("stand-up", a.received[0][1])

    def test_off_clock_personas_are_not_scheduled_until_clock_changes(self):
        a = FakeRole("A", [""])
        session = None

        def on_sleep(n):
            if n == 1:
                self.assertEqual(a.received, [])
                session.set_clock_state("on")
            else:
                session.request_stop("test")

        session = self.session({"A": a}, on_sleep=on_sleep, clock_state="off")
        self.run_quiet(session, initial_message="anyone there?")
        self.assertEqual(self.events(session, "session.sleeping")[0].payload["reason"], "off_clock")
        self.assertEqual(len(a.received), 1)
        self.assertIn("anyone there?", a.received[0][1])

    def test_bounded_off_clock_run_ends_instead_of_waiting_forever(self):
        a = FakeRole("A", ["unused"])
        session = self.session({"A": a}, clock_state="off")
        self.run_quiet(session, initial_message="anyone there?", max_turns=1)
        self.assertEqual(a.received, [])
        self.assertEqual(self.events(session, "session.idle_exit")[0].payload["reason"], "off_clock")

    def test_scheduled_break_boundary_is_a_wake_time(self):
        local = self.clock.now.astimezone()
        start = (local + timedelta(minutes=2)).strftime("%H:%M")
        end = (local + timedelta(minutes=3)).strftime("%H:%M")
        a = FakeRole("A")
        session = self.session({"A": a})
        with patch.object(_config, "break_schedule", [(start, end)]):
            wake = session._next_clock_change(self.clock.now)
        self.assertGreater(wake, self.clock.now)
        self.assertLessEqual(wake, self.clock.now + timedelta(minutes=2))

    def test_min_turn_interval_spaces_a_personas_turns(self):
        a = FakeRole("A", ["one", "two"])
        session = self.session({"A": a})
        with patch.object(_config, "min_turn_interval_seconds", 60.0):
            self.run_quiet(session, initial_message="go", max_turns=2)
        self.assertEqual(self.sleeper.calls, [60.0])
        self.assertEqual(len(a.received), 2)

    def test_turn_budget_defers_turns_to_the_next_hour(self):
        a = FakeRole("A", ["one", "two"])
        session = self.session({"A": a})
        with patch.object(_config, "max_turns_per_hour", 1), \
                patch.object(_config, "idle_max_sleep_seconds", 0.0):
            self.run_quiet(session, initial_message="go", max_turns=2)
        self.assertEqual(self.sleeper.calls, [3600.0])
        self.assertEqual(self.events(session, "session.sleeping")[0].payload["reason"], "turn_budget")

    def test_model_error_backs_off_and_retries_the_input(self):
        a = FakeRole("A", [RuntimeError("model down"), RuntimeError("still down"), "recovered"])
        session = self.session({"A": a})
        self.run_quiet(session, initial_message="hello", max_turns=1)
        self.assertEqual(self.sleeper.calls, [5.0, 10.0])
        self.assertEqual(len(a.received), 3)
        self.assertTrue(all("hello" in prompt for _, prompt in a.received))
        self.assertEqual(session.turns_completed, 1)
        self.assertEqual(len(self.events(session, "session.model_error")), 2)

    def test_failed_turn_does_not_repeat_the_input_in_history(self):
        class HistoryRole(FakeRole):
            def __init__(self, name, responses):
                super().__init__(name, responses)
                self.conversation_history = {name: [{"role": "system", "content": "s"}]}

            def interact(self, sender=None, prompt=None):
                self.conversation_history[self.name].append({"role": "user", "content": prompt})
                return super().interact(sender, prompt)

        a = HistoryRole("A", [RuntimeError("model down"), "recovered"])
        session = self.session({"A": a})
        self.run_quiet(session, initial_message="hello", max_turns=1)
        user_turns = [m for m in a.conversation_history["A"] if m["role"] == "user"]
        self.assertEqual(len(user_turns), 1)
        self.assertEqual(session.turns_completed, 1)

    def test_gives_up_after_max_consecutive_errors(self):
        a = FakeRole("A", [RuntimeError("down")] * 10)
        session = self.session({"A": a})
        with patch.object(_config, "max_consecutive_errors", 3):
            self.run_quiet(session, initial_message="hello", max_turns=1)
        self.assertEqual(len(a.received), 3)
        self.assertEqual(self.sleeper.calls, [5.0, 10.0])
        self.assertEqual(str(session.failed), "down")
        self.assertEqual(len(self.events(session, "session.error_exit")), 1)
        self.assertEqual(session.turns_completed, 0)

    def test_cli_exits_non_zero_after_giving_up(self):
        class Failing:
            headless = True
            failed = RuntimeError("down")
            system = None

            def run(self, **_):
                return self

            def stop_on_signals(self):
                from contextlib import nullcontext
                return nullcontext()

        with patch.object(main, "Session", return_value=Failing()), \
                patch.object(main, "load_tools"):
            with self.assertRaises(SystemExit) as raised:
                main.run_simulation(initial_message="hi", max_turns=1)
        self.assertEqual(raised.exception.code, 1)


class SigtermTests(HeadlessTestCase):
    def test_sigterm_interrupts_a_real_idle_sleep(self):
        a = FakeRole("A")
        session = main.Session(participants={"A": a}, run_id="sigterm-test", headless=True)
        self.addCleanup(session.close)
        timer = threading.Timer(0.1, os.kill, (os.getpid(), signal.SIGTERM))
        previous = signal.getsignal(signal.SIGTERM)
        started = time.monotonic()
        with session.stop_on_signals():
            timer.start()
            self.run_quiet(session)  # max_turns=None: would sleep 300s slices forever
        timer.join()
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(session._stop_requested, "SIGTERM")
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    def test_sigterm_during_a_turn_finishes_it_then_stops(self):
        session = None

        class TermOnTurn(FakeRole):
            def interact(self, sender=None, prompt=None):
                os.kill(os.getpid(), signal.SIGTERM)
                return super().interact(sender, prompt)

        a = TermOnTurn("A", ["B, carry on"])
        b = FakeRole("B", ["unused"])
        session = self.session({"A": a, "B": b})
        previous = signal.getsignal(signal.SIGTERM)
        with session.stop_on_signals():
            self.run_quiet(session, initial_message="go")
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)
        self.assertEqual(session.turns_completed, 1)
        self.assertEqual(b.received, [])


if __name__ == "__main__":
    unittest.main()
