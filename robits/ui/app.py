"""Textual TUI application for Robits simulation observability."""
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.pretty import Pretty
from rich.text import Text
import threading
from datetime import timezone
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Footer, Header, Label, ListItem, ListView, RichLog, Static, Input

import main
from robits.core.config import _config
from robits.core.session import Session
from robits.core.roles import load_tools


class RobitsDbReader:
    """Reads observability data from the SQLite database."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _get_conn(self) -> sqlite3.Connection:
        # Open in read-only mode to prevent write locks during simulation execution
        db_uri = Path(self.db_path).resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(db_uri, uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def list_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT session_id, started_at 
                FROM sessions 
                ORDER BY started_at DESC 
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error:
            return []
        finally:
            conn.close()

    def get_active_channels(self, session_id: str) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            # Select channels that have messages in this session
            cursor.execute(
                """
                SELECT DISTINCT c.channel_id, c.channel_type, c.participants_json, c.visibility
                FROM channels c
                JOIN messages m ON m.channel_id = c.channel_id
                WHERE m.session_id = ?
                """,
                (session_id,),
            )
            msg_channels = [dict(row) for row in cursor.fetchall()]

            # Select channels that have thoughts in this session
            cursor.execute(
                """
                SELECT DISTINCT c.channel_id, c.channel_type, c.participants_json, c.visibility
                FROM channels c
                JOIN thoughts t ON t.channel_id = c.channel_id
                WHERE t.session_id = ?
                """,
                (session_id,),
            )
            thought_channels = [dict(row) for row in cursor.fetchall()]

            seen = set()
            channels = []
            for ch in msg_channels + thought_channels:
                if ch["channel_id"] not in seen:
                    seen.add(ch["channel_id"])
                    channels.append(ch)
            return channels
        except sqlite3.Error:
            return []
        finally:
            conn.close()

    def get_channel_contents(
        self, session_id: str, channel_id: int, channel_type: str
    ) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            if channel_type == "agent_thought":
                cursor.execute(
                    """
                    SELECT 'thought' AS type, thought_id AS id, agent_id AS sender, 
                           NULL AS receiver, content, visibility, created_at
                    FROM thoughts
                    WHERE session_id = ? AND channel_id = ?
                    ORDER BY created_at, thought_id
                    """,
                    (session_id, channel_id),
                )
            else:
                cursor.execute(
                    """
                    SELECT 'message' AS type, message_id AS id, sender_agent_id AS sender, 
                           receiver_agent_id AS receiver, content, visibility, created_at
                    FROM messages
                    WHERE session_id = ? AND channel_id = ?
                    ORDER BY created_at, message_id
                    """,
                    (session_id, channel_id),
                )
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error:
            return []
        finally:
            conn.close()

    def get_runtime_events(self, session_id: str) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT event_id, sequence, event_type, visibility, payload_json, created_at
                FROM runtime_events
                WHERE session_id = ?
                ORDER BY COALESCE(sequence, event_id), event_id
                """,
                (session_id,),
            )
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error:
            return []
        finally:
            conn.close()

    def get_token_totals(self, session_id: str) -> Tuple[int, int, int]:
        conn = self._get_conn()
        prompt = 0
        completion = 0
        total = 0
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT payload_json FROM runtime_events
                WHERE session_id = ? AND event_type = 'token_usage'
                """,
                (session_id,),
            )
            for row in cursor.fetchall():
                try:
                    payload = json.loads(row["payload_json"])
                    prompt += payload.get("prompt_tokens", 0) or 0
                    completion += payload.get("completion_tokens", 0) or 0
                    total += payload.get("total_tokens", 0) or 0
                except Exception:
                    pass
            return prompt, completion, total
        except sqlite3.Error:
            return 0, 0, 0
        finally:
            conn.close()

    def get_agents(self) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT agent_id, role, display_name, username, lifecycle_state, metadata_json
                FROM agents
                ORDER BY agent_id
                """
            )
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error:
            return []
        finally:
            conn.close()



class RobitsTuiApp(App):
    """Textual TUI for Robits observability."""

    TITLE = "ROBITS Observability"
    SUB_TITLE = "Decoupled observer"

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("p", "cycle_policy", "Cycle Policy"),
    ]

    CSS = """
    Screen {
        background: #121820;
    }

    #sidebar {
        width: 32;
        border-right: tall #2A3644;
        background: #182230;
    }

    .sidebar-title {
        background: #1F2E44;
        color: #8AB4F8;
        text-align: center;
        text-style: bold;
        padding: 0 1;
    }

    #session_list {
        height: 8;
        border-bottom: solid #2A3644;
    }

    #channel_list {
        height: 1fr;
    }

    #main_panel {
        width: 3fr;
    }

    #transcript_title {
        background: #0D2636;
        color: #8AB4F8;
        text-align: center;
        text-style: bold;
        padding: 0 1;
    }

    #transcript_log {
        background: #090D14;
        border: none;
    }

    #event_panel {
        width: 2fr;
        border-left: tall #2A3644;
    }

    .panel-title {
        background: #1F2E44;
        color: #8AB4F8;
        text-align: center;
        text-style: bold;
        padding: 0 1;
    }

    #event_list {
        height: 2fr;
        border-bottom: solid #2A3644;
    }

    #event_inspector {
        height: 1fr;
        background: #090D14;
        padding: 1;
        overflow: auto;
    }

    ListItem {
        padding: 0 1;
    }

    ListItem:hover {
        background: #253346;
    }

    ListItem.-active {
        background: #1F3F66;
        color: #FFFFFF;
    }

    #agent_list {
        height: 1fr;
        border-top: solid #2A3644;
    }

    #status_indicator {
        background: #1B2936;
        color: #B0C4DE;
        padding: 0 1;
        height: 1;
        border-top: solid #2A3644;
        border-bottom: solid #2A3644;
        text-style: italic;
    }

    #message_input {
        background: #090D14;
        color: #E0E0E0;
        border: none;
        height: 3;
    }
    """

    def __init__(self, db_path: str, session_id: Optional[str] = None, policy: str = "full", interactive: bool = False):
        super().__init__()
        import getpass
        self.human_name = getpass.getuser()
        self.reader = RobitsDbReader(db_path)
        self.selected_session_id = session_id
        self.policy = policy
        self.interactive = interactive
        self.selected_channel_id: Optional[int] = None
        self.selected_channel_type: Optional[str] = None

        # Thread synchronization for interactive simulation
        self.input_received_event = threading.Event()
        self.last_user_input = ""
        self.session: Optional[Session] = None
        self.is_ceo_turn = False

        # Change-tracking states to optimize updates without full clears
        self.sessions_cache: List[Dict[str, Any]] = []
        self.channels_cache: List[Dict[str, Any]] = []
        self.agents_cache: List[Dict[str, Any]] = []
        self.loaded_message_ids: set = set()
        self.loaded_event_ids: set = set()
        self.events_cache: List[Dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Label("SESSIONS", classes="sidebar-title")
                yield ListView(id="session_list")
                yield Label("CHANNELS", classes="sidebar-title")
                yield ListView(id="channel_list")
                yield Label("AGENTS / PERSONAS", classes="sidebar-title")
                yield ListView(id="agent_list")
            with Vertical(id="main_panel"):
                yield Label("TRANSCRIPT (Policy: " + self.policy + ")", id="transcript_title")
                yield RichLog(id="transcript_log", max_lines=1000)
                
                # Dynamic status indicator and message input area
                placeholder_text = f"{self.human_name} > " if self.interactive else "[Read-Only Mode] Refresh: r, Cycle Policy: p"
                yield Label("Idle" if self.interactive else placeholder_text, id="status_indicator")
                yield Input(placeholder=placeholder_text, id="message_input", disabled=not self.interactive)
                
            with Vertical(id="event_panel"):
                yield Label("RUNTIME EVENTS", classes="panel-title")
                yield ListView(id="event_list")
                yield Label("EVENT INSPECTOR", classes="panel-title")
                yield Static(id="event_inspector")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_data()
        self.set_interval(1.0, self.refresh_data)
        if self.interactive:
            self.start_interactive_session()

    def start_interactive_session(self) -> None:
        from robits.memory.sqlite import SQLiteMemoryStore
        _config.memory_db_path = self.reader.db_path
        _config.memory_store = SQLiteMemoryStore(self.reader.db_path)

        # Initialize a new session
        self.session = Session()
        load_tools(self.session.system)

        # Point the UI to this active session
        self.selected_session_id = self.session.run_id

        # Patch CEO interaction callback
        self.patch_ceo_interact()

        # Log session startup message
        self.write_to_transcript(f"🚀 Started interactive session {self.session.run_id[:12]}...", "bold green")
        self.write_to_transcript("Type a message to start, or type /help for commands.\n", "dim")

        # Start worker thread
        self.run_simulation_worker()

    def patch_ceo_interact(self) -> None:
        ceo = next((p for p in self.session.participants.values() if p.__class__.__name__ == "Human"), None)
        if not ceo:
            return

        def tui_interact(*args, **kwargs):
            # Update turn state and notify UI
            self.call_from_thread(self.notify_ceo_turn)

            # Wait for user input in input box
            self.input_received_event.wait()
            self.input_received_event.clear()

            return self.last_user_input

        ceo.interact = tui_interact

    def notify_ceo_turn(self) -> None:
        self.is_ceo_turn = True
        status_lbl = self.query_one("#status_indicator", Label)
        status_lbl.update(f"🟢 {self.human_name}'s turn: Enter message or /command...")

        inp = self.query_one("#message_input", Input)
        inp.disabled = False
        inp.placeholder = "CEO: Enter message or /command..."
        inp.focus()

    @work(thread=True)
    def run_simulation_worker(self) -> None:
        try:
            self.session.run()
        except SystemExit:
            self.call_from_thread(self.action_quit)
        except Exception as e:
            import traceback
            err_msg = f"\n[Simulation Error] {e}\n{traceback.format_exc()}"
            self.call_from_thread(self.write_to_transcript, err_msg, "bold red")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return

        inp = self.query_one("#message_input", Input)
        inp.value = ""

        if text.startswith("/"):
            self.handle_slash_command(text)
            return

        if not self.interactive or not self.session:
            self.write_to_transcript("[System Warning] Read-only observability mode. Run with --interactive to chat.", "bold red")
            return

        # If it's the CEO's turn, we resume the worker thread
        if self.is_ceo_turn:
            self.is_ceo_turn = False
            self.last_user_input = text

            status_lbl = self.query_one("#status_indicator", Label)
            status_lbl.update("⏳ Agents are thinking/acting...")
            inp.disabled = True
            inp.placeholder = "Waiting for agents..."

            self.input_received_event.set()
        else:
            self.write_to_transcript("[System] Waiting for agents to finish their turns...", "bold yellow")

    def write_to_transcript(self, content: str, style: str = "bold yellow") -> None:
        log_pane = self.query_one("#transcript_log", RichLog)
        log_pane.write(Text(content, style=style))

    def action_refresh(self) -> None:
        self.refresh_data()

    def action_cycle_policy(self) -> None:
        policies = ["full", "restricted", "public-only"]
        curr_idx = policies.index(self.policy)
        self.policy = policies[(curr_idx + 1) % len(policies)]
        
        # Update title & force reload of channels and transcript
        title_widget = self.query_one("#transcript_title", Label)
        title_widget.update(f"TRANSCRIPT (Policy: {self.policy})")
        
        self.reload_channels_view()
        self.clear_transcript()
        self.load_channel_contents()
        self.clear_events()
        self.load_runtime_events()

    def refresh_data(self) -> None:
        # 1. Update session list
        sessions = self.reader.list_recent_sessions()
        if sessions != self.sessions_cache:
            self.sessions_cache = sessions
            session_list = self.query_one("#session_list", ListView)
            session_list.clear()
            for s in sessions:
                sid = s["session_id"]
                created = s["started_at"]
                display = f"{sid[:12]}... ({created[11:19]})"
                item = ListItem(Label(display))
                item.session_id = sid
                session_list.append(item)
            
            # Auto-select newest session if none selected
            if not self.selected_session_id and sessions:
                self.selected_session_id = sessions[0]["session_id"]
                session_list.index = 0

        if not self.selected_session_id:
            return

        # 2. Update token totals in footer
        p_tok, c_tok, t_tok = self.reader.get_token_totals(self.selected_session_id)
        self.SUB_TITLE = f"Session: {self.selected_session_id[:12]}... | Tokens: {p_tok} prompt, {c_tok} completion, {t_tok} total"

        # 3. Update channels list
        channels = self.reader.get_active_channels(self.selected_session_id)
        filtered_channels = self.filter_channels_by_policy(channels)
        if filtered_channels != self.channels_cache:
            self.channels_cache = filtered_channels
            self.reload_channels_view()

        # 4. Update agents list
        agents = []
        if self.interactive and self.session:
            for name, role in self.session.participants.items():
                agents.append({
                    "agent_id": name,
                    "role": role.__class__.__name__,
                    "lifecycle_state": getattr(role, "lifecycle_state", "active")
                })
        else:
            agents = self.reader.get_agents()

        if agents != self.agents_cache:
            self.agents_cache = agents
            agent_list = self.query_one("#agent_list", ListView)
            agent_list.clear()
            for a in agents:
                display = f"👤 {a['agent_id']} ({a['role']})"
                item = ListItem(Label(display))
                item.agent_name = a["agent_id"]
                agent_list.append(item)

        # 5. Load incremental channel contents
        if self.selected_channel_id is not None and self.selected_channel_type is not None:
            self.load_channel_contents()

        # 6. Load incremental events
        self.load_runtime_events()

        # 7. Update status indicator if running interactive
        if self.interactive and self.session:
            status_lbl = self.query_one("#status_indicator", Label)
            status_parts = []
            for name, role in self.session.participants.items():
                if role.__class__.__name__ == "Human":
                    if self.is_ceo_turn:
                        status_parts.append(f"🟢 [bold green]{name}[/]")
                    else:
                        status_parts.append(f"👤 {name}")
                    continue

                # Check suspension
                wu = getattr(role, "waiting_until", None)
                if wu:
                    now = datetime.now(timezone.utc)
                    rem = (wu - now).total_seconds()
                    if rem > 0:
                        status_parts.append(f"💤 {name} ({int(rem//60)}m)")
                        continue

                # Check active tool caller
                active_caller = getattr(_config, "active_tool_caller_name", None)
                if active_caller == name:
                    status_parts.append(f"⚙️ [yellow]{name}[/] (acting)")
                    continue

                # Check if currently thinking in session step
                if not self.is_ceo_turn and self.session.last_receiver and self.session.last_receiver.name == name:
                    status_parts.append(f"⏳ [cyan]{name}[/] (thinking)")
                    continue

                status_parts.append(f"⚪ {name}")

            status_lbl.update(" | ".join(status_parts))

    def filter_channels_by_policy(self, channels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered = []
        for ch in channels:
            ctype = ch["channel_type"]
            if self.policy == "public-only":
                if ctype == "org_chat":
                    filtered.append(ch)
            elif self.policy == "restricted":
                if ctype != "agent_thought":
                    filtered.append(ch)
            else:
                filtered.append(ch)
        return filtered

    def reload_channels_view(self) -> None:
        channel_list = self.query_one("#channel_list", ListView)
        channel_list.clear()
        
        for ch in self.channels_cache:
            cid = ch["channel_id"]
            ctype = ch["channel_type"]
            display_name = self.format_channel_name(ch)
            item = ListItem(Label(display_name))
            item.channel_id = cid
            item.channel_type = ctype
            channel_list.append(item)

        # Select first channel if none is currently selected
        if self.selected_channel_id is None and self.channels_cache:
            self.selected_channel_id = self.channels_cache[0]["channel_id"]
            self.selected_channel_type = self.channels_cache[0]["channel_type"]
            channel_list.index = 0
            self.clear_transcript()
            self.load_channel_contents()

    def format_channel_name(self, ch: Dict[str, Any]) -> str:
        ctype = ch["channel_type"]
        try:
            parts = json.loads(ch["participants_json"])
        except Exception:
            parts = []

        if ctype == "org_chat":
            return "# org-chat"
        elif ctype == "agent_thought":
            agent_name = parts[0] if parts else "unknown"
            return f"💭 thoughts ({agent_name})"
        elif ctype == "agent_dm":
            names = " <-> ".join(parts)
            return f"✉️ dm ({names})"
        return f"| {ctype} ({ch['channel_id']})"

    def clear_transcript(self) -> None:
        self.query_one("#transcript_log", RichLog).clear()
        self.loaded_message_ids.clear()

    def clear_events(self) -> None:
        self.loaded_event_ids.clear()
        self.events_cache.clear()
        self.query_one("#event_list", ListView).clear()
        self.query_one("#event_inspector", Static).update("")

    def load_channel_contents(self) -> None:
        if self.selected_session_id is None or self.selected_channel_id is None or self.selected_channel_type is None:
            return

        contents = self.reader.get_channel_contents(
            self.selected_session_id, self.selected_channel_id, self.selected_channel_type
        )
        
        log_pane = self.query_one("#transcript_log", RichLog)
        for item in contents:
            item_id = item["id"]
            item_type = item["type"]
            # Unique composite key for message vs thought
            key = f"{item_type}_{item_id}"
            if key in self.loaded_message_ids:
                continue
            
            # Policy-level check for message visibility
            if self.policy == "public-only" and item["visibility"] != "public":
                continue

            self.loaded_message_ids.add(key)
            formatted = self.format_transcript_item(item)
            log_pane.write(formatted)

    def format_transcript_item(self, item: Dict[str, Any]) -> Text:
        sender = item["sender"]
        content = item["content"]
        created = item["created_at"][11:19]
        
        text = Text()
        text.append(f"[{created}] ", style="dim")
        
        if item["type"] == "thought":
            text.append(f"💭 {sender} (thought): ", style="bold #808080")
            text.append(content, style="#A9A9A9 italic")
        else:
            receiver = item["receiver"]
            if receiver:
                text.append(f"✉️ {sender} -> {receiver}: ", style="bold #E066FF")
            else:
                text.append(f"🗣️ {sender}: ", style="bold #00E5FF")
            text.append(content, style="#E0E0E0")
        return text

    def load_runtime_events(self) -> None:
        if not self.selected_session_id:
            return

        events = self.reader.get_runtime_events(self.selected_session_id)
        event_list = self.query_one("#event_list", ListView)
        
        # If cache mismatch or we selected a new session, rebuild
        if len(events) < len(self.loaded_event_ids):
            event_list.clear()
            self.loaded_event_ids.clear()
            self.events_cache.clear()

        for idx, ev in enumerate(events):
            ev_id = ev["event_id"]
            if ev_id in self.loaded_event_ids:
                continue

            # Hide private event types from public-only policy
            if self.policy == "public-only" and ev["visibility"] == "private":
                continue

            self.loaded_event_ids.add(ev_id)
            self.events_cache.append(ev)
            
            etype = ev["event_type"]
            seq = ev["sequence"]
            seq_str = f"[{seq}]" if seq is not None else ""
            
            # Visual indicators based on event category
            if etype.startswith("tool_call.execute"):
                indicator = "⚙️ [green]"
            elif etype.startswith("tool_call.fail"):
                indicator = "❌ [red]"
            elif etype.startswith("tool_call"):
                indicator = "🔧 [yellow]"
            elif etype == "token_usage":
                indicator = "🪙 [cyan]"
            elif etype == "session.created":
                indicator = "🚀 [blue]"
            else:
                indicator = "🔹 [grey70]"
                
            display = Text.from_markup(f"{seq_str} {indicator}{etype}[/]")
            item = ListItem(Label(display))
            item.event_index = len(self.events_cache) - 1
            event_list.append(item)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        list_id = event.list_view.id
        if list_id == "session_list":
            selected_item = event.item
            if selected_item:
                sid = getattr(selected_item, "session_id", None)
                if sid and sid != self.selected_session_id:
                    self.selected_session_id = sid
                    self.selected_channel_id = None
                    self.selected_channel_type = None
                    
                    self.clear_transcript()
                    self.clear_events()
                    self.refresh_data()
        
        elif list_id == "channel_list":
            selected_item = event.item
            if selected_item:
                cid = getattr(selected_item, "channel_id", None)
                ctype = getattr(selected_item, "channel_type", None)
                if cid is not None and (cid != self.selected_channel_id or ctype != self.selected_channel_type):
                    self.selected_channel_id = cid
                    self.selected_channel_type = ctype
                    self.clear_transcript()
                    self.load_channel_contents()

        elif list_id == "agent_list":
            selected_item = event.item
            if selected_item:
                agent_name = getattr(selected_item, "agent_name", None)
                if agent_name:
                    self.inspect_agent(agent_name)

        elif list_id == "event_list":
            selected_item = event.item
            if selected_item:
                idx = getattr(selected_item, "event_index", None)
                if idx is not None and idx < len(self.events_cache):
                    ev = self.events_cache[idx]
                    try:
                        payload = json.loads(ev["payload_json"])
                    except Exception:
                        payload = ev["payload_json"]
                    
                    # Highlight selected event details using Pretty syntax highlighting
                    pretty_payload = Pretty(payload)
                    inspector = self.query_one("#event_inspector", Static)
                    inspector.update(pretty_payload)

    def inspect_agent(self, agent_name: str) -> None:
        inspector = self.query_one("#event_inspector", Static)
        info = []
        info.append(f"[bold cyan]👤 AGENT PERSONA: {agent_name}[/]")
        
        if self.interactive and self.session:
            role_obj = self.session.participants.get(agent_name)
            if role_obj:
                info.append(f"  [bold]Class:[/] {role_obj.__class__.__name__}")
                info.append(f"  [bold]Lifecycle State:[/] {getattr(role_obj, 'lifecycle_state', 'active')}")
                
                # Check waiting suspension
                wu = getattr(role_obj, "waiting_until", None)
                if wu:
                    now = datetime.now(timezone.utc)
                    remaining = (wu - now).total_seconds()
                    if remaining > 0:
                        info.append(f"  [bold]Status:[/] ⏳ Suspended/Napping (remains {int(remaining)}s)")
                    else:
                        info.append("  [bold]Status:[/] Active")
                else:
                    info.append("  [bold]Status:[/] Active")

                clock_state = getattr(role_obj, "runtime_clock_state", None) or getattr(_config, "clock_state", "on")
                info.append(f"  [bold]Clock State:[/] {clock_state}")
                info.append(f"  [bold]Temperature:[/] {getattr(role_obj, 'temperature', 0.7)}")
                info.append(f"  [bold]Allowed Tools:[/] {sorted(list(getattr(role_obj, 'allowed_tools', [])))}")
                
                # Active alarms
                alarms = getattr(role_obj, "alarms", [])
                if alarms:
                    info.append("  [bold]Active Alarms:[/]")
                    for alarm in alarms:
                        info.append(f"    - ID: {alarm.alarm_id} (due: {alarm.due_at})")
                        
                info.append("\n[bold]Current Preprompt Template:[/]")
                lines = role_obj.template.splitlines()
                for line in lines[:15]:
                    info.append(f"  {line}")
                if len(lines) > 15:
                    info.append("  ...")
            else:
                info.append("  Agent details not found in active session.")
        else:
            agents = self.reader.get_agents()
            a = next((x for x in agents if x["agent_id"] == agent_name), None)
            if a:
                info.append(f"  [bold]Role:[/] {a['role']}")
                info.append(f"  [bold]Lifecycle State:[/] {a['lifecycle_state']}")
                info.append(f"  [bold]Display Name:[/] {a['display_name']}")
                info.append(f"  [bold]Username:[/] {a['username']}")
                info.append(f"  [bold]Metadata JSON:[/] {a['metadata_json']}")
            else:
                info.append("  Agent details not found in database.")
                
        inspector.update("\n".join(info))

    def handle_slash_command(self, text: str) -> None:
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        log_pane = self.query_one("#transcript_log", RichLog)

        if cmd in ("/help", "/?"):
            log_pane.write(Text("\n=== TUI SLASH COMMANDS ===", style="bold yellow"))
            log_pane.write(Text("  /help, /?              - Show this help message", style="cyan"))
            log_pane.write(Text("  /personas, /agents     - List all loaded agent personas and details", style="cyan"))
            log_pane.write(Text("  /clock [@nick]         - Show global circadian clock or details for @nick", style="cyan"))
            log_pane.write(Text("  /session               - Show session ID and token usage statistics", style="cyan"))
            log_pane.write(Text("  /clear                 - Clear the transcript view log", style="cyan"))
            log_pane.write(Text("  /wait <mins> [@nick]   - Suspend/nap @nick or last active agent for <mins>", style="cyan"))
            log_pane.write(Text("  /quit, /exit           - Exit the application", style="cyan"))
            log_pane.write(Text("==========================\n", style="bold yellow"))

        elif cmd in ("/quit", "/exit"):
            self.action_quit()

        elif cmd == "/clear":
            self.clear_transcript()

        elif cmd in ("/personas", "/agents"):
            log_pane.write(Text("\n=== PERSONAS / AGENTS ===", style="bold yellow"))
            if self.interactive and self.session:
                for name, role in self.session.participants.items():
                    clock_state = getattr(role, "runtime_clock_state", None) or getattr(_config, "clock_state", "on")
                    wu = getattr(role, "waiting_until", None)
                    suspended_str = f"Suspended until {wu.strftime('%H:%M:%S')}" if wu else "Active"
                    log_pane.write(Text(f"👤 {name} ({role.__class__.__name__})", style="bold green"))
                    log_pane.write(Text(f"   Clock: {clock_state} | Status: {suspended_str}", style="dim"))
                    log_pane.write(Text(f"   Preprompt: {role.template[:120]}...", style="italic grey"))
            else:
                agents = self.reader.get_agents()
                for a in agents:
                    log_pane.write(Text(f"👤 {a['agent_id']} ({a['role']})", style="bold green"))
                    log_pane.write(Text(f"   Status: {a['lifecycle_state']}", style="dim"))
            log_pane.write(Text("==========================\n", style="bold yellow"))

        elif cmd == "/clock":
            log_pane.write(Text("\n=== CLOCK STATE ===", style="bold yellow"))
            target_agent = args.strip().lstrip("@")
            if target_agent:
                # Find details for a specific agent
                if self.interactive and self.session:
                    role = self.session.participants.get(target_agent)
                    if role:
                        clock_state = getattr(role, "runtime_clock_state", None) or getattr(_config, "clock_state", "on")
                        log_pane.write(Text(f"  Agent: {target_agent}", style="bold green"))
                        log_pane.write(Text(f"  Clock state: {clock_state.upper()}", style="cyan"))
                        log_pane.write(Text(f"  Status: {getattr(role, 'lifecycle_state', 'active')}", style="cyan"))
                    else:
                        log_pane.write(Text(f"  Agent '{target_agent}' not found in active session.", style="red"))
                else:
                    agents = self.reader.get_agents()
                    a = next((x for x in agents if x["agent_id"] == target_agent), None)
                    if a:
                        log_pane.write(Text(f"  Agent: {target_agent}", style="bold green"))
                        log_pane.write(Text(f"  Lifecycle state: {a['lifecycle_state']}", style="cyan"))
                    else:
                        log_pane.write(Text(f"  Agent '{target_agent}' not found in database.", style="red"))
            else:
                # Global clock and all participants' current states
                effective_clock = getattr(_config, "clock_state", "on")
                break_sched = getattr(_config, "break_schedule", [])
                log_pane.write(Text(f"  Global clock state: {effective_clock.upper()}", style="cyan"))
                log_pane.write(Text(f"  Break schedule: {break_sched}", style="cyan"))
                if self.interactive and self.session:
                    log_pane.write(Text("\n  Agent clock states:", style="dim"))
                    for name, role in self.session.participants.items():
                        if role.__class__.__name__ != "Human":
                            clock_state = getattr(role, "runtime_clock_state", None) or getattr(_config, "clock_state", "on")
                            log_pane.write(Text(f"    - {name}: {clock_state.upper()}", style="cyan"))
            log_pane.write(Text("====================\n", style="bold yellow"))

        elif cmd == "/session":
            log_pane.write(Text("\n=== SESSION INFO ===", style="bold yellow"))
            log_pane.write(Text(f"  Session ID: {self.selected_session_id}", style="cyan"))
            if self.selected_session_id:
                p, c, t = self.reader.get_token_totals(self.selected_session_id)
                log_pane.write(Text(f"  Token usage: Prompt={p}, Completion={c}, Total={t}", style="cyan"))
            log_pane.write(Text("====================\n", style="bold yellow"))

        elif cmd == "/wait":
            if not self.interactive or not self.session:
                log_pane.write(Text("[System] Wait command only supported in interactive mode.", style="red"))
                return

            arg_parts = args.strip().split(maxsplit=1)
            mins = 10
            target_agent = None
            if arg_parts:
                try:
                    mins = int(arg_parts[0])
                    if len(arg_parts) > 1:
                        target_agent = arg_parts[1].lstrip("@")
                except ValueError:
                    target_agent = arg_parts[0].lstrip("@")

            if target_agent:
                receiver = self.session.participants.get(target_agent)
                if not receiver:
                    log_pane.write(Text(f"[System] Agent '{target_agent}' not found in active session.", style="red"))
                    return
            else:
                receiver = self.session.last_receiver

            if receiver:
                from datetime import timedelta
                receiver.waiting_until = datetime.now(timezone.utc) + timedelta(minutes=mins)
                log_pane.write(Text(f"[System] Manually suspended {receiver.name} for {mins} minutes.", style="yellow"))
                self.refresh_data()
            else:
                log_pane.write(Text("[System] No active receiver found to suspend. Usage: /wait <minutes> [@nick]", style="red"))

        else:
            log_pane.write(Text(f"[System] Unknown command: {cmd}. Type /help for commands.", style="red"))
