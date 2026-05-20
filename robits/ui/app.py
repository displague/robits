"""Textual TUI application for Robits simulation observability."""
import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from rich.pretty import Pretty
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Footer, Header, Label, ListItem, ListView, RichLog, Static


class RobitsDbReader:
    """Reads observability data from the SQLite database."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _get_conn(self) -> sqlite3.Connection:
        # Open in read-only mode to prevent write locks during simulation execution
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
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
            if channel_type == "agent_thoughts":
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
    """

    def __init__(self, db_path: str, session_id: Optional[str] = None, policy: str = "full"):
        super().__init__()
        self.reader = RobitsDbReader(db_path)
        self.selected_session_id = session_id
        self.policy = policy
        self.selected_channel_id: Optional[int] = None
        self.selected_channel_type: Optional[str] = None

        # Change-tracking states to optimize updates without full clears
        self.sessions_cache: List[Dict[str, Any]] = []
        self.channels_cache: List[Dict[str, Any]] = []
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
            with Vertical(id="main_panel"):
                yield Label("TRANSCRIPT (Policy: " + self.policy + ")", id="transcript_title")
                yield RichLog(id="transcript_log", max_lines=1000)
            with Vertical(id="event_panel"):
                yield Label("RUNTIME EVENTS", classes="panel-title")
                yield ListView(id="event_list")
                yield Label("EVENT INSPECTOR", classes="panel-title")
                yield Static(id="event_inspector")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_data()
        self.set_interval(1.0, self.refresh_data)

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

        # 4. Load incremental channel contents
        if self.selected_channel_id is not None and self.selected_channel_type is not None:
            self.load_channel_contents()

        # 5. Load incremental events
        self.load_runtime_events()

    def filter_channels_by_policy(self, channels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered = []
        for ch in channels:
            ctype = ch["channel_type"]
            if self.policy == "public-only":
                if ctype == "org_chat":
                    filtered.append(ch)
            elif self.policy == "restricted":
                if ctype != "agent_thoughts":
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
        elif ctype == "agent_thoughts":
            agent_name = parts[0] if parts else "unknown"
            return f"💭 thoughts ({agent_name})"
        elif ctype == "agent_dms":
            names = " <-> ".join(parts)
            return f"✉️ dm ({names})"
        return f"| {ctype} ({ch['channel_id']})"

    def clear_transcript(self) -> None:
        self.query_one("#transcript_log", RichLog).clear()
        self.loaded_message_ids.clear()

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
                    self.loaded_event_ids.clear()
                    self.events_cache.clear()
                    self.query_one("#event_list", ListView).clear()
                    self.query_one("#event_inspector", Static).update("")
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
