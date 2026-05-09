from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ToolProposalStore:
    """Small JSON-backed registry for managed tool proposal lifecycle state."""

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self._proposals = {}
        if self.path is not None and self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"Could not load tool proposal store '{self.path}': {error}") from error
            for proposal in data.get("proposals", []):
                self._proposals[proposal["proposal_id"]] = proposal

    def _save(self):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"proposals": sorted(self._proposals.values(), key=lambda item: item["created_at"])}
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def clear(self):
        self._proposals.clear()
        self._save()

    def create(
        self,
        requested_by,
        tool_name,
        description,
        action="create",
        parameters=None,
        required_capabilities=None,
        owner_capability=None,
        safety_notes="",
        implementation_notes="",
        code="",
    ):
        now = _utc_now()
        proposal = {
            "proposal_id": f"tool-proposal-{uuid4()}",
            "requested_by": requested_by,
            "tool_name": tool_name,
            "action": action,
            "description": description,
            "parameters": deepcopy(parameters or {"type": "object", "properties": {}, "required": []}),
            "required_capabilities": deepcopy(list(required_capabilities or [])),
            "owner_capability": owner_capability,
            "safety_notes": safety_notes or "",
            "implementation_notes": implementation_notes or "",
            "code": code or "",
            "status": "proposed",
            "approver": None,
            "rejection_reason": "",
            "rollout_notes": "",
            "granted_roles": [],
            "created_at": now,
            "updated_at": now,
        }
        self._proposals[proposal["proposal_id"]] = proposal
        self._save()
        return deepcopy(proposal)

    def get(self, proposal_id):
        proposal = self._proposals.get(proposal_id)
        return deepcopy(proposal) if proposal else None

    def list(self, status=None):
        proposals = sorted(self._proposals.values(), key=lambda item: item["created_at"])
        if status:
            proposals = [proposal for proposal in proposals if proposal["status"] == status]
        return deepcopy(proposals)

    def update(self, proposal_id, **changes):
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            return None
        proposal.update(deepcopy(changes))
        proposal["updated_at"] = _utc_now()
        self._save()
        return deepcopy(proposal)
