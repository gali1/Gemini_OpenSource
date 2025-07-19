"""
History management for Gemini CLI
Provides command history, search, and persistence functionality
"""

import json
import os
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime
import logging


class HistoryEntry:
    """Represents a single history entry."""

    def __init__(self, command: str, timestamp: Optional[datetime] = None, session_id: Optional[str] = None):
        self.command = command
        self.timestamp = timestamp or datetime.now()
        self.session_id = session_id
        self.metadata: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "command": self.command,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HistoryEntry':
        """Create from dictionary."""
        entry = cls(
            command=data["command"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            session_id=data.get("session_id")
        )
        entry.metadata = data.get("metadata", {})
        return entry


class HistoryManager:
    """Manages command history with persistence and search capabilities."""

    def __init__(self, max_entries: int = 10000):
        self.max_entries = max_entries
        self.entries: List[HistoryEntry] = []
        self.current_session_id: Optional[str] = None
        self.logger = logging.getLogger(__name__)

        # History file location
        self.history_dir = Path.home() / ".gemini"
        self.history_file = self.history_dir / "history.json"

        # Ensure directory exists
        self.history_dir.mkdir(exist_ok=True)

        # Load existing history
        self.load_history()

    def set_session_id(self, session_id: str):
        """Set the current session ID for new entries."""
        self.current_session_id = session_id

    def add_entry(self, command: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a new command to history."""
        # Don't add empty commands or duplicates of the last command
        if not command.strip():
            return

        if self.entries and self.entries[-1].command == command:
            return

        entry = HistoryEntry(
            command=command,
            timestamp=datetime.now(),
            session_id=self.current_session_id
        )

        if metadata:
            entry.metadata.update(metadata)

        self.entries.append(entry)

        # Maintain size limit
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

        # Auto-save periodically
        if len(self.entries) % 10 == 0:
            self.save_history()

    def get_recent(self, count: int = 10) -> List[HistoryEntry]:
        """Get recent history entries."""
        return self.entries[-count:] if self.entries else []

    def get_all(self) -> List[HistoryEntry]:
        """Get all history entries."""
        return self.entries.copy()

    def search(self, query: str, limit: int = 50) -> List[HistoryEntry]:
        """Search history entries by command text."""
        query_lower = query.lower()
        matches = []

        # Search from most recent to oldest
        for entry in reversed(self.entries):
            if query_lower in entry.command.lower():
                matches.append(entry)
                if len(matches) >= limit:
                    break

        return matches

    def search_by_prefix(self, prefix: str, limit: int = 10) -> List[HistoryEntry]:
        """Search history entries that start with the given prefix."""
        prefix_lower = prefix.lower()
        matches = []

        # Search from most recent to oldest
        for entry in reversed(self.entries):
            if entry.command.lower().startswith(prefix_lower):
                matches.append(entry)
                if len(matches) >= limit:
                    break

        return matches

    def get_by_session(self, session_id: str) -> List[HistoryEntry]:
        """Get history entries for a specific session."""
        return [entry for entry in self.entries if entry.session_id == session_id]

    def get_unique_commands(self, limit: int = 100) -> List[str]:
        """Get unique commands from history (most recent first)."""
        seen = set()
        unique = []

        for entry in reversed(self.entries):
            if entry.command not in seen:
                seen.add(entry.command)
                unique.append(entry.command)
                if len(unique) >= limit:
                    break

        return unique

    def get_statistics(self) -> Dict[str, Any]:
        """Get history statistics."""
        if not self.entries:
            return {
                "total_entries": 0,
                "unique_commands": 0,
                "sessions": 0,
                "date_range": None,
                "most_used_commands": []
            }

        # Count command usage
        command_counts: Dict[str, int] = {}
        sessions = set()

        for entry in self.entries:
            command_counts[entry.command] = command_counts.get(entry.command, 0) + 1
            if entry.session_id:
                sessions.add(entry.session_id)

        # Sort by usage
        most_used = sorted(command_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Date range
        oldest = min(entry.timestamp for entry in self.entries)
        newest = max(entry.timestamp for entry in self.entries)

        return {
            "total_entries": len(self.entries),
            "unique_commands": len(command_counts),
            "sessions": len(sessions),
            "date_range": {
                "oldest": oldest.isoformat(),
                "newest": newest.isoformat()
            },
            "most_used_commands": [{"command": cmd, "count": count} for cmd, count in most_used]
        }

    def clear_history(self):
        """Clear all history entries."""
        self.entries.clear()
        self.save_history()

    def remove_entry(self, index: int) -> bool:
        """Remove a specific history entry by index."""
        try:
            if 0 <= index < len(self.entries):
                del self.entries[index]
                return True
            return False
        except IndexError:
            return False

    def remove_by_pattern(self, pattern: str) -> int:
        """Remove entries matching a pattern. Returns count of removed entries."""
        import re

        try:
            regex = re.compile(pattern, re.IGNORECASE)
            original_count = len(self.entries)
            self.entries = [entry for entry in self.entries if not regex.search(entry.command)]
            removed_count = original_count - len(self.entries)

            if removed_count > 0:
                self.save_history()

            return removed_count
        except re.error:
            # Invalid regex pattern
            return 0

    def load_history(self):
        """Load history from file."""
        try:
            if not self.history_file.exists():
                return

            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, dict) and "entries" in data:
                # New format with metadata
                entries_data = data["entries"]
            elif isinstance(data, list):
                # Old format - list of strings or entry dicts
                entries_data = data
            else:
                self.logger.warning("Invalid history file format")
                return

            self.entries = []
            for entry_data in entries_data:
                try:
                    if isinstance(entry_data, str):
                        # Old format - just command string
                        entry = HistoryEntry(entry_data)
                    elif isinstance(entry_data, dict):
                        # New format - full entry dict
                        entry = HistoryEntry.from_dict(entry_data)
                    else:
                        continue

                    self.entries.append(entry)
                except Exception as e:
                    self.logger.warning(f"Failed to load history entry: {e}")
                    continue

            self.logger.info(f"Loaded {len(self.entries)} history entries")

        except Exception as e:
            self.logger.warning(f"Failed to load history: {e}")
            self.entries = []

    def save_history(self):
        """Save history to file."""
        try:
            # Prepare data for saving
            history_data = {
                "version": "1.0",
                "saved_at": datetime.now().isoformat(),
                "entries": [entry.to_dict() for entry in self.entries]
            }

            # Write to temporary file first, then rename for atomic operation
            temp_file = self.history_file.with_suffix('.tmp')

            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)

            # Atomic rename
            temp_file.replace(self.history_file)

            self.logger.debug(f"Saved {len(self.entries)} history entries")

        except Exception as e:
            self.logger.error(f"Failed to save history: {e}")

    def export_history(self, output_file: Path, format: str = "json") -> bool:
        """Export history to a file in various formats."""
        try:
            if format == "json":
                data = {
                    "exported_at": datetime.now().isoformat(),
                    "total_entries": len(self.entries),
                    "entries": [entry.to_dict() for entry in self.entries]
                }

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

            elif format == "txt":
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"# Gemini CLI History Export\n")
                    f.write(f"# Exported at: {datetime.now().isoformat()}\n")
                    f.write(f"# Total entries: {len(self.entries)}\n\n")

                    for entry in self.entries:
                        f.write(f"# {entry.timestamp.isoformat()}\n")
                        f.write(f"{entry.command}\n\n")

            elif format == "csv":
                import csv

                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp", "command", "session_id"])

                    for entry in self.entries:
                        writer.writerow([
                            entry.timestamp.isoformat(),
                            entry.command,
                            entry.session_id or ""
                        ])

            else:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Failed to export history: {e}")
            return False

    def import_history(self, input_file: Path, format: str = "json", merge: bool = True) -> bool:
        """Import history from a file."""
        try:
            if not input_file.exists():
                return False

            imported_entries = []

            if format == "json":
                with open(input_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if isinstance(data, dict) and "entries" in data:
                    entries_data = data["entries"]
                elif isinstance(data, list):
                    entries_data = data
                else:
                    return False

                for entry_data in entries_data:
                    try:
                        if isinstance(entry_data, str):
                            entry = HistoryEntry(entry_data)
                        elif isinstance(entry_data, dict):
                            entry = HistoryEntry.from_dict(entry_data)
                        else:
                            continue

                        imported_entries.append(entry)
                    except Exception:
                        continue

            elif format == "txt":
                with open(input_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        imported_entries.append(HistoryEntry(line))

            else:
                return False

            if merge:
                # Merge with existing history
                existing_commands = {entry.command for entry in self.entries}
                new_entries = [entry for entry in imported_entries if entry.command not in existing_commands]
                self.entries.extend(new_entries)
            else:
                # Replace existing history
                self.entries = imported_entries

            # Maintain size limit
            if len(self.entries) > self.max_entries:
                self.entries = self.entries[-self.max_entries:]

            self.save_history()
            return True

        except Exception as e:
            self.logger.error(f"Failed to import history: {e}")
            return False

    def compact_history(self, keep_recent: int = 1000) -> int:
        """Compact history by removing old duplicate commands."""
        if len(self.entries) <= keep_recent:
            return 0

        # Keep the most recent entries as-is
        recent_entries = self.entries[-keep_recent:]
        older_entries = self.entries[:-keep_recent]

        # Remove duplicates from older entries, keeping the most recent occurrence
        seen_commands = set()
        compacted_older = []

        for entry in reversed(older_entries):
            if entry.command not in seen_commands:
                seen_commands.add(entry.command)
                compacted_older.append(entry)

        # Combine compacted older entries with recent entries
        compacted_older.reverse()  # Restore chronological order
        original_count = len(self.entries)
        self.entries = compacted_older + recent_entries

        removed_count = original_count - len(self.entries)

        if removed_count > 0:
            self.save_history()

        return removed_count