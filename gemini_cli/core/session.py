"""
Session management for Gemini CLI
Handles conversation state, history, and session persistence
"""

import json
import uuid
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import logging
import pickle
import gzip
import hashlib

from .exceptions import GeminiCLIError


class SessionManager:
    """Manages CLI sessions and conversation state."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Session storage
        self.sessions_dir = Path.home() / ".gemini" / "sessions"
        self.conversations_dir = Path.home() / ".gemini" / "conversations"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.conversations_dir.mkdir(parents=True, exist_ok=True)

        # Current session
        self.current_session_id = None
        self.current_session = None

        # Session limits
        self.max_sessions = 100
        self.max_conversation_age_days = 30
        self.max_message_history = 1000

        # Auto-cleanup on init
        asyncio.create_task(self._cleanup_old_sessions())

    def create_session(self) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())

        session_data = {
            "id": session_id,
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "message_count": 0,
            "tool_calls": 0,
            "total_tokens": 0,
            "conversation": [],
            "metadata": {
                "version": "1.0.0",
                "platform": self._get_platform_info()
            }
        }

        self.current_session_id = session_id
        self.current_session = session_data

        # Save session
        self._save_session(session_data)

        self.logger.info(f"Created new session: {session_id}")
        return session_id

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load an existing session."""
        try:
            session_file = self.sessions_dir / f"{session_id}.json"

            if not session_file.exists():
                self.logger.warning(f"Session not found: {session_id}")
                return None

            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)

            # Update last active
            session_data["last_active"] = datetime.now().isoformat()

            self.current_session_id = session_id
            self.current_session = session_data

            # Save updated session
            self._save_session(session_data)

            self.logger.info(f"Loaded session: {session_id}")
            return session_data

        except Exception as e:
            self.logger.error(f"Failed to load session {session_id}: {e}")
            return None

    def save_session(self, session_id: str, conversation: List[Dict[str, Any]]):
        """Save session with conversation data."""
        try:
            if not self.current_session:
                self.logger.warning("No current session to save")
                return

            # Update session data
            self.current_session["conversation"] = conversation
            self.current_session["last_active"] = datetime.now().isoformat()
            self.current_session["message_count"] = len(conversation)

            # Calculate total tokens
            total_tokens = 0
            for message in conversation:
                if isinstance(message, dict) and "usage" in message:
                    usage = message["usage"]
                    total_tokens += usage.get("total_tokens", 0)

            self.current_session["total_tokens"] = total_tokens

            # Save to file
            self._save_session(self.current_session)

            self.logger.debug(f"Saved session {session_id} with {len(conversation)} messages")

        except Exception as e:
            self.logger.error(f"Failed to save session {session_id}: {e}")

    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent sessions."""
        try:
            sessions = []

            for session_file in self.sessions_dir.glob("*.json"):
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)

                    # Extract summary info
                    session_info = {
                        "id": session_data["id"],
                        "created_at": session_data["created_at"],
                        "last_active": session_data["last_active"],
                        "message_count": session_data.get("message_count", 0),
                        "total_tokens": session_data.get("total_tokens", 0)
                    }

                    sessions.append(session_info)

                except Exception as e:
                    self.logger.warning(f"Failed to read session file {session_file}: {e}")
                    continue

            # Sort by last active (most recent first)
            sessions.sort(key=lambda x: x["last_active"], reverse=True)

            return sessions[:limit]

        except Exception as e:
            self.logger.error(f"Failed to list sessions: {e}")
            return []

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        try:
            session_file = self.sessions_dir / f"{session_id}.json"

            if session_file.exists():
                session_file.unlink()
                self.logger.info(f"Deleted session: {session_id}")

                # Clear current session if it's the one being deleted
                if self.current_session_id == session_id:
                    self.current_session_id = None
                    self.current_session = None

                return True
            else:
                self.logger.warning(f"Session not found for deletion: {session_id}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    def save_conversation(self, conversation: List[Dict[str, Any]], tag: str):
        """Save a conversation with a specific tag."""
        try:
            # Create a safe filename from tag
            safe_tag = self._sanitize_filename(tag)
            conversation_file = self.conversations_dir / f"{safe_tag}.json"

            conversation_data = {
                "tag": tag,
                "created_at": datetime.now().isoformat(),
                "conversation": conversation,
                "message_count": len(conversation),
                "checksum": self._calculate_conversation_checksum(conversation)
            }

            with open(conversation_file, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Saved conversation with tag: {tag}")

        except Exception as e:
            self.logger.error(f"Failed to save conversation {tag}: {e}")
            raise GeminiCLIError(f"Failed to save conversation: {e}")

    def load_conversation(self, tag: str) -> Optional[List[Dict[str, Any]]]:
        """Load a conversation by tag."""
        try:
            safe_tag = self._sanitize_filename(tag)
            conversation_file = self.conversations_dir / f"{safe_tag}.json"

            if not conversation_file.exists():
                self.logger.warning(f"Conversation not found: {tag}")
                return None

            with open(conversation_file, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)

            conversation = conversation_data.get("conversation", [])

            # Verify checksum if available
            stored_checksum = conversation_data.get("checksum")
            if stored_checksum:
                current_checksum = self._calculate_conversation_checksum(conversation)
                if stored_checksum != current_checksum:
                    self.logger.warning(f"Conversation checksum mismatch for {tag}")

            self.logger.info(f"Loaded conversation: {tag}")
            return conversation

        except Exception as e:
            self.logger.error(f"Failed to load conversation {tag}: {e}")
            return None

    def list_saved_conversations(self) -> List[str]:
        """List all saved conversation tags."""
        try:
            tags = []

            for conversation_file in self.conversations_dir.glob("*.json"):
                try:
                    with open(conversation_file, 'r', encoding='utf-8') as f:
                        conversation_data = json.load(f)

                    tag = conversation_data.get("tag", conversation_file.stem)
                    tags.append(tag)

                except Exception as e:
                    self.logger.warning(f"Failed to read conversation file {conversation_file}: {e}")
                    continue

            return sorted(tags)

        except Exception as e:
            self.logger.error(f"Failed to list conversations: {e}")
            return []

    def delete_conversation(self, tag: str) -> bool:
        """Delete a saved conversation."""
        try:
            safe_tag = self._sanitize_filename(tag)
            conversation_file = self.conversations_dir / f"{safe_tag}.json"

            if conversation_file.exists():
                conversation_file.unlink()
                self.logger.info(f"Deleted conversation: {tag}")
                return True
            else:
                self.logger.warning(f"Conversation not found for deletion: {tag}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to delete conversation {tag}: {e}")
            return False

    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about all sessions."""
        try:
            sessions = self.list_sessions(limit=1000)  # Get all sessions

            if not sessions:
                return {
                    "total_sessions": 0,
                    "total_messages": 0,
                    "total_tokens": 0,
                    "average_session_length": 0,
                    "oldest_session": None,
                    "newest_session": None
                }

            total_messages = sum(s.get("message_count", 0) for s in sessions)
            total_tokens = sum(s.get("total_tokens", 0) for s in sessions)

            # Calculate average session length
            message_counts = [s.get("message_count", 0) for s in sessions if s.get("message_count", 0) > 0]
            avg_length = sum(message_counts) / len(message_counts) if message_counts else 0

            # Find oldest and newest sessions
            sessions_with_dates = [s for s in sessions if s.get("created_at")]
            oldest = min(sessions_with_dates, key=lambda x: x["created_at"]) if sessions_with_dates else None
            newest = max(sessions_with_dates, key=lambda x: x["created_at"]) if sessions_with_dates else None

            return {
                "total_sessions": len(sessions),
                "total_messages": total_messages,
                "total_tokens": total_tokens,
                "average_session_length": round(avg_length, 1),
                "oldest_session": oldest["created_at"] if oldest else None,
                "newest_session": newest["created_at"] if newest else None
            }

        except Exception as e:
            self.logger.error(f"Failed to get session stats: {e}")
            return {}

    def export_session(self, session_id: str, format: str = "json") -> Optional[bytes]:
        """Export a session in various formats."""
        try:
            session_data = self.load_session(session_id)
            if not session_data:
                return None

            if format.lower() == "json":
                export_data = json.dumps(session_data, indent=2, ensure_ascii=False)
                return export_data.encode('utf-8')

            elif format.lower() == "compressed":
                json_data = json.dumps(session_data, ensure_ascii=False)
                return gzip.compress(json_data.encode('utf-8'))

            elif format.lower() == "pickle":
                return pickle.dumps(session_data)

            else:
                raise ValueError(f"Unsupported export format: {format}")

        except Exception as e:
            self.logger.error(f"Failed to export session {session_id}: {e}")
            return None

    def import_session(self, data: bytes, format: str = "json") -> Optional[str]:
        """Import a session from exported data."""
        try:
            if format.lower() == "json":
                session_data = json.loads(data.decode('utf-8'))

            elif format.lower() == "compressed":
                json_data = gzip.decompress(data).decode('utf-8')
                session_data = json.loads(json_data)

            elif format.lower() == "pickle":
                session_data = pickle.loads(data)

            else:
                raise ValueError(f"Unsupported import format: {format}")

            # Generate new session ID
            new_session_id = str(uuid.uuid4())
            session_data["id"] = new_session_id
            session_data["imported_at"] = datetime.now().isoformat()

            # Save imported session
            self._save_session(session_data)

            self.logger.info(f"Imported session as: {new_session_id}")
            return new_session_id

        except Exception as e:
            self.logger.error(f"Failed to import session: {e}")
            return None

    def compress_conversation(self, conversation: List[Dict[str, Any]],
                            keep_recent: int = 5) -> List[Dict[str, Any]]:
        """Compress conversation by summarizing older messages."""
        if len(conversation) <= keep_recent * 2:
            return conversation

        # Keep first message (usually system prompt), recent messages, and create summary
        first_message = conversation[0] if conversation else None
        recent_messages = conversation[-keep_recent:] if keep_recent > 0 else []
        middle_messages = conversation[1:-keep_recent] if keep_recent > 0 else conversation[1:]

        if not middle_messages:
            return conversation

        # Create summary of middle messages
        summary_text = self._create_conversation_summary(middle_messages)

        summary_message = {
            "role": "system",
            "content": f"[Conversation summary: {summary_text}]",
            "timestamp": datetime.now().isoformat(),
            "is_summary": True,
            "original_message_count": len(middle_messages)
        }

        # Reconstruct conversation
        compressed = []
        if first_message:
            compressed.append(first_message)
        compressed.append(summary_message)
        compressed.extend(recent_messages)

        self.logger.info(f"Compressed conversation from {len(conversation)} to {len(compressed)} messages")
        return compressed

    async def _cleanup_old_sessions(self):
        """Clean up old sessions and conversations."""
        try:
            current_time = datetime.now()
            cutoff_date = current_time - timedelta(days=self.max_conversation_age_days)

            # Clean up old sessions
            session_files = list(self.sessions_dir.glob("*.json"))

            if len(session_files) > self.max_sessions:
                # Sort by modification time and remove oldest
                session_files.sort(key=lambda x: x.stat().st_mtime)
                files_to_remove = session_files[:-self.max_sessions]

                for file_path in files_to_remove:
                    try:
                        file_path.unlink()
                        self.logger.debug(f"Cleaned up old session: {file_path.stem}")
                    except Exception as e:
                        self.logger.warning(f"Failed to remove old session {file_path}: {e}")

            # Clean up old conversations
            for conversation_file in self.conversations_dir.glob("*.json"):
                try:
                    file_time = datetime.fromtimestamp(conversation_file.stat().st_mtime)

                    if file_time < cutoff_date:
                        conversation_file.unlink()
                        self.logger.debug(f"Cleaned up old conversation: {conversation_file.stem}")

                except Exception as e:
                    self.logger.warning(f"Failed to check/remove conversation {conversation_file}: {e}")

            self.logger.debug("Session cleanup completed")

        except Exception as e:
            self.logger.error(f"Session cleanup failed: {e}")

    def _save_session(self, session_data: Dict[str, Any]):
        """Save session data to file."""
        session_id = session_data["id"]
        session_file = self.sessions_dir / f"{session_id}.json"

        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize a string for use as a filename."""
        # Remove/replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')

        # Limit length
        max_length = 100
        if len(filename) > max_length:
            filename = filename[:max_length]

        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')

        # Ensure it's not empty
        if not filename:
            filename = "unnamed"

        return filename

    def _calculate_conversation_checksum(self, conversation: List[Dict[str, Any]]) -> str:
        """Calculate checksum for conversation integrity."""
        # Create a stable string representation
        conversation_str = json.dumps(conversation, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(conversation_str.encode('utf-8')).hexdigest()[:16]

    def _create_conversation_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Create a summary of conversation messages."""
        # Simple summary by extracting key information
        summary_parts = []

        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]

        if user_messages:
            # Extract topics/keywords from user messages
            user_content = " ".join([msg.get("content", "") for msg in user_messages[:3]])
            summary_parts.append(f"User discussed: {user_content[:100]}...")

        if assistant_messages:
            assistant_content = " ".join([msg.get("content", "") for msg in assistant_messages[:2]])
            summary_parts.append(f"Assistant provided: {assistant_content[:100]}...")

        summary_parts.append(f"Total messages: {len(messages)}")

        return " | ".join(summary_parts)

    def _get_platform_info(self) -> Dict[str, str]:
        """Get platform information for session metadata."""
        import platform
        import sys

        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
            "hostname": platform.node()
        }

    def get_current_session_id(self) -> Optional[str]:
        """Get the current session ID."""
        return self.current_session_id

    def get_current_session(self) -> Optional[Dict[str, Any]]:
        """Get the current session data."""
        return self.current_session

    def update_session_metadata(self, metadata: Dict[str, Any]):
        """Update metadata for the current session."""
        if self.current_session:
            self.current_session.setdefault("metadata", {}).update(metadata)
            self._save_session(self.current_session)

    def record_tool_call(self, tool_name: str, success: bool, execution_time: float):
        """Record a tool call in the current session."""
        if self.current_session:
            self.current_session["tool_calls"] = self.current_session.get("tool_calls", 0) + 1

            # Track tool call details
            tool_calls_data = self.current_session.setdefault("tool_calls_data", [])
            tool_calls_data.append({
                "tool_name": tool_name,
                "success": success,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            })

            # Limit tool call history
            if len(tool_calls_data) > 100:
                self.current_session["tool_calls_data"] = tool_calls_data[-100:]

            self._save_session(self.current_session)