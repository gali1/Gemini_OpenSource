"""
Statistics tracking for Gemini CLI
Provides session statistics, usage tracking, and performance metrics
"""

import time
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import logging
from collections import defaultdict, deque


class StatsTracker:
    """Tracks various statistics during CLI usage."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id
        self.session_start_time = time.time()
        self.logger = logging.getLogger(__name__)

        # Basic counters
        self.total_prompts = 0
        self.total_api_calls = 0
        self.total_tool_calls = 0
        self.successful_tool_calls = 0
        self.failed_tool_calls = 0

        # Token usage
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0

        # Timing data
        self.response_times: deque = deque(maxlen=100)  # Keep last 100 response times
        self.tool_execution_times: deque = deque(maxlen=100)

        # Detailed tracking
        self.tool_usage_count: Dict[str, int] = defaultdict(int)
        self.tool_success_count: Dict[str, int] = defaultdict(int)
        self.tool_failure_count: Dict[str, int] = defaultdict(int)
        self.tool_execution_time_by_tool: Dict[str, List[float]] = defaultdict(list)

        # Error tracking
        self.error_count = 0
        self.error_types: Dict[str, int] = defaultdict(int)

        # File operation tracking
        self.files_read = 0
        self.files_written = 0
        self.bytes_read = 0
        self.bytes_written = 0

        # Memory usage tracking
        self.memory_snapshots: List[Dict[str, Any]] = []

        # Session metrics
        self.commands_per_session: List[int] = []
        self.session_durations: List[float] = []

        # Performance thresholds
        self.slow_response_threshold = 5.0  # seconds
        self.slow_responses = 0

        # Feature usage
        self.features_used: Set[str] = set()

        # Load persistent stats if available
        self.persistent_stats_file = Path.home() / ".gemini" / "stats.json"
        self.persistent_stats = self._load_persistent_stats()

    def record_user_prompt(self, prompt: str):
        """Record a user prompt."""
        self.total_prompts += 1
        self.features_used.add("user_prompts")

        # Track prompt characteristics
        prompt_length = len(prompt)
        self.persistent_stats["prompt_lengths"].append(prompt_length)

        # Keep only recent data
        if len(self.persistent_stats["prompt_lengths"]) > 1000:
            self.persistent_stats["prompt_lengths"] = self.persistent_stats["prompt_lengths"][-1000:]

    def record_api_request(self, usage: Dict[str, Any], response_time: Optional[float] = None):
        """Record an API request with token usage."""
        self.total_api_calls += 1

        # Track token usage
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens += total_tokens

        # Track response time
        if response_time is not None:
            self.response_times.append(response_time)
            if response_time > self.slow_response_threshold:
                self.slow_responses += 1

            # Update persistent stats
            self.persistent_stats["api_response_times"].append(response_time)
            if len(self.persistent_stats["api_response_times"]) > 1000:
                self.persistent_stats["api_response_times"] = self.persistent_stats["api_response_times"][-1000:]

        self.features_used.add("api_requests")

    def record_tool_call(self, tool_name: str, success: bool, execution_time: float):
        """Record a tool call."""
        self.total_tool_calls += 1
        self.tool_usage_count[tool_name] += 1

        if success:
            self.successful_tool_calls += 1
            self.tool_success_count[tool_name] += 1
        else:
            self.failed_tool_calls += 1
            self.tool_failure_count[tool_name] += 1

        # Track execution time
        self.tool_execution_times.append(execution_time)
        self.tool_execution_time_by_tool[tool_name].append(execution_time)

        # Keep only recent timing data per tool
        if len(self.tool_execution_time_by_tool[tool_name]) > 100:
            self.tool_execution_time_by_tool[tool_name] = self.tool_execution_time_by_tool[tool_name][-100:]

        self.features_used.add("tool_calls")
        self.features_used.add(f"tool_{tool_name}")

    def record_error(self, error_type: str, error_message: str):
        """Record an error occurrence."""
        self.error_count += 1
        self.error_types[error_type] += 1

        # Track in persistent stats
        self.persistent_stats["errors_by_type"][error_type] = self.persistent_stats["errors_by_type"].get(error_type, 0) + 1

    def record_file_operation(self, operation: str, file_size: int = 0):
        """Record file operations."""
        if operation == "read":
            self.files_read += 1
            self.bytes_read += file_size
        elif operation == "write":
            self.files_written += 1
            self.bytes_written += file_size

        self.features_used.add(f"file_{operation}")

    def record_feature_usage(self, feature: str):
        """Record usage of a specific feature."""
        self.features_used.add(feature)

    def take_memory_snapshot(self):
        """Take a snapshot of current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()

            snapshot = {
                "timestamp": time.time(),
                "rss": memory_info.rss,
                "vms": memory_info.vms,
                "percent": process.memory_percent()
            }

            self.memory_snapshots.append(snapshot)

            # Keep only recent snapshots
            if len(self.memory_snapshots) > 100:
                self.memory_snapshots = self.memory_snapshots[-100:]

        except ImportError:
            # psutil not available
            pass
        except Exception as e:
            self.logger.warning(f"Failed to take memory snapshot: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get current session statistics."""
        current_time = time.time()
        session_duration = current_time - self.session_start_time

        stats = {
            "session": {
                "id": self.session_id,
                "duration": session_duration,
                "start_time": self.session_start_time,
                "current_time": current_time
            },
            "prompts": {
                "total": self.total_prompts,
                "rate_per_minute": (self.total_prompts / session_duration * 60) if session_duration > 0 else 0
            },
            "api_calls": {
                "total": self.total_api_calls,
                "rate_per_minute": (self.total_api_calls / session_duration * 60) if session_duration > 0 else 0,
                "slow_responses": self.slow_responses,
                "slow_response_rate": (self.slow_responses / self.total_api_calls) if self.total_api_calls > 0 else 0
            },
            "tokens": {
                "total_input": self.total_input_tokens,
                "total_output": self.total_output_tokens,
                "total": self.total_tokens,
                "average_per_request": self.total_tokens / self.total_api_calls if self.total_api_calls > 0 else 0
            },
            "tools": {
                "total_calls": self.total_tool_calls,
                "successful": self.successful_tool_calls,
                "failed": self.failed_tool_calls,
                "success_rate": (self.successful_tool_calls / self.total_tool_calls) if self.total_tool_calls > 0 else 0,
                "usage_by_tool": dict(self.tool_usage_count),
                "success_by_tool": dict(self.tool_success_count),
                "failure_by_tool": dict(self.tool_failure_count)
            },
            "files": {
                "read": self.files_read,
                "written": self.files_written,
                "bytes_read": self.bytes_read,
                "bytes_written": self.bytes_written
            },
            "errors": {
                "total": self.error_count,
                "by_type": dict(self.error_types),
                "rate": (self.error_count / self.total_prompts) if self.total_prompts > 0 else 0
            },
            "performance": {
                "average_response_time": sum(self.response_times) / len(self.response_times) if self.response_times else 0,
                "median_response_time": self._calculate_median(self.response_times),
                "p95_response_time": self._calculate_percentile(self.response_times, 95),
                "average_tool_time": sum(self.tool_execution_times) / len(self.tool_execution_times) if self.tool_execution_times else 0
            },
            "features_used": list(self.features_used)
        }

        # Add tool-specific timing stats
        tool_timing_stats = {}
        for tool, times in self.tool_execution_time_by_tool.items():
            if times:
                tool_timing_stats[tool] = {
                    "average": sum(times) / len(times),
                    "median": self._calculate_median(times),
                    "min": min(times),
                    "max": max(times),
                    "count": len(times)
                }

        stats["tool_timing"] = tool_timing_stats

        return stats

    def get_historical_stats(self) -> Dict[str, Any]:
        """Get historical statistics across all sessions."""
        return self.persistent_stats.copy()

    def has_activity(self) -> bool:
        """Check if there has been any activity in this session."""
        return (self.total_prompts > 0 or
                self.total_tool_calls > 0 or
                self.total_api_calls > 0)

    def end_session(self):
        """End the current session and save stats."""
        if not self.has_activity():
            return

        session_duration = time.time() - self.session_start_time

        # Update persistent stats
        self.persistent_stats["total_sessions"] += 1
        self.persistent_stats["total_session_time"] += session_duration
        self.persistent_stats["total_prompts"] += self.total_prompts
        self.persistent_stats["total_tool_calls"] += self.total_tool_calls
        self.persistent_stats["total_tokens"] += self.total_tokens

        # Session-specific stats
        session_stats = {
            "duration": session_duration,
            "prompts": self.total_prompts,
            "tool_calls": self.total_tool_calls,
            "tokens": self.total_tokens,
            "ended_at": time.time()
        }

        self.persistent_stats["recent_sessions"].append(session_stats)

        # Keep only recent sessions
        if len(self.persistent_stats["recent_sessions"]) > 50:
            self.persistent_stats["recent_sessions"] = self.persistent_stats["recent_sessions"][-50:]

        # Update feature usage
        for feature in self.features_used:
            self.persistent_stats["features_used"][feature] = self.persistent_stats["features_used"].get(feature, 0) + 1

        # Save to file
        self._save_persistent_stats()

    def reset_session(self):
        """Reset session statistics."""
        self.session_start_time = time.time()
        self.total_prompts = 0
        self.total_api_calls = 0
        self.total_tool_calls = 0
        self.successful_tool_calls = 0
        self.failed_tool_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.response_times.clear()
        self.tool_execution_times.clear()
        self.tool_usage_count.clear()
        self.tool_success_count.clear()
        self.tool_failure_count.clear()
        self.tool_execution_time_by_tool.clear()
        self.error_count = 0
        self.error_types.clear()
        self.files_read = 0
        self.files_written = 0
        self.bytes_read = 0
        self.bytes_written = 0
        self.memory_snapshots.clear()
        self.slow_responses = 0
        self.features_used.clear()

    def _calculate_median(self, values: List[float]) -> float:
        """Calculate median of a list of values."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        n = len(sorted_values)

        if n % 2 == 0:
            return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
        else:
            return sorted_values[n//2]

    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of a list of values."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))

        if index >= len(sorted_values):
            return sorted_values[-1]

        return sorted_values[index]

    def _load_persistent_stats(self) -> Dict[str, Any]:
        """Load persistent statistics from file."""
        default_stats = {
            "total_sessions": 0,
            "total_session_time": 0.0,
            "total_prompts": 0,
            "total_tool_calls": 0,
            "total_tokens": 0,
            "recent_sessions": [],
            "features_used": {},
            "errors_by_type": {},
            "prompt_lengths": [],
            "api_response_times": [],
            "created_at": time.time(),
            "last_updated": time.time()
        }

        try:
            if self.persistent_stats_file.exists():
                with open(self.persistent_stats_file, 'r', encoding='utf-8') as f:
                    loaded_stats = json.load(f)

                # Merge with defaults to handle new fields
                for key, value in default_stats.items():
                    if key not in loaded_stats:
                        loaded_stats[key] = value

                return loaded_stats

        except Exception as e:
            self.logger.warning(f"Failed to load persistent stats: {e}")

        return default_stats

    def _save_persistent_stats(self):
        """Save persistent statistics to file."""
        try:
            self.persistent_stats["last_updated"] = time.time()

            # Ensure directory exists
            self.persistent_stats_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.persistent_stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.persistent_stats, f, indent=2, ensure_ascii=False)

        except Exception as e:
            self.logger.error(f"Failed to save persistent stats: {e}")

    def export_stats(self, output_file: Path, include_historical: bool = True) -> bool:
        """Export statistics to a file."""
        try:
            export_data = {
                "exported_at": time.time(),
                "session_stats": self.get_stats()
            }

            if include_historical:
                export_data["historical_stats"] = self.get_historical_stats()

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            self.logger.error(f"Failed to export stats: {e}")
            return False

    def clear_persistent_stats(self):
        """Clear all persistent statistics."""
        self.persistent_stats = self._load_persistent_stats()

        # Reset to defaults but keep creation time
        creation_time = self.persistent_stats.get("created_at", time.time())
        self.persistent_stats = {
            "total_sessions": 0,
            "total_session_time": 0.0,
            "total_prompts": 0,
            "total_tool_calls": 0,
            "total_tokens": 0,
            "recent_sessions": [],
            "features_used": {},
            "errors_by_type": {},
            "prompt_lengths": [],
            "api_response_times": [],
            "created_at": creation_time,
            "last_updated": time.time()
        }

        self._save_persistent_stats()

    def get_usage_summary(self) -> str:
        """Get a human-readable usage summary."""
        stats = self.get_stats()
        historical = self.get_historical_stats()

        lines = []
        lines.append("Gemini CLI Usage Summary")
        lines.append("=" * 25)

        # Current session
        lines.append(f"Current Session:")
        lines.append(f"  Duration: {stats['session']['duration']:.1f} seconds")
        lines.append(f"  Prompts: {stats['prompts']['total']}")
        lines.append(f"  Tool calls: {stats['tools']['total_calls']} ({stats['tools']['success_rate']:.1%} success rate)")
        lines.append(f"  Tokens: {stats['tokens']['total']:,}")

        # Historical
        lines.append(f"\nHistorical:")
        lines.append(f"  Total sessions: {historical['total_sessions']}")
        lines.append(f"  Total prompts: {historical['total_prompts']:,}")
        lines.append(f"  Total tool calls: {historical['total_tool_calls']:,}")
        lines.append(f"  Total tokens: {historical['total_tokens']:,}")

        # Most used tools
        if stats['tools']['usage_by_tool']:
            lines.append(f"\nMost used tools:")
            sorted_tools = sorted(stats['tools']['usage_by_tool'].items(), key=lambda x: x[1], reverse=True)
            for tool, count in sorted_tools[:5]:
                lines.append(f"  {tool}: {count} times")

        return "\n".join(lines)