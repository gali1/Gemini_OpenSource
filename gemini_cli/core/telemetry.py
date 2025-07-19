"""
Telemetry and observability for Gemini CLI
Collects metrics, traces, and logs for monitoring and improvement
"""

import asyncio
import json
import time
import uuid
import os
import platform
import sys
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading

from .exceptions import GeminiCLIError


@dataclass
class TelemetryEvent:
    """Represents a telemetry event."""
    event_type: str
    timestamp: str
    session_id: str
    event_data: Dict[str, Any]
    trace_id: Optional[str] = None
    span_id: Optional[str] = None


@dataclass
class Metric:
    """Represents a metric data point."""
    name: str
    value: Union[int, float]
    timestamp: str
    tags: Dict[str, str]
    metric_type: str  # counter, gauge, histogram


@dataclass
class LogEntry:
    """Represents a log entry."""
    level: str
    message: str
    timestamp: str
    session_id: str
    context: Dict[str, Any]


class LocalTelemetryExporter:
    """Exports telemetry data to local files."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.LocalTelemetryExporter")

        # Separate files for different data types
        self.events_file = output_dir / "events.jsonl"
        self.metrics_file = output_dir / "metrics.jsonl"
        self.logs_file = output_dir / "logs.jsonl"
        self.traces_file = output_dir / "traces.jsonl"

    async def export_event(self, event: TelemetryEvent):
        """Export a telemetry event."""
        try:
            event_json = json.dumps(asdict(event), ensure_ascii=False)

            async with aiofiles.open(self.events_file, 'a', encoding='utf-8') as f:
                await f.write(event_json + '\n')

        except Exception as e:
            self.logger.error(f"Failed to export event: {e}")

    async def export_metric(self, metric: Metric):
        """Export a metric."""
        try:
            metric_json = json.dumps(asdict(metric), ensure_ascii=False)

            async with aiofiles.open(self.metrics_file, 'a', encoding='utf-8') as f:
                await f.write(metric_json + '\n')

        except Exception as e:
            self.logger.error(f"Failed to export metric: {e}")

    async def export_log(self, log_entry: LogEntry):
        """Export a log entry."""
        try:
            log_json = json.dumps(asdict(log_entry), ensure_ascii=False)

            async with aiofiles.open(self.logs_file, 'a', encoding='utf-8') as f:
                await f.write(log_json + '\n')

        except Exception as e:
            self.logger.error(f"Failed to export log: {e}")

    async def export_trace(self, trace_data: Dict[str, Any]):
        """Export trace data."""
        try:
            trace_json = json.dumps(trace_data, ensure_ascii=False)

            async with aiofiles.open(self.traces_file, 'a', encoding='utf-8') as f:
                await f.write(trace_json + '\n')

        except Exception as e:
            self.logger.error(f"Failed to export trace: {e}")


class OTLPExporter:
    """Exports telemetry data using OTLP protocol."""

    def __init__(self, endpoint: str, headers: Optional[Dict[str, str]] = None):
        self.endpoint = endpoint
        self.headers = headers or {}
        self.logger = logging.getLogger(f"{__name__}.OTLPExporter")

        # Session for HTTP requests
        self.session = None

    async def initialize(self):
        """Initialize the OTLP exporter."""
        try:
            import aiohttp
            self.session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=10)
            )
        except ImportError:
            self.logger.warning("aiohttp not available, OTLP export disabled")

    async def export_traces(self, traces: List[Dict[str, Any]]):
        """Export traces via OTLP."""
        if not self.session:
            return

        try:
            # Format traces for OTLP
            otlp_data = self._format_traces_for_otlp(traces)

            async with self.session.post(
                f"{self.endpoint}/v1/traces",
                json=otlp_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    self.logger.warning(f"OTLP trace export failed: {response.status}")

        except Exception as e:
            self.logger.error(f"Failed to export traces via OTLP: {e}")

    async def export_metrics(self, metrics: List[Metric]):
        """Export metrics via OTLP."""
        if not self.session:
            return

        try:
            # Format metrics for OTLP
            otlp_data = self._format_metrics_for_otlp(metrics)

            async with self.session.post(
                f"{self.endpoint}/v1/metrics",
                json=otlp_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    self.logger.warning(f"OTLP metric export failed: {response.status}")

        except Exception as e:
            self.logger.error(f"Failed to export metrics via OTLP: {e}")

    async def export_logs(self, logs: List[LogEntry]):
        """Export logs via OTLP."""
        if not self.session:
            return

        try:
            # Format logs for OTLP
            otlp_data = self._format_logs_for_otlp(logs)

            async with self.session.post(
                f"{self.endpoint}/v1/logs",
                json=otlp_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    self.logger.warning(f"OTLP log export failed: {response.status}")

        except Exception as e:
            self.logger.error(f"Failed to export logs via OTLP: {e}")

    def _format_traces_for_otlp(self, traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format traces for OTLP protocol."""
        # Simplified OTLP trace format
        return {
            "resourceSpans": [{
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "gemini-cli"}},
                        {"key": "service.version", "value": {"stringValue": "1.0.0"}}
                    ]
                },
                "scopeSpans": [{
                    "scope": {
                        "name": "gemini-cli",
                        "version": "1.0.0"
                    },
                    "spans": traces
                }]
            }]
        }

    def _format_metrics_for_otlp(self, metrics: List[Metric]) -> Dict[str, Any]:
        """Format metrics for OTLP protocol."""
        # Group metrics by name and type
        metric_groups = defaultdict(list)
        for metric in metrics:
            metric_groups[(metric.name, metric.metric_type)].append(metric)

        otlp_metrics = []
        for (name, metric_type), metric_list in metric_groups.items():
            otlp_metric = {
                "name": name,
                "description": f"Gemini CLI metric: {name}",
                "unit": "",
            }

            if metric_type == "counter":
                otlp_metric["sum"] = {
                    "dataPoints": [
                        {
                            "attributes": [
                                {"key": k, "value": {"stringValue": v}}
                                for k, v in metric.tags.items()
                            ],
                            "timeUnixNano": str(int(time.time() * 1e9)),
                            "asDouble": float(metric.value)
                        }
                        for metric in metric_list
                    ],
                    "aggregationTemporality": 2,  # CUMULATIVE
                    "isMonotonic": True
                }
            elif metric_type == "gauge":
                otlp_metric["gauge"] = {
                    "dataPoints": [
                        {
                            "attributes": [
                                {"key": k, "value": {"stringValue": v}}
                                for k, v in metric.tags.items()
                            ],
                            "timeUnixNano": str(int(time.time() * 1e9)),
                            "asDouble": float(metric.value)
                        }
                        for metric in metric_list
                    ]
                }

            otlp_metrics.append(otlp_metric)

        return {
            "resourceMetrics": [{
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "gemini-cli"}},
                        {"key": "service.version", "value": {"stringValue": "1.0.0"}}
                    ]
                },
                "scopeMetrics": [{
                    "scope": {
                        "name": "gemini-cli",
                        "version": "1.0.0"
                    },
                    "metrics": otlp_metrics
                }]
            }]
        }

    def _format_logs_for_otlp(self, logs: List[LogEntry]) -> Dict[str, Any]:
        """Format logs for OTLP protocol."""
        return {
            "resourceLogs": [{
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "gemini-cli"}},
                        {"key": "service.version", "value": {"stringValue": "1.0.0"}}
                    ]
                },
                "scopeLogs": [{
                    "scope": {
                        "name": "gemini-cli",
                        "version": "1.0.0"
                    },
                    "logRecords": [
                        {
                            "timeUnixNano": str(int(time.time() * 1e9)),
                            "severityText": log.level,
                            "body": {"stringValue": log.message},
                            "attributes": [
                                {"key": k, "value": {"stringValue": str(v)}}
                                for k, v in log.context.items()
                            ]
                        }
                        for log in logs
                    ]
                }]
            }]
        }

    async def shutdown(self):
        """Shutdown the OTLP exporter."""
        if self.session:
            await self.session.close()


class TelemetryManager:
    """Manages telemetry collection and export."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.telemetry_config = config.get("telemetry", {})
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.enabled = self.telemetry_config.get("enabled", False)
        self.target = self.telemetry_config.get("target", "local")  # "local", "gcp", "otlp"
        self.otlp_endpoint = self.telemetry_config.get("otlpEndpoint", "http://localhost:4317")
        self.log_prompts = self.telemetry_config.get("logPrompts", True)

        # Session info
        self.session_id = str(uuid.uuid4())
        self.start_time = time.time()

        # Collectors
        self.events: deque = deque(maxlen=1000)
        self.metrics: deque = deque(maxlen=1000)
        self.logs: deque = deque(maxlen=1000)
        self.traces: deque = deque(maxlen=1000)

        # Counters for metrics
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)

        # Exporters
        self.local_exporter = None
        self.otlp_exporter = None

        # Background tasks
        self.export_task = None
        self.export_interval = 30  # seconds

        # Thread safety
        self.lock = threading.Lock()

    async def initialize(self):
        """Initialize telemetry manager."""
        if not self.enabled:
            self.logger.info("Telemetry disabled")
            return

        try:
            # Setup exporters based on target
            if self.target in ["local", "gcp"]:
                output_dir = self._get_output_directory()
                self.local_exporter = LocalTelemetryExporter(output_dir)

            if self.target in ["otlp", "gcp"]:
                self.otlp_exporter = OTLPExporter(self.otlp_endpoint)
                await self.otlp_exporter.initialize()

            # Start background export task
            self.export_task = asyncio.create_task(self._export_loop())

            # Record initialization
            await self._record_config_event()

            self.logger.info(f"Telemetry initialized (target: {self.target})")

        except Exception as e:
            self.logger.error(f"Failed to initialize telemetry: {e}")
            self.enabled = False

    async def record_user_prompt(self, prompt: str, auth_type: str = "local"):
        """Record a user prompt event."""
        if not self.enabled:
            return

        event_data = {
            "prompt_length": len(prompt),
            "auth_type": auth_type
        }

        if self.log_prompts:
            event_data["prompt"] = prompt

        await self._record_event("gemini_cli.user_prompt", event_data)
        await self._increment_counter("gemini_cli.session.count", {"session_id": self.session_id})

    async def record_tool_call(
        self,
        function_name: str,
        function_args: Dict[str, Any],
        duration_ms: float,
        success: bool,
        decision: Optional[str] = None,
        error: Optional[str] = None,
        error_type: Optional[str] = None
    ):
        """Record a tool call event."""
        if not self.enabled:
            return

        event_data = {
            "function_name": function_name,
            "function_args": function_args,
            "duration_ms": duration_ms,
            "success": success
        }

        if decision:
            event_data["decision"] = decision
        if error:
            event_data["error"] = error
        if error_type:
            event_data["error_type"] = error_type

        await self._record_event("gemini_cli.tool_call", event_data)

        # Record metrics
        tags = {
            "function_name": function_name,
            "success": str(success)
        }
        if decision:
            tags["decision"] = decision

        await self._increment_counter("gemini_cli.tool.call.count", tags)
        await self._record_histogram("gemini_cli.tool.call.latency", duration_ms, tags)

    async def record_api_request(
        self,
        model: str,
        duration_ms: float,
        status_code: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        thoughts_tokens: int = 0,
        tool_tokens: int = 0,
        auth_type: str = "local",
        error: Optional[str] = None,
        error_type: Optional[str] = None
    ):
        """Record an API request event."""
        if not self.enabled:
            return

        event_data = {
            "model": model,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "input_token_count": input_tokens,
            "output_token_count": output_tokens,
            "cached_content_token_count": cached_tokens,
            "thoughts_token_count": thoughts_tokens,
            "tool_token_count": tool_tokens,
            "auth_type": auth_type
        }

        if error:
            event_data["error"] = error
            event_data["error_type"] = error_type
            await self._record_event("gemini_cli.api_error", event_data)
        else:
            await self._record_event("gemini_cli.api_response", event_data)

        # Record metrics
        tags = {
            "model": model,
            "status_code": str(status_code)
        }
        if error_type:
            tags["error_type"] = error_type

        await self._increment_counter("gemini_cli.api.request.count", tags)
        await self._record_histogram("gemini_cli.api.request.latency", duration_ms, {"model": model})

        # Token metrics
        token_types = [
            ("input", input_tokens),
            ("output", output_tokens),
            ("cache", cached_tokens),
            ("thought", thoughts_tokens),
            ("tool", tool_tokens)
        ]

        for token_type, count in token_types:
            if count > 0:
                await self._add_counter("gemini_cli.token.usage", count, {
                    "model": model,
                    "type": token_type
                })

    async def record_flash_fallback(self, auth_type: str = "local"):
        """Record when CLI switches to flash model as fallback."""
        if not self.enabled:
            return

        await self._record_event("gemini_cli.flash_fallback", {"auth_type": auth_type})

    async def record_file_operation(
        self,
        operation: str,
        lines: Optional[int] = None,
        mimetype: Optional[str] = None,
        extension: Optional[str] = None
    ):
        """Record a file operation."""
        if not self.enabled:
            return

        tags = {"operation": operation}
        if lines is not None:
            tags["lines"] = str(lines)
        if mimetype:
            tags["mimetype"] = mimetype
        if extension:
            tags["extension"] = extension

        await self._increment_counter("gemini_cli.file.operation.count", tags)

    async def _record_config_event(self):
        """Record configuration event at startup."""
        config_data = {
            "model": self.config.get("model", "unknown"),
            "embedding_model": self.config.get("embeddingModel", "unknown"),
            "sandbox_enabled": bool(self.config.get("sandbox", {}).get("enabled")),
            "core_tools_enabled": ",".join(self.config.get("coreTools", [])),
            "approval_mode": "auto" if self.config.get("autoAccept") else "manual",
            "api_key_enabled": bool(self.config.get("auth", {}).get("apiKey")),
            "vertex_ai_enabled": self.config.get("auth", {}).get("type") == "vertex_ai",
            "code_assist_enabled": self.config.get("auth", {}).get("type") == "code_assist",
            "log_prompts_enabled": self.log_prompts,
            "file_filtering_respect_git_ignore": self.config.get("fileFiltering", {}).get("respectGitIgnore", True),
            "debug_mode": self.config.get("debug", False),
            "mcp_servers": ",".join(self.config.get("mcpServers", {}).keys())
        }

        await self._record_event("gemini_cli.config", config_data)

    async def _record_event(self, event_type: str, event_data: Dict[str, Any]):
        """Record a telemetry event."""
        try:
            event = TelemetryEvent(
                event_type=event_type,
                timestamp=datetime.now(timezone.utc).isoformat(),
                session_id=self.session_id,
                event_data=event_data
            )

            with self.lock:
                self.events.append(event)

        except Exception as e:
            self.logger.error(f"Failed to record event {event_type}: {e}")

    async def _increment_counter(self, name: str, tags: Dict[str, str]):
        """Increment a counter metric."""
        await self._add_counter(name, 1, tags)

    async def _add_counter(self, name: str, value: Union[int, float], tags: Dict[str, str]):
        """Add to a counter metric."""
        try:
            metric = Metric(
                name=name,
                value=value,
                timestamp=datetime.now(timezone.utc).isoformat(),
                tags=tags,
                metric_type="counter"
            )

            with self.lock:
                self.metrics.append(metric)
                # Also update internal counter
                key = f"{name}:{','.join(f'{k}={v}' for k, v in sorted(tags.items()))}"
                self.counters[key] += value

        except Exception as e:
            self.logger.error(f"Failed to record counter {name}: {e}")

    async def _set_gauge(self, name: str, value: Union[int, float], tags: Dict[str, str]):
        """Set a gauge metric."""
        try:
            metric = Metric(
                name=name,
                value=value,
                timestamp=datetime.now(timezone.utc).isoformat(),
                tags=tags,
                metric_type="gauge"
            )

            with self.lock:
                self.metrics.append(metric)
                # Also update internal gauge
                key = f"{name}:{','.join(f'{k}={v}' for k, v in sorted(tags.items()))}"
                self.gauges[key] = value

        except Exception as e:
            self.logger.error(f"Failed to record gauge {name}: {e}")

    async def _record_histogram(self, name: str, value: Union[int, float], tags: Dict[str, str]):
        """Record a histogram metric."""
        try:
            metric = Metric(
                name=name,
                value=value,
                timestamp=datetime.now(timezone.utc).isoformat(),
                tags=tags,
                metric_type="histogram"
            )

            with self.lock:
                self.metrics.append(metric)
                # Also update internal histogram
                key = f"{name}:{','.join(f'{k}={v}' for k, v in sorted(tags.items()))}"
                self.histograms[key].append(value)
                # Keep only recent values
                if len(self.histograms[key]) > 100:
                    self.histograms[key] = self.histograms[key][-100:]

        except Exception as e:
            self.logger.error(f"Failed to record histogram {name}: {e}")

    async def _export_loop(self):
        """Background loop for exporting telemetry data."""
        while self.enabled:
            try:
                await asyncio.sleep(self.export_interval)
                await self._export_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in telemetry export loop: {e}")

    async def _export_data(self):
        """Export accumulated telemetry data."""
        if not self.enabled:
            return

        try:
            # Get data to export
            with self.lock:
                events_to_export = list(self.events)
                metrics_to_export = list(self.metrics)
                logs_to_export = list(self.logs)
                traces_to_export = list(self.traces)

                # Clear exported data
                self.events.clear()
                self.metrics.clear()
                self.logs.clear()
                self.traces.clear()

            # Export via local exporter
            if self.local_exporter:
                for event in events_to_export:
                    await self.local_exporter.export_event(event)

                for metric in metrics_to_export:
                    await self.local_exporter.export_metric(metric)

                for log_entry in logs_to_export:
                    await self.local_exporter.export_log(log_entry)

                for trace in traces_to_export:
                    await self.local_exporter.export_trace(trace)

            # Export via OTLP exporter
            if self.otlp_exporter:
                if metrics_to_export:
                    await self.otlp_exporter.export_metrics(metrics_to_export)

                if logs_to_export:
                    await self.otlp_exporter.export_logs(logs_to_export)

                if traces_to_export:
                    await self.otlp_exporter.export_traces(traces_to_export)

        except Exception as e:
            self.logger.error(f"Failed to export telemetry data: {e}")

    def _get_output_directory(self) -> Path:
        """Get output directory for local telemetry data."""
        # Calculate project hash for unique directory
        project_path = str(Path.cwd().resolve())
        import hashlib
        project_hash = hashlib.sha256(project_path.encode()).hexdigest()[:16]

        return Path.home() / ".gemini" / "tmp" / project_hash / "telemetry"

    def get_telemetry_stats(self) -> Dict[str, Any]:
        """Get telemetry statistics."""
        with self.lock:
            return {
                "enabled": self.enabled,
                "target": self.target,
                "session_id": self.session_id,
                "session_duration": time.time() - self.start_time,
                "events_collected": len(self.events),
                "metrics_collected": len(self.metrics),
                "logs_collected": len(self.logs),
                "traces_collected": len(self.traces),
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histogram_counts": {k: len(v) for k, v in self.histograms.items()}
            }

    async def record_log(self, level: str, message: str, context: Dict[str, Any]):
        """Record a log entry."""
        if not self.enabled:
            return

        try:
            log_entry = LogEntry(
                level=level,
                message=message,
                timestamp=datetime.now(timezone.utc).isoformat(),
                session_id=self.session_id,
                context=context
            )

            with self.lock:
                self.logs.append(log_entry)

        except Exception as e:
            self.logger.error(f"Failed to record log: {e}")

    async def start_trace(self, operation_name: str) -> str:
        """Start a new trace."""
        trace_id = str(uuid.uuid4())

        if self.enabled:
            trace_data = {
                "traceId": trace_id,
                "name": operation_name,
                "startTime": time.time_ns(),
                "attributes": {
                    "session.id": self.session_id,
                    "service.name": "gemini-cli"
                }
            }

            with self.lock:
                self.traces.append(trace_data)

        return trace_id

    async def end_trace(self, trace_id: str, status: str = "OK", error: Optional[str] = None):
        """End a trace."""
        if not self.enabled:
            return

        try:
            with self.lock:
                # Find and update trace
                for trace in self.traces:
                    if isinstance(trace, dict) and trace.get("traceId") == trace_id:
                        trace["endTime"] = time.time_ns()
                        trace["status"] = {"code": status}
                        if error:
                            trace["status"]["message"] = error
                        break

        except Exception as e:
            self.logger.error(f"Failed to end trace {trace_id}: {e}")

    async def shutdown(self):
        """Shutdown telemetry manager."""
        if not self.enabled:
            return

        try:
            # Cancel export task
            if self.export_task:
                self.export_task.cancel()
                try:
                    await self.export_task
                except asyncio.CancelledError:
                    pass

            # Final export
            await self._export_data()

            # Shutdown exporters
            if self.otlp_exporter:
                await self.otlp_exporter.shutdown()

            self.logger.info("Telemetry shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during telemetry shutdown: {e}")