"""
Formatting utilities for Gemini CLI
Provides functions for formatting various data types for display
"""

import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union
from pathlib import Path


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB", "PB"]
    size_index = 0
    size_value = float(size_bytes)

    while size_value >= 1024 and size_index < len(size_names) - 1:
        size_value /= 1024
        size_index += 1

    # Format with appropriate precision
    if size_index == 0:
        return f"{int(size_value)} {size_names[size_index]}"
    elif size_value >= 100:
        return f"{size_value:.0f} {size_names[size_index]}"
    elif size_value >= 10:
        return f"{size_value:.1f} {size_names[size_index]}"
    else:
        return f"{size_value:.2f} {size_names[size_index]}"


def format_duration(seconds: float) -> str:
    """Format duration in human readable format."""
    if seconds < 0:
        return "0s"

    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        if remaining_seconds < 1:
            return f"{minutes}m"
        return f"{minutes}m {remaining_seconds:.0f}s"
    elif seconds < 86400:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        if remaining_minutes == 0:
            return f"{hours}h"
        return f"{hours}h {remaining_minutes}m"
    else:
        days = int(seconds // 86400)
        remaining_hours = int((seconds % 86400) // 3600)
        if remaining_hours == 0:
            return f"{days}d"
        return f"{days}d {remaining_hours}h"


def format_timestamp(timestamp: Union[str, datetime, float], format_type: str = "relative") -> str:
    """Format timestamp in various formats."""
    if isinstance(timestamp, str):
        try:
            # Try to parse ISO format
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            return timestamp
    elif isinstance(timestamp, float):
        dt = datetime.fromtimestamp(timestamp)
    elif isinstance(timestamp, datetime):
        dt = timestamp
    else:
        return str(timestamp)

    now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()

    if format_type == "relative":
        return format_relative_time(dt, now)
    elif format_type == "short":
        return dt.strftime("%m/%d %H:%M")
    elif format_type == "medium":
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    elif format_type == "long":
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    else:
        return dt.isoformat()


def format_relative_time(dt: datetime, now: Optional[datetime] = None) -> str:
    """Format datetime as relative time (e.g., '2 minutes ago')."""
    if now is None:
        now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()

    delta = now - dt

    # Handle future timestamps
    if delta.total_seconds() < 0:
        delta = dt - now
        future = True
    else:
        future = False

    seconds = int(delta.total_seconds())

    if seconds < 60:
        time_str = "just now" if not future else "in a moment"
    elif seconds < 3600:
        minutes = seconds // 60
        time_str = f"{minutes} minute{'s' if minutes != 1 else ''}"
    elif seconds < 86400:
        hours = seconds // 3600
        time_str = f"{hours} hour{'s' if hours != 1 else ''}"
    elif seconds < 2592000:  # 30 days
        days = seconds // 86400
        time_str = f"{days} day{'s' if days != 1 else ''}"
    elif seconds < 31536000:  # 365 days
        months = seconds // 2592000
        time_str = f"{months} month{'s' if months != 1 else ''}"
    else:
        years = seconds // 31536000
        time_str = f"{years} year{'s' if years != 1 else ''}"

    if time_str in ("just now", "in a moment"):
        return time_str

    return f"in {time_str}" if future else f"{time_str} ago"


def format_memory_usage(bytes_used: int, total_bytes: Optional[int] = None) -> str:
    """Format memory usage information."""
    used_str = format_file_size(bytes_used)

    if total_bytes is None:
        return used_str

    total_str = format_file_size(total_bytes)
    percentage = (bytes_used / total_bytes) * 100 if total_bytes > 0 else 0

    return f"{used_str} / {total_str} ({percentage:.1f}%)"


def format_token_count(tokens: int, context: Optional[str] = None) -> str:
    """Format token count with context."""
    if tokens < 1000:
        count_str = str(tokens)
    elif tokens < 1000000:
        count_str = f"{tokens/1000:.1f}K"
    else:
        count_str = f"{tokens/1000000:.1f}M"

    if context:
        return f"{count_str} tokens ({context})"

    return f"{count_str} tokens"


def truncate_text(text: str, max_length: int, ellipsis: str = "...") -> str:
    """Truncate text to maximum length with ellipsis."""
    if len(text) <= max_length:
        return text

    if max_length <= len(ellipsis):
        return ellipsis[:max_length]

    truncate_length = max_length - len(ellipsis)
    return text[:truncate_length] + ellipsis


def truncate_middle(text: str, max_length: int, separator: str = "...") -> str:
    """Truncate text in the middle, preserving start and end."""
    if len(text) <= max_length:
        return text

    if max_length <= len(separator):
        return separator[:max_length]

    available_length = max_length - len(separator)
    start_length = available_length // 2
    end_length = available_length - start_length

    return text[:start_length] + separator + text[-end_length:]


def format_json(data: Any, indent: int = 2, max_width: Optional[int] = None) -> str:
    """Format JSON data for display."""
    try:
        json_str = json.dumps(data, indent=indent, ensure_ascii=False, default=str)

        if max_width and len(json_str) > max_width:
            # Try compact format
            compact_str = json.dumps(data, separators=(',', ':'), ensure_ascii=False, default=str)
            if len(compact_str) <= max_width:
                return compact_str
            else:
                return truncate_text(compact_str, max_width)

        return json_str

    except (TypeError, ValueError):
        return str(data)


def format_code_block(code: str, language: str = "", max_lines: Optional[int] = None) -> str:
    """Format code block with syntax highlighting indicators."""
    lines = code.split('\n')

    if max_lines and len(lines) > max_lines:
        displayed_lines = lines[:max_lines]
        remaining = len(lines) - max_lines
        displayed_lines.append(f"... ({remaining} more lines)")
        lines = displayed_lines

    # Add line numbers for longer code blocks
    if len(lines) > 5:
        width = len(str(len(lines)))
        numbered_lines = []
        for i, line in enumerate(lines, 1):
            if line.startswith("..."):
                numbered_lines.append(f"{'':>{width}} {line}")
            else:
                numbered_lines.append(f"{i:>{width}} {line}")
        lines = numbered_lines

    # Wrap in code block markers
    if language:
        header = f"```{language}"
    else:
        header = "```"

    return header + "\n" + "\n".join(lines) + "\n```"


def format_table(data: list, headers: Optional[list] = None, max_width: Optional[int] = None) -> str:
    """Format data as a simple table."""
    if not data:
        return ""

    # Convert all data to strings
    if headers:
        table_data = [headers] + [[str(cell) for cell in row] for row in data]
    else:
        table_data = [[str(cell) for cell in row] for row in data]

    if not table_data:
        return ""

    # Calculate column widths
    num_cols = max(len(row) for row in table_data)
    col_widths = []

    for col in range(num_cols):
        max_width_col = max(len(row[col]) if col < len(row) else 0 for row in table_data)
        col_widths.append(max_width_col)

    # Apply maximum width constraint
    if max_width:
        total_width = sum(col_widths) + (num_cols - 1) * 3  # 3 chars for " | "
        if total_width > max_width:
            # Proportionally reduce column widths
            reduction_factor = (max_width - (num_cols - 1) * 3) / sum(col_widths)
            col_widths = [max(5, int(w * reduction_factor)) for w in col_widths]

    # Format rows
    formatted_rows = []
    for i, row in enumerate(table_data):
        formatted_cells = []
        for j, cell in enumerate(row):
            if j < len(col_widths):
                width = col_widths[j]
                if len(cell) > width:
                    cell = truncate_text(cell, width)
                formatted_cells.append(cell.ljust(width))
            else:
                formatted_cells.append(cell)

        formatted_rows.append(" | ".join(formatted_cells))

        # Add separator after header
        if headers and i == 0:
            separator = " | ".join("-" * w for w in col_widths)
            formatted_rows.append(separator)

    return "\n".join(formatted_rows)


def format_key_value_pairs(data: Dict[str, Any], indent: int = 0, max_value_length: Optional[int] = None) -> str:
    """Format dictionary as key-value pairs."""
    lines = []
    indent_str = " " * indent

    max_key_length = max(len(str(key)) for key in data.keys()) if data else 0

    for key, value in data.items():
        key_str = str(key).ljust(max_key_length)
        value_str = str(value)

        if max_value_length and len(value_str) > max_value_length:
            value_str = truncate_text(value_str, max_value_length)

        lines.append(f"{indent_str}{key_str}: {value_str}")

    return "\n".join(lines)


def format_list(items: list, bullet: str = "•", indent: int = 0, max_item_length: Optional[int] = None) -> str:
    """Format list items with bullets."""
    lines = []
    indent_str = " " * indent

    for item in items:
        item_str = str(item)

        if max_item_length and len(item_str) > max_item_length:
            item_str = truncate_text(item_str, max_item_length)

        lines.append(f"{indent_str}{bullet} {item_str}")

    return "\n".join(lines)


def format_progress_bar(current: int, total: int, width: int = 20, filled_char: str = "█", empty_char: str = "░") -> str:
    """Format a progress bar."""
    if total <= 0:
        return empty_char * width

    progress = min(current / total, 1.0)
    filled_width = int(progress * width)
    empty_width = width - filled_width

    bar = filled_char * filled_width + empty_char * empty_width
    percentage = progress * 100

    return f"{bar} {percentage:.1f}%"


def format_bytes_diff(old_size: int, new_size: int) -> str:
    """Format the difference between two byte sizes."""
    diff = new_size - old_size

    if diff == 0:
        return "no change"
    elif diff > 0:
        return f"+{format_file_size(diff)}"
    else:
        return f"-{format_file_size(abs(diff))}"


def format_path(path: Union[str, Path], max_length: Optional[int] = None, style: str = "auto") -> str:
    """Format file path for display."""
    path_str = str(path)

    if max_length and len(path_str) > max_length:
        if style == "start":
            return "..." + path_str[-(max_length-3):]
        elif style == "middle":
            return truncate_middle(path_str, max_length)
        else:  # auto or end
            return truncate_text(path_str, max_length)

    return path_str


def wrap_text(text: str, width: int = 80, indent: int = 0) -> str:
    """Wrap text to specified width."""
    import textwrap

    wrapper = textwrap.TextWrapper(
        width=width,
        initial_indent=" " * indent,
        subsequent_indent=" " * indent,
        break_long_words=False,
        break_on_hyphens=False
    )

    return wrapper.fill(text)


def align_text(text: str, width: int, alignment: str = "left") -> str:
    """Align text within specified width."""
    if alignment == "center":
        return text.center(width)
    elif alignment == "right":
        return text.rjust(width)
    else:  # left
        return text.ljust(width)