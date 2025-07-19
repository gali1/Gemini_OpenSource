"""
Color utilities for Gemini CLI
Provides color codes and theming support
"""

import os
from typing import Dict, Optional, Any
from enum import Enum


class ColorCode(Enum):
    """ANSI color codes."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    REVERSE = "\033[7m"
    STRIKETHROUGH = "\033[9m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

    # Bright background colors
    BG_BRIGHT_BLACK = "\033[100m"
    BG_BRIGHT_RED = "\033[101m"
    BG_BRIGHT_GREEN = "\033[102m"
    BG_BRIGHT_YELLOW = "\033[103m"
    BG_BRIGHT_BLUE = "\033[104m"
    BG_BRIGHT_MAGENTA = "\033[105m"
    BG_BRIGHT_CYAN = "\033[106m"
    BG_BRIGHT_WHITE = "\033[107m"


class Colors:
    """Color utility class for terminal output."""

    def __init__(self, enabled: Optional[bool] = None):
        if enabled is None:
            # Auto-detect color support
            self.enabled = self._supports_color()
        else:
            self.enabled = enabled

    def _supports_color(self) -> bool:
        """Check if terminal supports colors."""
        # Check NO_COLOR environment variable
        if os.environ.get("NO_COLOR"):
            return False

        # Check FORCE_COLOR
        if os.environ.get("FORCE_COLOR"):
            return True

        # Check if output is a TTY
        if not hasattr(os.sys.stdout, "isatty") or not os.sys.stdout.isatty():
            return False

        # Check TERM environment variable
        term = os.environ.get("TERM", "")
        if term in ("dumb", "unknown"):
            return False

        # Check for common color-supporting terminals
        if any(term.startswith(prefix) for prefix in ("xterm", "screen", "tmux", "konsole", "gnome")):
            return True

        # Check COLORTERM
        if os.environ.get("COLORTERM"):
            return True

        return True

    def colorize(self, text: str, color: ColorCode, reset: bool = True) -> str:
        """Apply color to text."""
        if not self.enabled:
            return text

        colored_text = f"{color.value}{text}"
        if reset:
            colored_text += ColorCode.RESET.value

        return colored_text

    def red(self, text: str) -> str:
        """Make text red."""
        return self.colorize(text, ColorCode.RED)

    def green(self, text: str) -> str:
        """Make text green."""
        return self.colorize(text, ColorCode.GREEN)

    def yellow(self, text: str) -> str:
        """Make text yellow."""
        return self.colorize(text, ColorCode.YELLOW)

    def blue(self, text: str) -> str:
        """Make text blue."""
        return self.colorize(text, ColorCode.BLUE)

    def magenta(self, text: str) -> str:
        """Make text magenta."""
        return self.colorize(text, ColorCode.MAGENTA)

    def cyan(self, text: str) -> str:
        """Make text cyan."""
        return self.colorize(text, ColorCode.CYAN)

    def white(self, text: str) -> str:
        """Make text white."""
        return self.colorize(text, ColorCode.WHITE)

    def bold(self, text: str) -> str:
        """Make text bold."""
        return self.colorize(text, ColorCode.BOLD)

    def dim(self, text: str) -> str:
        """Make text dim."""
        return self.colorize(text, ColorCode.DIM)

    def italic(self, text: str) -> str:
        """Make text italic."""
        return self.colorize(text, ColorCode.ITALIC)

    def underline(self, text: str) -> str:
        """Make text underlined."""
        return self.colorize(text, ColorCode.UNDERLINE)

    def combine(self, text: str, *colors: ColorCode) -> str:
        """Combine multiple color codes."""
        if not self.enabled:
            return text

        codes = "".join([color.value for color in colors])
        return f"{codes}{text}{ColorCode.RESET.value}"


class ColorTheme:
    """Color theme definitions."""

    THEMES = {
        "Default": {
            "primary": ColorCode.BLUE,
            "secondary": ColorCode.CYAN,
            "success": ColorCode.GREEN,
            "warning": ColorCode.YELLOW,
            "error": ColorCode.RED,
            "info": ColorCode.BLUE,
            "prompt": ColorCode.GREEN,
            "user_input": ColorCode.WHITE,
            "ai_response": ColorCode.CYAN,
            "tool_call": ColorCode.MAGENTA,
            "system": ColorCode.DIM,
            "accent": ColorCode.BRIGHT_BLUE,
        },

        "Dark": {
            "primary": ColorCode.BRIGHT_BLUE,
            "secondary": ColorCode.BRIGHT_CYAN,
            "success": ColorCode.BRIGHT_GREEN,
            "warning": ColorCode.BRIGHT_YELLOW,
            "error": ColorCode.BRIGHT_RED,
            "info": ColorCode.BRIGHT_BLUE,
            "prompt": ColorCode.BRIGHT_GREEN,
            "user_input": ColorCode.BRIGHT_WHITE,
            "ai_response": ColorCode.BRIGHT_CYAN,
            "tool_call": ColorCode.BRIGHT_MAGENTA,
            "system": ColorCode.BRIGHT_BLACK,
            "accent": ColorCode.BRIGHT_BLUE,
        },

        "Light": {
            "primary": ColorCode.BLUE,
            "secondary": ColorCode.CYAN,
            "success": ColorCode.GREEN,
            "warning": ColorCode.YELLOW,
            "error": ColorCode.RED,
            "info": ColorCode.BLUE,
            "prompt": ColorCode.GREEN,
            "user_input": ColorCode.BLACK,
            "ai_response": ColorCode.BLUE,
            "tool_call": ColorCode.MAGENTA,
            "system": ColorCode.DIM,
            "accent": ColorCode.BLUE,
        },

        "Monokai": {
            "primary": ColorCode.MAGENTA,
            "secondary": ColorCode.CYAN,
            "success": ColorCode.GREEN,
            "warning": ColorCode.YELLOW,
            "error": ColorCode.RED,
            "info": ColorCode.CYAN,
            "prompt": ColorCode.GREEN,
            "user_input": ColorCode.WHITE,
            "ai_response": ColorCode.MAGENTA,
            "tool_call": ColorCode.YELLOW,
            "system": ColorCode.DIM,
            "accent": ColorCode.BRIGHT_MAGENTA,
        },

        "GitHub": {
            "primary": ColorCode.BLUE,
            "secondary": ColorCode.CYAN,
            "success": ColorCode.GREEN,
            "warning": ColorCode.YELLOW,
            "error": ColorCode.RED,
            "info": ColorCode.BLUE,
            "prompt": ColorCode.GREEN,
            "user_input": ColorCode.BLACK,
            "ai_response": ColorCode.BLUE,
            "tool_call": ColorCode.MAGENTA,
            "system": ColorCode.DIM,
            "accent": ColorCode.BLUE,
        },
    }

    def __init__(self, theme_name: str = "Default"):
        self.theme_name = theme_name
        self.colors = Colors()
        self.theme = self.THEMES.get(theme_name, self.THEMES["Default"])

    def get_color(self, element: str) -> ColorCode:
        """Get color for a UI element."""
        return self.theme.get(element, ColorCode.WHITE)

    def apply(self, text: str, element: str) -> str:
        """Apply theme color to text."""
        color = self.get_color(element)
        return self.colors.colorize(text, color)

    def primary(self, text: str) -> str:
        """Apply primary color."""
        return self.apply(text, "primary")

    def secondary(self, text: str) -> str:
        """Apply secondary color."""
        return self.apply(text, "secondary")

    def success(self, text: str) -> str:
        """Apply success color."""
        return self.apply(text, "success")

    def warning(self, text: str) -> str:
        """Apply warning color."""
        return self.apply(text, "warning")

    def error(self, text: str) -> str:
        """Apply error color."""
        return self.apply(text, "error")

    def info(self, text: str) -> str:
        """Apply info color."""
        return self.apply(text, "info")

    def prompt(self, text: str) -> str:
        """Apply prompt color."""
        return self.apply(text, "prompt")

    def user_input(self, text: str) -> str:
        """Apply user input color."""
        return self.apply(text, "user_input")

    def ai_response(self, text: str) -> str:
        """Apply AI response color."""
        return self.apply(text, "ai_response")

    def tool_call(self, text: str) -> str:
        """Apply tool call color."""
        return self.apply(text, "tool_call")

    def system(self, text: str) -> str:
        """Apply system color."""
        return self.apply(text, "system")

    def accent(self, text: str) -> str:
        """Apply accent color."""
        return self.apply(text, "accent")

    @classmethod
    def list_themes(cls) -> list:
        """List available themes."""
        return list(cls.THEMES.keys())

    @classmethod
    def get_theme_colors(cls, theme_name: str) -> Dict[str, ColorCode]:
        """Get colors for a specific theme."""
        return cls.THEMES.get(theme_name, cls.THEMES["Default"])


def strip_color(text: str) -> str:
    """Remove ANSI color codes from text."""
    import re
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def get_text_width(text: str) -> int:
    """Get the display width of text (without color codes)."""
    return len(strip_color(text))


def pad_text(text: str, width: int, align: str = "left") -> str:
    """Pad text to a specific width, accounting for color codes."""
    text_width = get_text_width(text)
    padding = width - text_width

    if padding <= 0:
        return text

    if align == "left":
        return text + " " * padding
    elif align == "right":
        return " " * padding + text
    elif align == "center":
        left_padding = padding // 2
        right_padding = padding - left_padding
        return " " * left_padding + text + " " * right_padding
    else:
        return text


def truncate_with_ellipsis(text: str, max_width: int, ellipsis: str = "...") -> str:
    """Truncate text to max width, preserving color codes."""
    stripped = strip_color(text)

    if len(stripped) <= max_width:
        return text

    # Find truncation point
    ellipsis_width = len(ellipsis)
    if max_width <= ellipsis_width:
        return ellipsis[:max_width]

    truncate_at = max_width - ellipsis_width

    # Simple truncation (could be improved to handle color codes better)
    if len(text) <= truncate_at:
        return text + ellipsis

    return strip_color(text)[:truncate_at] + ellipsis