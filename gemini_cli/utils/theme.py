"""
Theme management for Gemini CLI
Provides theme switching and customization functionality
"""

import json
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from .colors import ColorTheme, ColorCode


class ThemeManager:
    """Manages CLI themes and theme switching."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.current_theme_name = "Default"
        self.current_theme = ColorTheme(self.current_theme_name)

        # Theme directories
        self.user_themes_dir = Path.home() / ".gemini" / "themes"
        self.system_themes_dir = Path(__file__).parent / "themes"

        # Ensure user themes directory exists
        self.user_themes_dir.mkdir(parents=True, exist_ok=True)

        # Load custom themes
        self.custom_themes = self._load_custom_themes()

    def list_themes(self) -> List[str]:
        """List all available themes."""
        builtin_themes = list(ColorTheme.THEMES.keys())
        custom_themes = list(self.custom_themes.keys())

        # Combine and deduplicate (custom themes override builtin)
        all_themes = builtin_themes + [theme for theme in custom_themes if theme not in builtin_themes]

        return sorted(all_themes)

    def get_current_theme_name(self) -> str:
        """Get the name of the current theme."""
        return self.current_theme_name

    def get_current_theme(self) -> ColorTheme:
        """Get the current theme object."""
        return self.current_theme

    def set_theme(self, theme_name: str) -> bool:
        """Set the current theme."""
        try:
            # Check if theme exists
            if not self.theme_exists(theme_name):
                self.logger.error(f"Theme not found: {theme_name}")
                return False

            # Create new theme instance
            if theme_name in self.custom_themes:
                # Use custom theme
                theme_config = self.custom_themes[theme_name]
                self.current_theme = self._create_custom_theme(theme_name, theme_config)
            else:
                # Use builtin theme
                self.current_theme = ColorTheme(theme_name)

            self.current_theme_name = theme_name
            self.logger.info(f"Theme changed to: {theme_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to set theme {theme_name}: {e}")
            return False

    def theme_exists(self, theme_name: str) -> bool:
        """Check if a theme exists."""
        return (theme_name in ColorTheme.THEMES or
                theme_name in self.custom_themes)

    def get_theme_info(self, theme_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a theme."""
        if theme_name in ColorTheme.THEMES:
            return {
                "name": theme_name,
                "type": "builtin",
                "description": f"Built-in {theme_name} theme",
                "colors": {k: v.name for k, v in ColorTheme.THEMES[theme_name].items()}
            }
        elif theme_name in self.custom_themes:
            config = self.custom_themes[theme_name]
            return {
                "name": theme_name,
                "type": "custom",
                "description": config.get("description", f"Custom {theme_name} theme"),
                "colors": config.get("colors", {}),
                "author": config.get("author", "Unknown"),
                "version": config.get("version", "1.0")
            }
        else:
            return None

    def create_custom_theme(self, name: str, colors: Dict[str, str], description: str = "", author: str = "", version: str = "1.0") -> bool:
        """Create a new custom theme."""
        try:
            # Validate color mappings
            if not self._validate_theme_colors(colors):
                self.logger.error("Invalid color definitions in theme")
                return False

            # Create theme configuration
            theme_config = {
                "name": name,
                "description": description,
                "author": author,
                "version": version,
                "colors": colors
            }

            # Save to file
            theme_file = self.user_themes_dir / f"{name}.json"
            with open(theme_file, 'w', encoding='utf-8') as f:
                json.dump(theme_config, f, indent=2, ensure_ascii=False)

            # Add to custom themes
            self.custom_themes[name] = theme_config

            self.logger.info(f"Created custom theme: {name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create custom theme {name}: {e}")
            return False

    def delete_custom_theme(self, name: str) -> bool:
        """Delete a custom theme."""
        try:
            if name not in self.custom_themes:
                self.logger.error(f"Custom theme not found: {name}")
                return False

            # Don't allow deleting the current theme
            if name == self.current_theme_name:
                self.logger.error("Cannot delete the currently active theme")
                return False

            # Remove theme file
            theme_file = self.user_themes_dir / f"{name}.json"
            if theme_file.exists():
                theme_file.unlink()

            # Remove from custom themes
            del self.custom_themes[name]

            self.logger.info(f"Deleted custom theme: {name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete custom theme {name}: {e}")
            return False

    def export_theme(self, theme_name: str, output_file: Path) -> bool:
        """Export a theme to a file."""
        try:
            theme_info = self.get_theme_info(theme_name)
            if not theme_info:
                return False

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(theme_info, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            self.logger.error(f"Failed to export theme {theme_name}: {e}")
            return False

    def import_theme(self, input_file: Path) -> Optional[str]:
        """Import a theme from a file."""
        try:
            if not input_file.exists():
                return None

            with open(input_file, 'r', encoding='utf-8') as f:
                theme_config = json.load(f)

            name = theme_config.get("name")
            if not name:
                self.logger.error("Theme file missing name")
                return None

            colors = theme_config.get("colors", {})
            if not self._validate_theme_colors(colors):
                self.logger.error("Invalid theme colors")
                return None

            # Create the theme
            success = self.create_custom_theme(
                name=name,
                colors=colors,
                description=theme_config.get("description", ""),
                author=theme_config.get("author", ""),
                version=theme_config.get("version", "1.0")
            )

            return name if success else None

        except Exception as e:
            self.logger.error(f"Failed to import theme: {e}")
            return None

    def preview_theme(self, theme_name: str) -> str:
        """Generate a preview of a theme."""
        if not self.theme_exists(theme_name):
            return f"Theme '{theme_name}' not found"

        # Create temporary theme instance
        if theme_name in self.custom_themes:
            theme_config = self.custom_themes[theme_name]
            theme = self._create_custom_theme(theme_name, theme_config)
        else:
            theme = ColorTheme(theme_name)

        # Generate preview text
        preview_lines = [
            f"Theme: {theme_name}",
            "=" * 30,
            theme.primary("Primary text"),
            theme.secondary("Secondary text"),
            theme.success("Success message"),
            theme.warning("Warning message"),
            theme.error("Error message"),
            theme.info("Info message"),
            theme.prompt("Prompt text"),
            theme.user_input("User input"),
            theme.ai_response("AI response"),
            theme.tool_call("Tool call"),
            theme.system("System message"),
            theme.accent("Accent text")
        ]

        return "\n".join(preview_lines)

    def reset_to_default(self):
        """Reset to the default theme."""
        self.set_theme("Default")

    def _load_custom_themes(self) -> Dict[str, Dict[str, Any]]:
        """Load custom themes from files."""
        custom_themes = {}

        # Load from user themes directory
        if self.user_themes_dir.exists():
            for theme_file in self.user_themes_dir.glob("*.json"):
                try:
                    with open(theme_file, 'r', encoding='utf-8') as f:
                        theme_config = json.load(f)

                    name = theme_config.get("name", theme_file.stem)
                    custom_themes[name] = theme_config

                except Exception as e:
                    self.logger.warning(f"Failed to load theme {theme_file}: {e}")

        return custom_themes

    def _validate_theme_colors(self, colors: Dict[str, str]) -> bool:
        """Validate theme color definitions."""
        required_elements = {
            "primary", "secondary", "success", "warning", "error",
            "info", "prompt", "user_input", "ai_response", "tool_call",
            "system", "accent"
        }

        valid_colors = {code.name for code in ColorCode}

        # Check that all required elements are present
        if not all(element in colors for element in required_elements):
            return False

        # Check that color values are valid
        for element, color_name in colors.items():
            if color_name not in valid_colors:
                self.logger.warning(f"Invalid color '{color_name}' for element '{element}'")
                return False

        return True

    def _create_custom_theme(self, name: str, config: Dict[str, Any]) -> ColorTheme:
        """Create a ColorTheme instance from custom configuration."""
        # Start with default theme as base
        theme = ColorTheme("Default")

        # Override with custom colors
        colors = config.get("colors", {})
        custom_color_map = {}

        for element, color_name in colors.items():
            try:
                color_code = ColorCode[color_name]
                custom_color_map[element] = color_code
            except KeyError:
                self.logger.warning(f"Unknown color '{color_name}' in theme '{name}'")
                continue

        # Update theme colors
        theme.theme.update(custom_color_map)
        theme.theme_name = name

        return theme

    def get_theme_suggestions(self, current_time_hour: Optional[int] = None) -> List[str]:
        """Get theme suggestions based on time of day or other factors."""
        import datetime

        if current_time_hour is None:
            current_time_hour = datetime.datetime.now().hour

        suggestions = []

        # Time-based suggestions
        if 6 <= current_time_hour < 18:  # Daytime
            suggestions.extend(["Light", "GitHub"])
        else:  # Evening/Night
            suggestions.extend(["Dark", "Monokai"])

        # Always suggest default
        if "Default" not in suggestions:
            suggestions.append("Default")

        return suggestions

    def auto_detect_theme(self) -> str:
        """Auto-detect appropriate theme based on terminal and environment."""
        # Check environment variables
        if os.environ.get("DARK_MODE") == "1":
            return "Dark"

        if os.environ.get("LIGHT_MODE") == "1":
            return "Light"

        # Check terminal background (simplified heuristic)
        term = os.environ.get("TERM", "").lower()
        if "dark" in term:
            return "Dark"
        elif "light" in term:
            return "Light"

        # Default fallback
        return "Default"

    def get_theme_stats(self) -> Dict[str, Any]:
        """Get statistics about available themes."""
        return {
            "total_themes": len(self.list_themes()),
            "builtin_themes": len(ColorTheme.THEMES),
            "custom_themes": len(self.custom_themes),
            "current_theme": self.current_theme_name,
            "themes_directory": str(self.user_themes_dir)
        }