"""Jinja2 templating singleton for O-Beast pages.

Pages render through a single shared Environment rooted at this package's
``templates`` directory. ``render`` returns HTML as a plain string so route
functions and unit tests can use it without constructing a Starlette Request.
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

_env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=select_autoescape(["html"]),
)


def render(name: str, context: dict | None = None) -> str:
    """Render the named template with ``context`` and return HTML text."""
    return _env.get_template(name).render(**(context or {}))
