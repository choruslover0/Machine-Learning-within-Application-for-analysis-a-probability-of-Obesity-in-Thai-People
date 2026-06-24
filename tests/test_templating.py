import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TemplatingTests(unittest.TestCase):
    def test_render_returns_html_string(self):
        from obesity_ml.templating import render

        html = render("_smoke.html", {"value": "hello-obeast"})
        self.assertIn("hello-obeast", html)

    def test_render_escapes_by_default(self):
        from obesity_ml.templating import render

        html = render("_smoke.html", {"value": "<script>"})
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)


if __name__ == "__main__":
    unittest.main()
