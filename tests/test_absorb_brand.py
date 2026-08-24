import base64
import hashlib
import json
import re
import unittest
from pathlib import Path

from PIL import Image
from absorb.conversation.prompts import SYSTEM_PROMPT


ROOT = Path(__file__).resolve().parents[1]
BRAND = ROOT / "static" / "brand"


class AbsorbBrandTests(unittest.TestCase):
    def test_canonical_logo_is_exact_attached_source(self):
        canonical = BRAND / "absorb-mark.png"
        self.assertEqual(
            hashlib.sha256(canonical.read_bytes()).hexdigest(),
            "2e7b3950809748d5e02648dfc26b0b403f7cabd2d706ce3130b28bad86c9443d",
        )

    def test_required_logo_derivatives_are_png_with_expected_dimensions(self):
        expected = {
            "absorb-mark-16.png": (16, 16),
            "absorb-mark-32.png": (32, 32),
            "absorb-mark-48.png": (48, 48),
            "absorb-mark-64.png": (64, 64),
            "absorb-mark-128.png": (128, 128),
            "absorb-mark-192.png": (192, 192),
            "absorb-mark-512.png": (512, 512),
            "favicon-16x16.png": (16, 16),
            "favicon-32x32.png": (32, 32),
            "apple-touch-icon.png": (180, 180),
            "android-chrome-192x192.png": (192, 192),
            "android-chrome-512x512.png": (512, 512),
            "maskable-icon-512x512.png": (512, 512),
            "line-profile-640x640.png": (640, 640),
            "social-preview-1200x630.png": (1200, 630),
        }
        for filename, dimensions in expected.items():
            with self.subTest(filename=filename):
                with Image.open(BRAND / filename) as image:
                    self.assertEqual(image.format, "PNG")
                    self.assertEqual(image.size, dimensions)
                    self.assertEqual(image.convert("RGB").getpixel((0, 0)), (255, 255, 255))

    def test_manifest_references_existing_absorb_icons(self):
        manifest = json.loads((ROOT / "static" / "manifest.webmanifest").read_text(encoding="utf-8"))
        self.assertEqual(manifest["name"], "ABSORB")
        self.assertEqual(manifest["theme_color"], "#122643")
        for icon in manifest["icons"]:
            self.assertTrue((ROOT / icon["src"].removeprefix("/")).is_file())

    def test_design_system_uses_measured_navy_and_prohibits_old_persona(self):
        design = (ROOT / "DESIGN.md").read_text(encoding="utf-8")
        self.assertIn("--absorb-navy: #122643", design)
        self.assertIn("導覽文字標誌固定為 `ABSORB`", design)
        self.assertIn("Avenir Next", design)
        self.assertIn("不下載、內嵌或提交專有字型", design)
        self.assertIn("不使用玻璃擬態", design)

    def test_new_user_visible_sources_do_not_contain_legacy_brand(self):
        paths = list((ROOT / "templates").glob("*.html"))
        paths += [
            ROOT / "static" / "app.css",
            ROOT / "static" / "app.js",
            ROOT / "assets" / "rich-menu.svg",
            ROOT / "reporting" / "pdf_generator.py",
            ROOT / "stock_papi" / "integrations" / "line" / "flex.py",
            ROOT / "stock_papi" / "integrations" / "line" / "presentation.py",
            ROOT / "stock_papi" / "services" / "papi.py",
            ROOT / "scripts" / "build_competition_doc.py",
            ROOT / "scripts" / "generate_sample_daily_report.py",
            ROOT / "scripts" / "generate_release_report.py",
        ]
        for path in paths:
            with self.subTest(path=path.name):
                text = path.read_text(encoding="utf-8").lower()
                self.assertNotIn("stock papi", text)
                self.assertNotIn("papillon", text)
                self.assertNotIn("老爸", text)
                self.assertNotIn("蝴蝶", text)

    def test_prompt_and_http_user_agents_use_absorb_identity(self):
        self.assertIn("你是 ABSORB", SYSTEM_PROMPT)
        self.assertNotIn("Papi", SYSTEM_PROMPT)
        self.assertNotIn("老爸", SYSTEM_PROMPT)
        self.assertNotIn("蝴蝶", SYSTEM_PROMPT)
        user_agent_sources = (
            (ROOT / "local_quant.py").read_text(encoding="utf-8"),
            (ROOT / "stock_papi" / "integrations" / "news" / "provider.py").read_text(encoding="utf-8"),
        )
        for source in user_agent_sources:
            self.assertIn("ABSORB/1.0", source)

    def test_web_metadata_footer_and_error_pages_are_branded(self):
        base = (ROOT / "templates" / "base.html").read_text(encoding="utf-8")
        for required in (
            'property="og:title"', 'property="og:description"',
            'name="twitter:card"', 'type="application/ld+json"',
            'class="site-footer"',
        ):
            self.assertIn(required, base)
        structured_data = re.search(
            r'<script type="application/ld\+json">(.*?)</script>', base
        ).group(1)
        digest = base64.b64encode(hashlib.sha256(structured_data.encode()).digest()).decode()
        csp_source = (ROOT / "stock_papi" / "web" / "app_factory.py").read_text(encoding="utf-8")
        self.assertIn(f"'sha256-{digest}'", csp_source)
        for name in ("404.html", "500.html"):
            self.assertIn("ABSORB", (ROOT / "templates" / name).read_text(encoding="utf-8"))

    def test_product_shell_uses_accessible_uppercase_product_labels(self):
        base = (ROOT / "templates" / "base.html").read_text(encoding="utf-8")

        self.assertIn('aria-label="回到 ABSORB 主畫面"', base)
        self.assertIn('>ABSORB</a>', base)
        self.assertIn(">ASK ABSORB</button>", base)
        self.assertIn(">ASK ABSORB</h2>", base)
        self.assertIn('aria-label="關閉 ASK ABSORB"', base)

    def test_static_accessibility_contract(self):
        templates = list((ROOT / "templates").glob("*.html"))
        base = (ROOT / "templates" / "base.html").read_text(encoding="utf-8")
        dashboard = (ROOT / "templates" / "dashboard.html").read_text(encoding="utf-8")
        self.assertIn('class="skip-link"', base)
        self.assertIn('id="main-content"', base)
        self.assertIn('aria-live="polite"', dashboard)
        base_ids = re.findall(r'\bid="([^"]+)"', base)
        self.assertEqual(len(base_ids), len(set(base_ids)))
        for path in templates:
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                for image in re.findall(r"<img\b[^>]*>", source, re.IGNORECASE):
                    self.assertRegex(image, r'\balt="[^"]*"')
                if path.name != "base.html":
                    rendered_ids = base_ids + re.findall(r'\bid="([^"]+)"', source)
                    self.assertEqual(len(rendered_ids), len(set(rendered_ids)))


if __name__ == "__main__":
    unittest.main()
