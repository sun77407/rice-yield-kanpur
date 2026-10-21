"""Screenshot the three app tabs (run with: uv run --with playwright python scripts/app_screenshots.py)."""

from __future__ import annotations

import time

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8501"


def main() -> None:
    """Drive the running app, generate one advisory, save the three tab screenshots."""
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        pg = b.new_page(viewport={"width": 1400, "height": 1000})
        pg.goto(URL)
        pg.get_by_text("P50 (t/ha)").wait_for(timeout=60000)
        time.sleep(3)
        pg.screenshot(path="figures/app_predict.png", full_page=True)
        pg.get_by_role("tab", name="Advisory").click()
        pg.get_by_role("button", name="Generate advisory").click()
        t0 = time.time()
        pg.get_by_text("Fully supported").first.wait_for(timeout=300000)
        print("advisory round trip (click to result) s:", round(time.time() - t0, 1))
        time.sleep(2)
        pg.screenshot(path="figures/app_advisory.png", full_page=True)
        pg.get_by_role("tab", name="Diagnostics").click()
        time.sleep(4)
        pg.screenshot(path="figures/app_diagnostics.png", full_page=True)
        b.close()


if __name__ == "__main__":
    main()
