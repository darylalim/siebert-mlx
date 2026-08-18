"""Recapture `docs/screenshot-dark.png` by driving a running app with Playwright.

    uv run streamlit run streamlit_app.py            # in one shell
    uv run --with playwright python docs/capture_screenshot.py [port]

Playwright is deliberately *not* a project dependency -- it is a manual tool
run a few times a year, and `--with` keeps it out of `uv sync` and CI. It
drives the **system** Chrome (`channel="chrome"`), so nothing is downloaded.

Every non-obvious line here is a trap documented in CLAUDE.md's README
Screenshots section; read that before changing any of them.
"""

import sys

# Not a project dependency (see the module docstring), so it cannot resolve
# here -- same suppression as mlx.core's missing stubs in streamlit_app.py.
from playwright.sync_api import sync_playwright  # ty: ignore[unresolved-import]

OUT = "screenshot-dark.png"
# Above ~800 only adds side margin: the centered layout caps content at 736px.
WIDTH = 800
DEFAULT_PORT = 8501

# The toolbar lands in the frame otherwise. The stMain rule is the headed-only
# scrollbar: stMain keeps ~160px of overflow past the trimmed viewport because
# of the shell's bottom spacer, and macOS Chrome paints a real bar for it,
# which both shows and shifts the centered column left by its own width. Page
# scroller only -- the results grid's own scrollbar is a real affordance.
CHROME_CSS = """
header, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }
[data-testid="stMainBlockContainer"] { padding-top: 2rem !important; }
section[data-testid="stMain"] { scrollbar-width: none !important; }
section[data-testid="stMain"]::-webkit-scrollbar {
  width: 0 !important; height: 0 !important;
}
"""

# stMain is the scroller -- not stAppViewContainer (its scrollHeight equals the
# viewport) and not document.body (literally 0). scrollHeight itself overshoots
# by the bottom spacer, so the capture height is the last block's bottom edge.
MEASURE = """() => {
  const main = document.querySelector('section[data-testid="stMain"]');
  const block = document.querySelector('[data-testid="stMainBlockContainer"]');
  return {
    scrollHeight: main.scrollHeight,
    contentBottom: Math.ceil(
      block.lastElementChild.getBoundingClientRect().bottom + main.scrollTop),
  };
}"""


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    out_path = __file__.rsplit("/", 1)[0] + "/" + OUT

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            channel="chrome",
            # Headed, so it renders on the real display. Headless does not show
            # the scrollbar CHROME_CSS suppresses, so the two are not
            # interchangeable.
            headless=False,
            args=["--window-position=0,0"],
        )
        # Dark comes from the context, not from a Streamlit flag: with no
        # .streamlit/config.toml the app follows prefers-color-scheme, so this
        # exercises the shipped default-theme path. `--theme.base=dark` would
        # instead screenshot a custom theme the app does not ship.
        context = browser.new_context(
            viewport={"width": WIDTH, "height": 1000},
            color_scheme="dark",
        )
        page = context.new_page()
        page.goto(f"http://localhost:{port}", wait_until="networkidle", timeout=120_000)
        page.wait_for_timeout(3_000)
        page.add_style_tag(content=CHROME_CSS)

        # The README's alt text says "classifying sample data", so classify.
        page.locator('button:has-text("Sample")').first.click()
        page.wait_for_timeout(2_000)
        page.locator('button:has-text("Classify")').first.click()
        page.wait_for_selector("text=Classification complete", timeout=180_000)
        page.wait_for_timeout(2_500)

        info = page.evaluate(MEASURE)
        height = info["contentBottom"]
        print(f"measured {info} -> capturing {WIDTH}x{height}")

        page.set_viewport_size({"width": WIDTH, "height": height})
        page.wait_for_timeout(800)

        # A headed window cannot exceed the physical screen, so the viewport can
        # come back short and clip the capture the same way full_page=True does.
        # CDP is not window-bounded.
        if page.evaluate("() => window.innerHeight") < height:
            print(f"window capped below {height}px; overriding metrics via CDP")
            cdp = context.new_cdp_session(page)
            cdp.send(
                "Emulation.setDeviceMetricsOverride",
                {
                    "width": WIDTH,
                    "height": height,
                    "deviceScaleFactor": 1,
                    "mobile": False,
                },
            )
            page.wait_for_timeout(800)

        page.screenshot(path=out_path)
        browser.close()

    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
