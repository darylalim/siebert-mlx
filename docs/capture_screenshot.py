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

# Not a project dependency (see the module docstring), so ty cannot resolve it
# here; the suppression is load-bearing (removing it fails `ty check .`).
from playwright.sync_api import sync_playwright  # ty: ignore[unresolved-import]

OUT = "screenshot-dark.png"
# layout="wide" has no content cap, so width is a real choice: with the 300px
# sidebar open the main block is the window minus 300 and the results grid the
# window minus ~494. 1440 is a common laptop viewport, and the narrowest where
# the distribution chart in the metric row still gets a comfortable card (at
# 1280 it is squeezed to ~215px; at 1100 it wraps onto its own row). Wider
# only shrinks the text once GitHub scales the image to its column. Never
# below 864 (a padding breakpoint makes the grid non-monotonic) or 768 (an
# open sidebar overlays the main area instead of pushing it aside).
WIDTH = 1440
DEFAULT_PORT = 8501

# The toolbar lands in the frame otherwise. The stMain rule is the headed-only
# scrollbar: stMain keeps ~160px of overflow past the trimmed viewport because
# of the shell's bottom spacer, and macOS Chrome paints a real bar for it,
# which both shows and narrows the main area (and the grid in it) by its own
# width. Page scroller only -- the results grid's own scrollbar is a real
# affordance. `header` matches only stHeader, the main-area toolbar: the
# sidebar's own header (stSidebarHeader, holding the collapse button) is not a
# <header> element and stays in the shot, which is right -- it is real UI.
CHROME_CSS = """
header, [data-testid="stToolbar"],
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
        # Dark comes from the context, not from a Streamlit flag: the menu
        # defaults to System, which follows prefers-color-scheme, so this
        # exercises the shipped [theme.dark] in .streamlit/config.toml. A
        # `--theme.base=dark` flag would do nothing -- with [theme.dark]
        # present the frontend sets base itself for each mode -- so the
        # browser's color scheme is the only switch.
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
        # "Classification complete!" is a toast (~4s), not part of the page:
        # wait it out so the shot shows the state the user is left with.
        page.wait_for_selector(
            '[data-testid="stToast"]', state="detached", timeout=30_000
        )
        page.wait_for_timeout(1_000)

        info = page.evaluate(MEASURE)
        height = info["contentBottom"]
        print(f"measured {info} -> capturing {WIDTH}x{height}")

        page.set_viewport_size({"width": WIDTH, "height": height})
        page.wait_for_timeout(800)

        # A headed window cannot exceed the physical screen, so the viewport can
        # come back short and clip the capture the same way full_page=True does
        # -- in width too, now that WIDTH is 1440, which a scaled display can
        # undercut. CDP is not window-bounded.
        inner_w, inner_h = page.evaluate(
            "() => [window.innerWidth, window.innerHeight]"
        )
        if inner_w < WIDTH or inner_h < height:
            print(f"window capped below {WIDTH}x{height}; overriding metrics via CDP")
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

        # The pointer is still over Classify from the click, which would
        # capture the button in its hover color; park it on the (hidden)
        # header's blank corner.
        page.mouse.move(WIDTH - 5, 5)
        page.wait_for_timeout(300)
        page.screenshot(path=out_path)
        browser.close()

    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
