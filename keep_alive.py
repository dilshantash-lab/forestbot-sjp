"""keep_alive.py - opens the ForestBot app like a visitor, and presses
'Yes, get this app back up' if the app is asleep."""
import sys
from playwright.sync_api import sync_playwright

APP_URL = "https://forestbot-sjp-bdsjjng84fm9kurmb3il2j.streamlit.app"
WAKE_TEXT = "get this app back up"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(APP_URL, timeout=90000)
    page.wait_for_timeout(8000)
    button = page.get_by_text(WAKE_TEXT)
    if button.count() > 0:
        print("App was asleep. Pressing the wake-up button...")
        button.first.click()
        page.wait_for_timeout(90000)
        if page.get_by_text(WAKE_TEXT).count() > 0:
            print("Still asleep after waiting.")
            browser.close()
            sys.exit(1)
        print("App is awake now.")
    else:
        print("App is already awake.")
    browser.close()
