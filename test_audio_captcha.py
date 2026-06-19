"""
Standalone test for the OFFLINE audio CAPTCHA feature.
Run: python test_audio_captcha.py

Opens the GST portal, clicks the sound button, downloads the audio, transcribes
it offline with HuggingFace Whisper (no API key, no ffmpeg), and prints the digits.

Prerequisite: internet access on first run so HuggingFace can download
openai/whisper-tiny (~39 MB) into ~/.cache/huggingface. All subsequent runs
are fully offline.
"""

import urllib.request

from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth

# Reuse the exact production logic so the test mirrors the real app.
from main import _transcribe_audio, _words_to_digits


def test_audio():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled", "--start-maximized"]
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="en-IN",
            timezone_id="Asia/Kolkata",
            viewport={"width": 1280, "height": 800},
            extra_http_headers={"Accept-Language": "en-IN,en;q=0.9"},
        )
        page = context.new_page()
        Stealth().apply_stealth_sync(page)

        print("[test] Navigating to GST portal...")
        page.goto("https://services.gst.gov.in/services/searchtp", timeout=60000)
        page.wait_for_load_state("networkidle")
        print("[test] Page loaded.")

        # ── Step 1: List ALL buttons on the page ─────────────────────────────
        buttons = page.get_by_role("button").all()
        print(f"\n[test] Found {len(buttons)} buttons on the page:")
        for i, btn in enumerate(buttons):
            try:
                txt = btn.inner_text().strip() or "(no text)"
                aria = btn.get_attribute("aria-label") or ""
                title = btn.get_attribute("title") or ""
                print(f"  [{i}] text='{txt}'  aria-label='{aria}'  title='{title}'")
            except Exception as e:
                print(f"  [{i}] Error reading button: {e}")

        # ── Step 2: Intercept ALL network responses ───────────────────────────
        all_responses = []

        def log_response(response):
            all_responses.append(response.url)

        page.on("response", log_response)

        # ── Step 3: Try clicking sound button by common patterns ──────────────
        sound_btn = None
        for btn in buttons:
            try:
                aria = (btn.get_attribute("aria-label") or "").lower()
                title = (btn.get_attribute("title") or "").lower()
                txt = btn.inner_text().strip().lower()
                if any(kw in aria + title + txt for kw in ["sound", "audio", "speaker", "listen"]):
                    sound_btn = btn
                    print(f"\n[test] Found sound button by keyword: aria='{aria}' title='{title}' txt='{txt}'")
                    break
            except Exception:
                pass

        if sound_btn is None:
            print("\n[test] Could not find sound button by keyword - falling back to nth(4)")
            sound_btn = page.get_by_role("button").nth(4)

        print("[test] Clicking sound button...")
        sound_btn.click()
        print("[test] Waiting 12 seconds for audio to load...")
        page.wait_for_timeout(12000)
        page.remove_listener("response", log_response)

        # ── Step 4: Show ALL intercepted URLs ─────────────────────────────────
        print(f"\n[test] {len(all_responses)} total network responses intercepted:")
        for url in all_responses:
            print(f"  {url}")

        audio_url = None
        for url in all_responses:
            if any(ext in url.lower() for ext in [".mp3", ".wav", ".ogg", "audio", "sound", "captcha"]):
                audio_url = url
                print(f"\n[test] Matched audio URL: {url}")
                break

        if not audio_url:
            print("\n[test] No audio URL found. Check the list above for what the portal actually loaded.")
            context.close()
            browser.close()
            return

        # ── Step 5: Download the audio ────────────────────────────────────────
        print(f"\n[test] Downloading audio from: {audio_url}")
        try:
            req = urllib.request.Request(
                audio_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0"}
            )
            audio_data = urllib.request.urlopen(req).read()
            print(f"[test] Downloaded {len(audio_data)} bytes")

            # Save to disk so you can listen to it manually
            with open("captcha_audio_sample.wav", "wb") as f:
                f.write(audio_data)
            print("[test] Saved as captcha_audio_sample.wav - open it to hear what was downloaded")
        except Exception as e:
            print(f"[test] Download failed: {e}")
            context.close()
            browser.close()
            return

        # ── Step 6: Transcribe OFFLINE with HuggingFace Whisper ──────────────
        print("\n[test] Transcribing offline with HuggingFace Whisper...")
        text = _transcribe_audio(audio_data)
        if not text:
            print("[test] Whisper produced no text. Check captcha_audio_sample.wav manually.")
        else:
            print(f"[test] Transcription: '{text}'")
            digits = _words_to_digits(text)
            print(f"[test] Extracted digits: '{digits}' (length: {len(digits)})")
            if len(digits) == 6:
                print("[test] SUCCESS - 6 digits extracted correctly!")
            else:
                print(f"[test] Got {len(digits)} digits instead of 6")

        context.close()
        browser.close()


if __name__ == "__main__":
    test_audio()
