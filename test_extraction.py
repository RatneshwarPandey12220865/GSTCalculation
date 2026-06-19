"""
Standalone extraction tester.

Usage:
  1. Run this script
  2. A browser opens on the GST portal
  3. Manually fill the GSTIN and CAPTCHA and click Search
  4. Once the detail page loads, press ENTER in this terminal
  5. The script dumps all extracted data to the console

This lets you test/debug extraction WITHOUT going through the CAPTCHA solver.
"""

import json
from playwright.sync_api import sync_playwright


def extract(page, gstin: str) -> dict:
    data = page.evaluate("""() => {
        const out = {};
        const txt = el => (el ? el.innerText.trim() : "");

        // ── 1. Main grid: #lottable .col-sm-6 ────────────────────────────────
        document.querySelectorAll('#lottable .col-sm-6').forEach(col => {
            const bold = col.querySelector('b');
            if (!bold) return;
            const label = txt(bold).replace(/:$/, '').trim();
            if (!label) return;
            const clone = col.cloneNode(true);
            clone.querySelector('b').remove();
            const value = clone.innerText.trim();
            if (value) out[label] = value;
        });

        // ── 2. Fallback for fields outside #lottable ──────────────────────────
        if (Object.keys(out).length < 3) {
            document.querySelectorAll('b, strong').forEach(bold => {
                const label = txt(bold).replace(/:$/, '').trim();
                if (!label || label.length > 60) return;
                let next = bold.parentElement && bold.parentElement.nextElementSibling;
                if (!next) next = bold.nextElementSibling;
                if (next) out[label] = txt(next);
            });
        }

        // ── 3. Nature panels ──────────────────────────────────────────────────
        document.querySelectorAll('.panel').forEach(panel => {
            const headingEl = panel.querySelector('.panel-heading, .panel-title')
                           || panel.querySelector('[class*="heading"]');
            const body = panel.querySelector('.panel-body, [class*="body"]');
            if (!headingEl || !body) return;

            const h = txt(headingEl).toLowerCase();
            if (h.includes('nature of core business')) {
                const link = body.querySelector('a');
                out['Nature Of Core Business Activity'] = link ? txt(link) : txt(body);
            }
            if (h.includes('nature of business activities')) {
                const items = body.querySelectorAll('li');
                if (items.length > 0) {
                    out['Nature of Business Activities'] = Array.from(items)
                        .map(li => txt(li).replace(/^\\d+\\.\\s*/, ''))
                        .join('\\n');
                } else {
                    out['Nature of Business Activities'] = txt(body);
                }
            }
        });

        // ── 4. Goods & Services table ─────────────────────────────────────────
        const goodsHSN = [], goodsDesc = [], servicesHSN = [], servicesDesc = [];
        document.querySelectorAll('table').forEach(table => {
            const allText = txt(table).toLowerCase();
            if (!allText.includes('goods') && !allText.includes('hsn')) return;

            Array.from(table.querySelectorAll('tr')).forEach(row => {
                const cells = row.querySelectorAll('td');
                if (cells.length === 0) return;
                const c0 = txt(cells[0]);
                if (!c0 || ['hsn','goods','services'].includes(c0.toLowerCase())) return;
                const c1 = cells[1] ? txt(cells[1]) : '';
                const c2 = cells[2] ? txt(cells[2]) : '';
                const c3 = cells[3] ? txt(cells[3]) : '';
                if (c0) { goodsHSN.push(c0);  goodsDesc.push(c1); }
                if (c2) { servicesHSN.push(c2); servicesDesc.push(c3); }
            });
        });
        out['Goods HSN']         = goodsHSN.join('\\n');
        out['Goods Description'] = goodsDesc.join('\\n');
        out['Services HSN']      = servicesHSN.join('\\n');
        out['Services Desc']     = servicesDesc.join('\\n');

        // ── 5. Also dump the entire page DOM keys for debugging ───────────────
        out['__all_bold_labels__'] = Array.from(document.querySelectorAll('b, strong'))
            .map(el => el.innerText.trim())
            .filter(t => t && t.length < 80)
            .join(' | ');

        return out;
    }""")

    print("\n" + "="*60)
    print(f"GSTIN: {gstin}")
    print("="*60)

    debug_keys   = data.pop("__debug_keys__", "")
    all_labels   = data.pop("__all_bold_labels__", "")
    print(f"\nAll <b>/<strong> labels found on page:\n  {all_labels}\n")
    print("Extracted fields:")
    for k, v in data.items():
        if v:
            print(f"  [{k}] = {repr(v)}")

    print("\nEmpty / missing fields:")
    for k, v in data.items():
        if not v:
            print(f"  [{k}] = (empty)")

    return data


def main():
    gstin = input("Enter GSTIN to test (e.g. 06AJWPP4177G1ZL): ").strip().upper()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, args=["--start-maximized"])
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="en-IN",
            timezone_id="Asia/Kolkata",
            viewport={"width": 1280, "height": 800},
        )
        page = context.new_page()
        page.goto("https://services.gst.gov.in/services/searchtp", timeout=60000)

        print("\n" + "="*60)
        print("Browser is open. Please:")
        print("  1. Fill the GSTIN in the search box")
        print("  2. Solve the CAPTCHA manually")
        print("  3. Click Search and wait for the detail page to load")
        print("  4. Come back here and press ENTER")
        print("="*60)
        input("\nPress ENTER when the detail page is fully loaded...")

        extract(page, gstin)

        input("\nPress ENTER to close the browser...")
        context.close()
        browser.close()


if __name__ == "__main__":
    main()
