"""Optional browser QA: pip install playwright==1.55.0; python -m playwright install chromium."""
from pathlib import Path
import json
from playwright.sync_api import sync_playwright

root=Path(__file__).resolve().parents[1]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=["--no-sandbox"])
    page=browser.new_page(viewport={"width":1440,"height":1080},device_scale_factor=1)
    errors=[];page.on("pageerror",lambda e:errors.append(str(e)))
    page.goto((root/"demo/liquiditylens_demo.html").as_uri());page.wait_for_timeout(300)
    assert page.locator("#company option").count()>0
    page.screenshot(path=str(root/"reports/figures/demo_desktop.png"),full_page=True)
    page.locator("#show-outcome").check();assert page.locator("#outcome").is_visible()
    assert " pp " in page.locator("#outcome").inner_text()
    page.get_by_role("button",name="Review queue").click();assert page.locator("#queue-body tr").count()>0
    page.get_by_role("button",name="Inspect").first.click()
    page.get_by_role("button",name="Model evidence").click();assert page.locator("#metric-body tr").count()>=4
    page.locator("#metric-cohort").select_option("matched_text");assert page.locator("#metric-body tr").count()>=4
    page.locator("#metric-group").select_option("test_unseen");assert page.locator("#metric-body tr").count()>=4
    page.locator("#metric-group").select_option("test_seen")
    page.screenshot(path=str(root/"reports/figures/demo_research.png"),full_page=True)
    page.set_viewport_size({"width":390,"height":844});page.get_by_role("button",name="Company review").click()
    page.screenshot(path=str(root/"reports/figures/demo_mobile.png"),full_page=True)
    overflow=page.evaluate("document.documentElement.scrollWidth > window.innerWidth")
    assert not overflow and not errors
    report={"passed":True,"page_errors":errors,"mobile_horizontal_overflow":overflow,
            "desktop_width":1440,"mobile_width":390,"interactive_paths_checked":["company review","outcome reveal","quarter queue","case inspection","numeric/text metrics","unseen-company metrics"]}
    (root/"reports/demo_qa.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));browser.close()

