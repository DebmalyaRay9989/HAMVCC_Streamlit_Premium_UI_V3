"""Optional browser test. Start app.py first; pip install playwright; playwright install chromium."""
import os
from pathlib import Path
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    options={'headless':True,'args':['--no-sandbox']}
    if os.environ.get('CHROMIUM_PATH'):
        options['executable_path']=os.environ['CHROMIUM_PATH']
        options['args'] += ['--disable-dev-shm-usage','--no-zygote','--single-process']
    browser=p.chromium.launch(**options)
    page=browser.new_page(viewport={'width':1440,'height':1050},device_scale_factor=1)
    errors=[]
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto('http://127.0.0.1:8050',wait_until='networkidle')
    page.locator('#result-bars .main-svg').first.wait_for()
    assert '0.0176' in page.locator('#result-stats').inner_text()
    with page.expect_download() as download:
        page.locator('#export-runs').click()
    assert download.value.suggested_filename.endswith('.csv')
    screenshot=os.environ.get('SCREENSHOT_PATH')
    if screenshot:page.screenshot(path=screenshot,full_page=True)
    page.get_by_text('02  Gate lab',exact=True).click()
    page.locator('#gate-weights .main-svg').first.wait_for()
    page.wait_for_function("document.querySelector('#gate-status').innerText.includes('0.752')")
    for box in page.locator('#available input').all():box.uncheck()
    page.wait_for_function("document.querySelector('#gate-status').innerText.includes('Unassignable')")
    page.get_by_text('03  Documents',exact=True).click()
    page.locator('#doc-scatter .main-svg').first.wait_for()
    page.wait_for_function("document.querySelector('#doc-stats').innerText.includes('199')")
    page.locator('#run-cluster').click()
    page.locator('#fresh-output .main-svg').first.wait_for()
    assert '0.2623' in page.locator('#fresh-output').inner_text()
    with page.expect_download() as download:
        page.locator('#export-clusters').click()
    assert download.value.suggested_filename.endswith('.zip')
    sample=b'document_id,x,y\na,0,0\nb,0.1,0.2\nc,3,3\nd,3.2,3.1\n'
    page.locator('#csv-upload input[type=file]').set_input_files({'name':'sample.csv','mimeType':'text/csv','buffer':sample})
    page.wait_for_function("document.querySelector('#upload-status').innerText.includes('4 rows')")
    page.locator('#run-cluster').click()
    page.wait_for_function("document.querySelector('#fresh-output').innerText.includes('Uploaded CSV')")
    page.get_by_text('04  Evidence',exact=True).click()
    page.get_by_text('Keep every claim traceable.',exact=True).wait_for()
    page.set_viewport_size({'width':390,'height':844})
    page.get_by_text('01  Results',exact=True).click()
    page.locator('#result-bars .main-svg').first.wait_for()
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 2')
    assert not errors,errors
    print('PASS: tabs, plots, gate rejection, CSV upload, clustering, downloads, mobile width; no page errors')
    browser.close()
