/* Original optical photo -> live OCR -> visible review details. Requires Playwright and server. */
const assert = require('node:assert/strict');
const path = require('node:path');
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({headless: true});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1050}});
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(process.env.DOCSCAN_TEST_URL || 'http://127.0.0.1:8000/');
    if (process.env.DOCSCAN_API_KEY) {
      await page.locator('#settings').click();
      await page.locator('#key').fill(process.env.DOCSCAN_API_KEY);
      await page.locator('#saveSettings').click();
    }
    await page.locator('#files').setInputFiles(path.join(__dirname, 'golden/optical-handwritten.jpg'));
    const response = page.waitForResponse(r => r.url().includes('/v1/scan'), {timeout: 120000});
    await page.locator('#go').click();
    const api = await response;
    assert.equal(api.status(), 200);
    const result = await api.json();
    assert.equal(result.receipts.length, 1);
    const receipt = result.receipts[0];
    assert.equal(receipt.transactionDate, '2026-10-02');
    assert.equal(receipt.amount, 350000);
    assert.equal(receipt.fields.leftSphere.source, 'inferred');
    await page.waitForFunction(() => !busy && done().length === 1);
    await page.locator('tr.row .view-btn').click();
    assert.equal(await page.locator('#dInfo [data-f="date"]').inputValue(), '2026-10-02');
    assert.equal(await page.locator('#dInfo [data-f="patient"]').inputValue(), 'Rita');
    const optical = page.locator('details').filter({has: page.locator('summary', {hasText: 'Optical receipt details'})});
    assert.equal(await optical.getAttribute('open'), '');
    const text = await optical.innerText();
    for (const expected of ['Palembang', '082151096882', 'Photochromic', '-2.25', 'inferred', 'Unclear']) {
      assert(text.includes(expected), expected);
    }
    assert.deepEqual(errors, []);
    if (process.env.DOCSCAN_TEST_SCREENSHOT) await page.screenshot({path: process.env.DOCSCAN_TEST_SCREENSHOT});
    console.log('PASS: original optical receipt date, amount, patient and visible optical details');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
