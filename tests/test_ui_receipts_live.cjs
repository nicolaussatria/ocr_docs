/* Real reused-paper upload against a running server; requires Playwright. */
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
    await page.locator('#files').setInputFiles(path.join(__dirname, 'golden/receipts-on-used-paper.jpg'));
    const response = page.waitForResponse(r => r.url().includes('/v1/scan'), {timeout: 120000});
    await page.locator('#go').click();
    const api = await response;
    assert.equal(api.status(), 200);
    const result = await api.json();
    assert.equal(result.receipts.length, 3);
    assert.equal(result.summary.totalAmount, 122000);
    assert.deepEqual(result.receipts.map(r => [r.transactionDate, r.amount]).sort(), [
      ['2026-09-28', 42000], ['2026-09-29', 40000], ['2026-10-02', 40000]
    ]);
    await page.waitForFunction(() => !busy && done().length === 1);
    assert.equal(await page.locator('tr.row').count(), 3);
    for (let i = 0; i < 3; i++) {
      await page.locator('tr.row').nth(i).locator('.view-btn').click();
      assert(!['', null].includes(await page.locator('#dInfo [data-f="date"]').inputValue()));
      await page.keyboard.press('Escape');
    }
    assert.deepEqual(errors, []);
    console.log('PASS: real upload shows exactly 3 complete fuel receipts, total Rp 122,000');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
