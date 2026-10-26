/* Real upload -> OCR API -> review dialog. Requires a running local server and Playwright.
 * DOCSCAN_TEST_DOCUMENT=/path/to/original.jpg node tests/test_ui_live.cjs
 * Defaults to the smaller golden page. Uses an isolated browser context, without API mocks.
 */
const assert = require('node:assert/strict');
const path = require('node:path');
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({headless: true});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1050}, locale: 'en-GB'});
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(process.env.DOCSCAN_TEST_URL || 'http://127.0.0.1:8000/');
    if (process.env.DOCSCAN_API_KEY) {
      await page.locator('#settings').click();
      await page.locator('#key').fill(process.env.DOCSCAN_API_KEY);
      await page.locator('#saveSettings').click();
    }
    await page.locator('#files').setInputFiles(process.env.DOCSCAN_TEST_DOCUMENT || path.join(__dirname, 'golden/page2.jpg'));
    const response = page.waitForResponse(r => r.url().includes('/v1/scan'), {timeout: 120000});
    await page.locator('#go').click();
    const api = await response;
    assert.equal(api.status(), 200);
    const body = await api.json();
    const receipt = body.receipts.find(r => r.docType === 'kwitansi');
    assert(receipt, 'Kwitansi must be detected');
    assert.equal(receipt.transactionDate, '2026-10-02', 'API signing date');
    assert.equal(receipt.fields.transactionDate.value, '2026-10-02');
    assert(!receipt.reviewReasons.includes('Missing transactionDate'));
    await page.waitForFunction(() => !busy && done().length === 1);
    const uid = await page.evaluate(() => done()[0].result.receipts.find(r => r.docType === 'kwitansi').uid);
    // Click the receipt's actual review button, then verify the editable date.
    const row = page.locator('tr.row').filter({has: page.locator(`[data-f="date"][data-id="${uid}"]`)});
    await row.locator('.view-btn').click();
    assert.equal(await page.locator('#dInfo [data-f="date"]').inputValue(), '2026-10-02');
    assert(!(await page.locator('#dInfo').innerText()).includes('2026-10-13'));
    if (process.env.DOCSCAN_TEST_SCREENSHOT) await page.screenshot({path: process.env.DOCSCAN_TEST_SCREENSHOT});
    assert.deepEqual(errors, []);
    console.log('PASS real document upload, OCR API date, and review date input: 02/10/2026');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
