/* Browser regressions for the receipt workspace.
 * Run with Playwright installed: node tests/test_ui.cjs
 * Optional: DOCSCAN_UI_SCREENSHOTS=/absolute/path for preview PNGs.
 * API responses are mocked; the Python suite exercises the actual scan endpoint.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');
const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'docscan/static/review.html'), 'utf8');
const results = [];
const screenshotDir = process.env.DOCSCAN_UI_SCREENSHOTS;
function receipt(id, amount, status = 'verified') {
  return {
    id, docType: 'mobile_topup', category: 'pulsa', status, amount, confidence: .97,
    merchantName: 'TELKOMSEL', description: 'Pulsa Telkomsel', transactionDate: '2026-10-01',
    transactionAt: '2026-10-01T06:23:35', source: { file: 'topups.jpg', page: 1 }, bbox: [0, 0, 100, 100],
    fields: {amountPaid: {value: amount, confidence: .97, source: 'ocr', raw: String(amount)}},
    checks: [], reviewReasons: status === 'needs_review' ? ['Check the printed amount against the original.'] : [],
    rawText: ['TELKOMSEL PULSA'], fingerprint: id,
  };
}
const payload = { schemaVersion: '1.0', elapsedMs: 1200, summary: {}, receipts: [receipt('r1', 202000), receipt('r2', 77000, 'needs_review')] };
async function screenshot(page, name) {
  if (screenshotDir) {
    fs.mkdirSync(screenshotDir, {recursive: true});
    await page.locator('#toast').evaluate(el => { el.hidden = true; });
    await page.screenshot({path: path.join(screenshotDir, name + '.png'), fullPage: !name.endsWith('review')});
  }
}
async function downloadText(page, selector) {
  const promise = page.waitForEvent('download');
  await page.locator(selector).click();
  const download = await promise;
  return fs.readFileSync(await download.path(), 'utf8');
}
(async () => {
  const browser = await chromium.launch({headless: true});
  const page = await browser.newPage({viewport: {width: 1440, height: 1050}, reducedMotion: 'reduce'});
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  let failNext = false, responsePayload = payload, scanRequests = 0, lastKey;
  await page.route('http://docscan.test/**', async route => {
    const url = new URL(route.request().url());
    if (url.pathname === '/v1/scan') {
      scanRequests++;
      lastKey = route.request().headers()['x-api-key'];
      await new Promise(resolve => setTimeout(resolve, 100));
      if (failNext) { failNext = false; return route.fulfill({status: 401, contentType: 'application/json', body: JSON.stringify({detail:'invalid key'})}); }
      return route.fulfill({contentType: 'application/json', body: JSON.stringify(responsePayload)});
    }
    if (url.pathname === '/healthz') return route.fulfill({contentType: 'application/json', body:'{"ui":"test"}'});
    return route.fulfill({contentType: 'text/html', body: html});
  });
  try {
    await page.goto('http://docscan.test/');
    assert(await page.locator('#go').isDisabled());
    assert(await page.locator('#csv').isDisabled());
    assert(await page.locator('#emptyResults').isVisible());
    assert.equal(await page.locator('#guide, #help').count(), 0);
    const upload = await page.locator('#uploadPanel').boundingBox();
    const receipts = await page.locator('#results').boundingBox();
    assert.equal(Math.round(upload.width), Math.round(receipts.width));
    assert(upload.height < 350, 'Desktop upload should be compact');
    await screenshot(page, 'desktop-empty');
    await page.setViewportSize({width:390,height:844});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await screenshot(page, 'mobile-empty');
    await page.setViewportSize({width:1440,height:1050});
    results.push('Full-width compact upload, desktop/mobile layouts and removed guide');

    await page.locator('#settings').click();
    await page.locator('#key').fill('test-key');
    await page.locator('#saveSettings').click();
    assert.equal(await page.evaluate(() => localStorage.getItem('docscan.key')), 'test-key');
    results.push('Optional connection settings');

    await page.locator('#files').setInputFiles({name:'unsupported.txt',mimeType:'text/plain',buffer:Buffer.from('test')});
    assert.equal(await page.locator('.queue-item').count(), 0);
    assert((await page.locator('#toast').innerText()).includes('could not be added'));
    const file = {name:'topups.jpg',mimeType:'image/jpeg',buffer:Buffer.from('fixture')};
    await page.locator('#files').setInputFiles(file);
    // Re-select the same File, including its lastModified timestamp.
    await page.evaluate(() => {
      const transfer = new DataTransfer(); transfer.items.add(docs[0].file);
      $('files').files = transfer.files; $('files').dispatchEvent(new Event('change'));
    });
    assert.equal(await page.locator('.queue-item').count(), 1);
    assert(!(await page.locator('#go').isDisabled()));
    await page.locator('#go').click();
    await page.waitForFunction(() => !busy && done().length === 1);
    assert.equal(lastKey, 'test-key');
    assert.equal(scanRequests, 1);
    assert.equal(await page.locator('tr.row').count(), 2);
    assert.equal(await page.locator('#countReview').innerText(), '1');
    assert((await page.locator('.metric.total').innerText()).includes('279.000'));
    results.push('Upload validation, deduplication, scan request and receipt totals');

    await page.locator('[data-filter="attention"]').click();
    assert.equal(await page.locator('tr.row').count(), 1);
    await page.locator('[data-filter="all"]').click();
    await page.locator('#search').fill('no such merchant');
    assert.equal(await page.locator('tr.row').count(), 0);
    assert(await page.locator('.empty-filter').isVisible());
    await page.locator('#search').fill('');
    results.push('Receipt filters and search');

    await page.locator('#reviewNext').click();
    assert.equal(await page.locator('#detailPosition').innerText(), '2 of 2');
    await page.keyboard.press('Escape');
    results.push('Review next opens the first receipt needing attention');

    await page.locator('tr.row .view-btn').first().click();
    assert(await page.locator('#dlg').isVisible());
    await page.locator('#dInfo [data-f="merchant"]').fill('TELKOMSEL OFFICE');
    await page.locator('#dInfo [data-f="merchant"]').press('Tab');
    assert((await page.locator('#dInfo [data-f="note"]').inputValue()).includes('TELKOMSEL OFFICE'));
    await page.locator('#dInfo [data-f="type"]').selectOption('internet');
    await page.locator('#dInfo [data-f="date"]').fill('2026-10-02');
    await page.locator('#dInfo [data-f="date"]').press('Tab');
    await page.locator('#dInfo [data-f="amount"]').fill('-1');
    await page.locator('#markReviewed').click();
    assert.equal(await page.evaluate(() => !!edits['d1/r1']?.reviewed), false);
    await page.locator('#dInfo [data-f="amount"]').fill('205000');
    await page.locator('#dInfo [data-f="amount"]').press('Tab');
    await page.locator('#dInfo [data-f="note"]').fill('Office internet, October');
    await page.locator('#markReviewed').click();
    assert.equal(await page.locator('#countReviewed').innerText(), '1');
    assert.equal(await page.locator('#countReview').innerText(), '1');
    await page.locator('#nextReceipt').click();
    assert.equal(await page.locator('#detailPosition').innerText(), '2 of 2');
    await page.keyboard.press('Escape');
    assert(!(await page.locator('#dlg').isVisible()));
    results.push('Side-by-side review, field edits, validation, review status and keyboard dismissal');

    const csv = await downloadText(page, '#csv');
    assert(csv.includes('2026-10-02'));
    assert(csv.includes('"Office internet, October"'));
    assert(csv.includes('205000'));
    assert(csv.includes('282000'));
    const json = JSON.parse(await downloadText(page, '#json'));
    assert.equal(json.receipts[0].amount, 202000);
    assert.equal(json.receipts[0].userEdits.amount, 205000);
    assert.equal(json.receipts[0].userEdits.type, 'internet');
    assert.equal(json.receipts[0].userEdits.reviewed, true);
    assert(!('uid' in json.receipts[0]));
    assert(!('image' in json.receipts[0]));
    results.push('CSV corrections, totals, quoting and original-data JSON export');

    failNext = true;
    await page.locator('#files').setInputFiles({name:'second.pdf',mimeType:'application/pdf',buffer:Buffer.from('fixture2')});
    await page.locator('#go').click();
    await page.waitForFunction(() => !busy && docs.some(d => d.state === 'error'));
    assert((await page.locator('.error-text').innerText()).includes('Connection settings'));
    await page.locator('[data-act="retry"]').click();
    await page.locator('#go').click();
    await page.waitForFunction(() => !busy && done().length === 2);
    assert(await page.locator('#oneWrap').isVisible());
    assert((await page.locator('#exportWarning').innerText()).includes('still to review'));
    const separate = await downloadText(page, '#csv');
    assert(separate.includes('topups.jpg') && separate.includes('second.pdf'));
    await page.locator('#one').check();
    const combined = await downloadText(page, '#csv');
    assert(!combined.includes('topups.jpg'));
    assert(combined.includes('561000'));
    results.push('API errors, retry, duplicate warnings, separate and combined claims');

    await page.setViewportSize({width:390,height:844});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.locator('tr.row .view-btn').first().click();
    assert(await page.locator('#dlg').isVisible());
    assert(await page.evaluate(() => document.getElementById('dlg').scrollWidth <= document.getElementById('dlg').clientWidth));
    await screenshot(page, 'mobile-review');
    await page.keyboard.press('Escape');
    results.push('Populated mobile layout and review dialog');

    // Render an actual OCR response when supplied, otherwise the deterministic fixture.
    if (process.env.DOCSCAN_UI_SCAN_RESULT) {
      const actual = JSON.parse(fs.readFileSync(process.env.DOCSCAN_UI_SCAN_RESULT, 'utf8'));
      await page.evaluate(actual => {
        docs.splice(0); for (const key of Object.keys(edits)) delete edits[key];
        actual.receipts.forEach(r => {r.uid = 'preview/' + r.id;});
        docs.push({key:'preview',file:{name:'Expense receipts.jpg',size:604490,type:'image/jpeg'},state:'done',result:actual});
        filter = 'all'; $('search').value = ''; render();
      }, actual);
    }
    await page.setViewportSize({width:1440,height:1050});
    await page.evaluate(() => scrollTo(0,0));
    await screenshot(page, 'desktop-results');
    await page.locator('tr.row .view-btn').first().click();
    await screenshot(page, 'desktop-review');
    await page.keyboard.press('Escape');
    await page.locator('[data-act="remove"]').first().click();
    assert.equal(errors.length, 0, errors.join('\n'));
    console.log(results.map(x => 'PASS ' + x).join('\n'));
    console.log('All UI browser checks passed; no browser errors.');
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exitCode = 1; });
