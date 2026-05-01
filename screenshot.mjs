import { chromium } from 'playwright';
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1440, height: 900 } });
await p.goto('http://localhost:3000/', { waitUntil: 'networkidle' });
await p.waitForTimeout(2000);
// Click to expand data flow section
await p.evaluate(() => {
  const panel = document.querySelector('.border-l');
  if (!panel) return;
  const btns = panel.querySelectorAll('button');
  for (const b of btns) {
    if (b.textContent?.includes('数据流路径')) { b.click(); break; }
  }
});
await p.waitForTimeout(500);
await p.screenshot({ path: '/tmp/analysis-flow.png' });
console.log('screenshot saved to /tmp/analysis-flow.png');
await b.close();
