const { test, expect } = require('@playwright/test');

test('KYMA renders a live synthetic biosignal scope', async ({ page, request }) => {
  await page.goto('/');
  await expect(page.locator('#emg-canvas')).toBeVisible();

  const response = await request.post('/api/stream/start', {
    data: { source: 'synthetic', synthetic_scenario: 'clean' },
  });
  expect(response.ok()).toBeTruthy();

  await page.waitForFunction(() => window.S && Number(window.S.emgTotal || 0) > 30);
  const panel = page.locator('#pred-summary');
  await expect(panel).toBeVisible();

  const canvasStats = await page.locator('#emg-canvas').evaluate((canvas) => {
    const ctx = canvas.getContext('2d');
    const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let colored = 0;
    let opaque = 0;
    for (let i = 0; i < pixels.length; i += 4) {
      const r = pixels[i];
      const g = pixels[i + 1];
      const b = pixels[i + 2];
      const a = pixels[i + 3];
      if (a > 0) opaque += 1;
      if (a > 0 && Math.max(r, g, b) - Math.min(r, g, b) > 35) colored += 1;
    }
    return { width: canvas.width, height: canvas.height, colored, opaque };
  });

  expect(canvasStats.width).toBeGreaterThan(300);
  expect(canvasStats.height).toBeGreaterThan(200);
  expect(canvasStats.colored).toBeGreaterThan(1000);

  await page.screenshot({
    path: 'output/playwright/ci-kyma-smoke.png',
    fullPage: true,
  });
});
