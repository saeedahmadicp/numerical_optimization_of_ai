import { expect, test } from '@playwright/test';

// Every "Try this" preset in every lab must load without a render error: a preset changes the
// problem inside the open lab (e.g. to a larger matrix), a path a direct page load never takes.
const LABS = [
  'roots',
  'scalar',
  'line-search',
  'unconstrained',
  'systems',
  'least-squares',
  'linalg',
  'constrained',
  'lp',
  'global',
  'stochastic',
  'combinatorial',
  'integration',
  'interpolation',
  'differentiation',
  'regression',
];

for (const id of LABS) {
  test(`every Try-this preset loads: ${id}`, async ({ page, isMobile }) => {
    test.skip(isMobile, 'the rail is in the sheet on phones');
    test.setTimeout(90_000);
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.goto(`#/lab/${id}`);
    await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
    const presets = page
      .locator('section', { has: page.getByRole('heading', { name: 'Try this', exact: true }) })
      .locator('button[aria-pressed]');
    const n = await presets.count();
    for (let i = 0; i < n; i++) {
      const p = presets.nth(i);
      await p.scrollIntoViewIfNeeded();
      await p.click();
      await expect(p).toHaveAttribute('aria-pressed', 'true');
      await page.waitForTimeout(400);
      await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
      await expect(page.getByText('This page stopped with an error.')).toHaveCount(0);
    }
    expect(errors).toEqual([]);
  });
}
