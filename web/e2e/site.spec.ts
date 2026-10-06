import { expect, test, type Page } from '@playwright/test';

const noOverflow = async (page: Page) =>
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth),
  ).toBeLessThanOrEqual(0);

test('methods: facets and text filter narrow the catalog; a row opens the method page', async ({
  page,
  isMobile,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('#/methods');
  await expect(page.getByRole('heading', { level: 1, name: 'Methods' })).toBeVisible();
  await expect(page.getByText(/^168 of 168$/)).toBeVisible();

  if (isMobile) await page.getByRole('button', { name: /Filters/ }).click();
  const derivatives = page.getByRole('group', { name: 'Derivatives' });
  await derivatives.getByRole('button', { name: /Uses ∇²f/ }).click();
  await expect(page).toHaveURL(/needs=hess/);
  await expect(page.getByText(/^\d+ of 168$/)).not.toHaveText('168 of 168');
  await expect(page.getByRole('link', { name: /Halley/ })).toBeVisible();
  await expect(page.getByRole('link', { name: /^Bisection/ })).toHaveCount(0);

  await page.getByRole('searchbox', { name: 'Filter methods' }).fill('nocedal 3.3');
  await expect(page.getByRole('link', { name: /Damped Newton/ })).toBeVisible();
  await page.getByRole('button', { name: 'Clear all' }).click();
  await expect(page.getByText(/^168 of 168$/)).toBeVisible();

  await page.getByRole('link', { name: /^BFGS/ }).click();
  await expect(page).toHaveURL(/#\/method\/bfgs$/);
  await expect(page.getByRole('heading', { level: 1, name: 'BFGS' })).toBeVisible();
  await noOverflow(page);
  expect(errors).toEqual([]);
});

test('method page: live run, parameters, Python call and a live parity check', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('#/method/bisection');
  await expect(page.getByRole('heading', { level: 1, name: 'Bisection' })).toBeVisible();
  // The TS port replays the Python record in the browser.
  await expect(page.getByText('1 of 1 case matches')).toBeVisible();
  const parity = page.getByRole('region', { name: 'Parity cases' });
  await expect(parity.getByText('first 10 iterates')).toBeVisible();
  await expect(parity.getByText('33 = 33')).toBeVisible();
  // The live run: a curve, a convergence chart, a player and the MethodCard.
  await expect(page.getByRole('img', { name: /Square root of 2: the curve of f/ })).toBeVisible();
  await expect(page.getByRole('slider', { name: 'Iteration', exact: true })).toBeVisible();
  await expect(page.getByRole('region', { name: 'Bisection details' })).toBeVisible();
  // Parameters from the ParamSpecs, the Python call with its recorded output.
  const params = page.getByRole('region', { name: 'Parameters' });
  await expect(params.getByText('xtol', { exact: true })).toBeVisible();
  await expect(page.getByText('numopt.run("bisection", problems.get("sqrt2"))')).toBeVisible();
  await expect(page.getByText('(True, 33)')).toBeVisible();
  // Deep link into the lab with the method preselected.
  await expect(page.getByRole('link', { name: /Open in the root finding lab/ })).toHaveAttribute(
    'href',
    '#/lab/roots?m=bisection~0&p=sqrt2',
  );
  await page.getByRole('link', { name: /Next/ }).click();
  await expect(page).toHaveURL(/#\/method\/brent$/);
  await noOverflow(page);
  expect(errors).toEqual([]);
});

test('method page: a 2-D method draws its path on the landscape', async ({ page }) => {
  await page.goto('#/method/nelder_mead');
  await expect(page.getByRole('img', { name: /Contour plot of Rosenbrock/ })).toBeVisible();
  await expect(page.getByText(/2 of 2 cases match/)).toBeVisible();
  await page.goto('#/method/not_a_method');
  await expect(
    page.getByRole('heading', { name: 'This page is not in the catalog.' }),
  ).toBeVisible();
});

test('research: the studies, then one study with math, figures and section links', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('#/research');
  await expect(page.getByRole('heading', { level: 1, name: 'Research' })).toBeVisible();
  await expect(page.getByText(/^\d+ studies/)).toBeVisible();
  const study = page.getByRole('link', { name: /Certified step-size schedules/ }).first();
  await study.click();
  await expect(page).toHaveURL(/#\/research\/certified-stepsize-schedules$/);
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Certified step-size');
  const doc = page.locator('article');
  await expect(doc.locator('.katex').first()).toBeVisible();
  const img = doc.locator('img').first();
  await img.scrollIntoViewIfNeeded();
  await expect(img).toBeVisible();
  expect(await img.evaluate((el: HTMLImageElement) => el.naturalWidth)).toBeGreaterThan(0);
  // A section link scrolls to the section and is shareable.
  await page.goto('#/research/certified-stepsize-schedules?s=results');
  await expect(page.locator('#results')).toBeInViewport();
  await noOverflow(page);
  expect(errors).toEqual([]);
});

test('python: install, the call, Result and Step, the CLI and parity', async ({ page }) => {
  await page.goto('#/python');
  await expect(page.getByRole('heading', { level: 1, name: 'Python' })).toBeVisible();
  for (const h of [
    'Install',
    'A first run',
    'Result and Step',
    'The command line',
    'How parity works',
  ])
    await expect(page.getByRole('heading', { level: 2, name: h })).toBeVisible();
  await expect(page.getByRole('region', { name: 'Result fields' })).toContainText('converged');
  await expect(page.getByText(/fixture cases/)).toBeVisible();
  await noOverflow(page);
});
