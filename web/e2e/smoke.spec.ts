import { expect, test } from '@playwright/test';

test('home: hero scenes, gallery cards, card links', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('#/');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('iterate by iterate');
  await expect(page.getByText(/\d+ methods · \d+ families · \d+ problems/)).toBeVisible();

  // Hero: four scenes, the first one shown; every scene names its run and links to its lab.
  const scenes = page.getByRole('group', { name: 'Scenes' }).getByRole('button');
  await expect(scenes).toHaveText(['Descent', 'Roots', 'Simplex', 'Interpolation']);
  await expect(scenes.first()).toHaveAttribute('aria-pressed', 'true');
  const hero = page.locator('figure').first();
  await expect(hero.getByRole('img', { name: /Himmelblau/ })).toBeVisible();
  await expect(hero.getByText(/one stopping test for all/)).toBeVisible();
  await page.getByRole('button', { name: 'Roots', exact: true }).click();
  await expect(scenes.nth(1)).toHaveAttribute('aria-pressed', 'true');
  await expect(hero.getByRole('img', { name: /cubic/ })).toBeVisible();
  await expect(hero.getByRole('link', { name: /Open this lab/ })).toHaveAttribute(
    'href',
    '#/lab/roots',
  );
  // The looping showcase has a pause control. Pause it here, so that the loop does not race the
  // scene checks below.
  await page.getByRole('button', { name: 'Pause the showcase' }).click();
  await expect(page.getByRole('button', { name: 'Play the showcase' })).toBeVisible();
  // No layout shift: the figure (stage, legend, controls, caption) keeps one height in every
  // scene, although the scenes have different legends and captions.
  const heights: number[] = [];
  for (let i = 0; i < 4; i++) {
    await scenes.nth(i).click();
    await expect(scenes.nth(i)).toHaveAttribute('aria-pressed', 'true');
    heights.push(Math.round((await hero.boundingBox())!.height));
  }
  expect(new Set(heights).size, `hero heights per scene: ${heights.join(', ')}`).toBe(1);
  // Only the active scene's caption is exposed: one "Open this lab" link.
  await expect(hero.getByRole('link', { name: /Open this lab/ })).toHaveAttribute(
    'href',
    '#/lab/interpolation',
  );

  // Gallery: one card per lab, grouped by topic; a card plays on hover and is one link.
  const gallery = page.getByRole('region', { name: 'Labs' });
  const cards = gallery.locator('a[href^="#/lab/"]');
  await expect(cards).toHaveCount(16);
  await expect(gallery.getByRole('heading', { name: 'Equations' })).toBeVisible();
  const roots = gallery.getByRole('link', { name: /Root finding/ });
  await roots.scrollIntoViewIfNeeded();
  const still = await roots.boundingBox();
  await roots.hover();
  await expect(roots.locator('[data-ready]')).toHaveCount(1);
  // Hover changes only the ring and title color: the card does not move or lift.
  await page.waitForTimeout(300);
  expect(await roots.boundingBox()).toEqual(still);
  expect(await roots.evaluate((el) => getComputedStyle(el).transform)).toBe('none');
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth),
  ).toBeLessThanOrEqual(0);
  await roots.click();
  await expect(page).toHaveURL(/#\/lab\/roots/);
  await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
  expect(errors).toEqual([]);
});

test('descent lab: playback, keyboard, click-to-start, URL state', async ({ page, isMobile }) => {
  await page.goto('#/lab/unconstrained');
  const scrub = page.getByRole('slider', { name: 'Iteration', exact: true });
  await expect(scrub).toBeVisible();
  await page.getByRole('button', { name: 'Pause' }).click();
  await page.keyboard.press('Home');
  await expect(scrub).toHaveAttribute('aria-valuenow', '0');
  await page.keyboard.press('ArrowRight');
  await page.keyboard.press('ArrowRight');
  await expect(scrub).toHaveAttribute('aria-valuenow', '2');

  // Clicking the contour sets the start point and writes it to the URL.
  const plot = page.getByRole('img', { name: /Contour plot of Rosenbrock/ });
  const box = (await plot.boundingBox())!;
  await page.mouse.click(box.x + box.width * 0.3, box.y + box.height * 0.3);
  await expect(page).toHaveURL(/x0=/);

  // The theme toggle switches to the other theme on the first click (OS is light here).
  await page.getByRole('button', { name: 'Switch to dark theme' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  // Back to the OS theme: the explicit choice is dropped.
  await page.getByRole('button', { name: 'Switch to light theme' }).click();
  await expect(page.locator('html')).not.toHaveAttribute('data-theme', /./);

  if (!isMobile) {
    await page.getByRole('button', { name: /Add a method/ }).click();
    await page.getByRole('option').first().click();
    await expect(page).toHaveURL(/m=/);
  }
});

test('arrow keys on widgets do not move the playhead', async ({ page, isMobile }) => {
  test.skip(isMobile, 'the rail is in the sheet on phones');
  await page.goto('#/lab/unconstrained');
  const scrub = page.getByRole('slider', { name: 'Iteration', exact: true });
  await page.getByRole('button', { name: 'Pause' }).click();
  await page.keyboard.press('Home');
  await page.getByRole('button', { name: /^Problem:/ }).focus();
  await page.keyboard.press('ArrowRight');
  await expect(scrub).toHaveAttribute('aria-valuenow', '0');
  // The focus picker is a real radiogroup: arrows change the selection, not the playhead.
  const radios = page
    .getByRole('radiogroup', { name: 'Method shown in detail' })
    .getByRole('radio');
  await radios.first().focus();
  await page.keyboard.press('ArrowRight');
  await expect(radios.nth(1)).toHaveAttribute('aria-checked', 'true');
  await expect(scrub).toHaveAttribute('aria-valuenow', '0');
});

test('URL state is validated', async ({ page, isMobile }) => {
  test.skip(isMobile, 'desktop layout shows the counter');
  await page.goto('#/lab/unconstrained?m=nesterov~3,momentum~3,not_a_method~1&x0=1');
  // The unknown id is dropped (with a notice); the colliding slot is reassigned.
  await expect(page.getByText(/This link asked for not_a_method/)).toBeVisible();
  await expect(page.getByText('2 / 4')).toBeVisible();
  // x0=1 has the wrong length, so the problem's default start point is used.
  await expect(page.getByRole('textbox', { name: 'Start y' })).toHaveValue('1');
});

test('mobile controls sheet is a modal dialog', async ({ page, isMobile }) => {
  test.skip(!isMobile, 'phones only');
  await page.goto('#/lab/unconstrained');
  const fab = page.getByRole('button', { name: 'Controls' });
  await fab.click();
  const dialog = page.getByRole('dialog', { name: 'Controls' });
  await expect(dialog).toBeVisible();
  await expect(page.getByRole('button', { name: 'Close controls' })).toBeFocused();
  for (let i = 0; i < 12; i++) {
    await page.keyboard.press('Tab');
    expect(await page.evaluate(() => !!document.activeElement?.closest('[role=dialog]'))).toBe(
      true,
    );
  }
  await page.keyboard.press('Escape');
  await expect(dialog).toBeHidden();
  await expect(page.getByRole('button', { name: 'Controls' })).toBeFocused();
});

test('mobile controls sheet: a fast Tab out of a number field keeps focus moving', async ({
  page,
  isMobile,
}) => {
  test.skip(!isMobile, 'phones only');
  // Regression: a number field selects its text one animation frame after it gets focus. Keys
  // that arrive before that frame (Tab into the field, Tab out, Escape) must not pull focus back
  // into the field, or the closed sheet keeps focus and the Controls button does not get it.
  await page.goto('#/lab/unconstrained');
  await page.getByRole('button', { name: 'Controls' }).click();
  const dialog = page.getByRole('dialog', { name: 'Controls' });
  await expect(page.getByRole('button', { name: 'Close controls' })).toBeFocused();
  const field = dialog.getByRole('textbox').first();
  await field.focus();
  await page.keyboard.press('Shift+Tab');
  await expect(field).not.toBeFocused();
  // Let the field's deferred select() run before the burst of keys.
  await page.evaluate(
    () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))),
  );
  // Send Tab, Tab, Escape in one burst, so that no animation frame runs between them.
  const cdp = await page.context().newCDPSession(page);
  const key = (type: 'rawKeyDown' | 'keyUp', name: string, vk: number) =>
    cdp.send('Input.dispatchKeyEvent', { type, key: name, code: name, windowsVirtualKeyCode: vk });
  await Promise.all([
    key('rawKeyDown', 'Tab', 9),
    key('keyUp', 'Tab', 9),
    key('rawKeyDown', 'Tab', 9),
    key('keyUp', 'Tab', 9),
    key('rawKeyDown', 'Escape', 27),
    key('keyUp', 'Escape', 27),
  ]);
  await expect(dialog).toBeHidden();
  await expect(page.getByRole('button', { name: 'Controls' })).toBeFocused();
});

test('every lab in the gallery is open (no planned cards)', async ({ page }) => {
  await page.goto('#/labs');
  const gallery = page.locator('a[href^="#/lab/"]');
  await expect(gallery).toHaveCount(16);
  await expect(page.locator('a[href^="#/lab/"][data-status="planned"]')).toHaveCount(0);
});

test('search palette: "/" opens it, a method opens its page', async ({ page }) => {
  await page.goto('#/');
  // The shortcut listener is attached once the header has mounted.
  await expect(page.getByRole('button', { name: /Search/ })).toBeVisible();
  await page.waitForLoadState('networkidle');
  await page.keyboard.press('/');
  const box = page.getByRole('combobox', { name: 'Search' });
  await expect(box).toBeFocused();
  await box.fill('bfgs');
  await expect(page.getByRole('option').first()).toContainText('BFGS');
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(/#\/method\/bfgs$/);
});

test('methods page filters by the query in the URL', async ({ page }) => {
  await page.goto('#/methods?q=wolfe');
  await expect(page.getByRole('heading', { level: 1, name: 'Methods' })).toBeVisible();
  await expect(page.getByText(/Strong Wolfe/i).first()).toBeVisible();
});

test('unknown routes show the 404 page', async ({ page }) => {
  await page.goto('#/nowhere');
  await expect(page.getByText('This page is not in the catalog.')).toBeVisible();
});
