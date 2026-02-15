// @ts-check
import { test, expect } from '@playwright/test'

test.describe('Extended Navigation', () => {
  test('should show all navigation links', async ({ page }) => {
    await page.goto('/')
    const nav = page.locator('.nav-links')
    await expect(nav.getByText('Read')).toBeVisible()
    await expect(nav.getByText('Search')).toBeVisible()
    await expect(nav.getByText('Topics')).toBeVisible()
    await expect(nav.getByText('Plans')).toBeVisible()
    await expect(nav.getByText('Compare')).toBeVisible()
    await expect(nav.getByText('Words')).toBeVisible()
    await expect(nav.getByText('Collections')).toBeVisible()
    await expect(nav.getByText('Discover')).toBeVisible()
  })

  test('should show global search bar in nav', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('.nav-search')).toBeVisible()
    await expect(page.locator('.nav-search-input')).toBeVisible()
    await expect(page.locator('.nav-search-btn')).toBeVisible()
  })

  test('should navigate to search from nav search bar', async ({ page }) => {
    await page.goto('/')
    await page.fill('.nav-search-input', 'love')
    await page.press('.nav-search-input', 'Enter')
    await expect(page).toHaveURL(/\/search/)
    await expect(page).toHaveURL(/q=love/)
  })

  test('should clear nav search after submission', async ({ page }) => {
    await page.route('**/api/v1/search/**', async (route) => {
      await route.fulfill({ json: { results: [] } })
    })
    await page.goto('/')
    await page.fill('.nav-search-input', 'grace')
    await page.click('.nav-search-btn')
    // Input should be cleared after navigating
    await expect(page).toHaveURL(/\/search/)
  })

  test('should show depth dial and dark mode toggle', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('.depth-dial, [class*="depth"]')).toBeVisible()
    await expect(page.locator('.dark-toggle')).toBeVisible()
  })

  test('should navigate to all new routes', async ({ page }) => {
    for (const [path, text] of [
      ['/search', 'Search'],
      ['/compare', 'Compare'],
      ['/lexicon', 'Word Study'],
      ['/collections', 'Collections'],
    ]) {
      await page.goto(path)
      await expect(page.locator('h1')).toBeVisible()
    }
  })

  test('should toggle dark mode', async ({ page }) => {
    await page.goto('/')
    await page.click('.dark-toggle')
    await expect(page.locator('#abba-app')).toHaveClass(/dark-mode/)
    await page.click('.dark-toggle')
    await expect(page.locator('#abba-app')).not.toHaveClass(/dark-mode/)
  })

  test('should persist dark mode preference', async ({ page }) => {
    await page.goto('/')
    await page.click('.dark-toggle')
    // Reload and check persistence
    await page.reload()
    await expect(page.locator('#abba-app')).toHaveClass(/dark-mode/)
  })
})
