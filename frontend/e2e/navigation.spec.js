// @ts-check
import { test, expect } from '@playwright/test'

test.describe('Navigation', () => {
  test('should load the home page with ABBA brand', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('.nav-brand')).toHaveText('ABBA')
  })

  test('should show navigation links', async ({ page }) => {
    await page.goto('/')
    const nav = page.locator('.nav-links')
    await expect(nav.getByText('Read')).toBeVisible()
    await expect(nav.getByText('Topics')).toBeVisible()
    await expect(nav.getByText('Plans')).toBeVisible()
  })

  test('should navigate to Topics page', async ({ page }) => {
    await page.goto('/')
    await page.click('.nav-links >> text=Topics')
    await expect(page).toHaveURL('/topics')
    await expect(page.locator('.page-title')).toHaveText('Life Topics')
  })

  test('should navigate to Plans page', async ({ page }) => {
    await page.goto('/')
    await page.click('.nav-links >> text=Plans')
    await expect(page).toHaveURL('/plans')
  })

  test('should navigate to Discover page', async ({ page }) => {
    await page.goto('/discover')
    await expect(page.locator('.discover-title, h1')).toBeVisible()
  })

  test('should navigate to Study page via URL', async ({ page }) => {
    await page.goto('/study/Genesis/1/1')
    await expect(page.locator('.study-view, .study-header, .loading')).toBeVisible()
  })

  test('should show 404 page for unknown routes', async ({ page }) => {
    await page.goto('/nonexistent-route')
    await expect(page.locator('.not-found, .app-main')).toBeVisible()
  })

  test('should highlight active nav link', async ({ page }) => {
    await page.goto('/topics')
    const topicsLink = page.locator('.nav-links a[href="/topics"]')
    await expect(topicsLink).toHaveClass(/router-link-active/)
  })
})
