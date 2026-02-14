// @ts-check
import { test, expect } from '@playwright/test'

test.describe('Reading Pane', () => {
  test('should display book and chapter selectors', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('.control-select').first()).toBeVisible()
  })

  test('should show placeholder when no chapter selected', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('.reading-placeholder')).toContainText('Select a book')
  })

  test('should display depth dial component', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('.depth-dial, [class*="depth"]')).toBeVisible()
  })

  test('should have responsive layout on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 })
    await page.goto('/')
    // Navigation should still be visible
    await expect(page.locator('.app-nav')).toBeVisible()
    // Main content should not overflow
    const main = page.locator('.app-main')
    await expect(main).toBeVisible()
    const box = await main.boundingBox()
    if (box) {
      expect(box.width).toBeLessThanOrEqual(375)
    }
  })
})
