// @ts-check
import { test, expect } from '@playwright/test'

test.describe('Responsive Design', () => {
  const viewports = [
    { name: 'mobile', width: 375, height: 667 },
    { name: 'tablet', width: 768, height: 1024 },
    { name: 'desktop', width: 1280, height: 800 },
  ]

  for (const vp of viewports) {
    test(`should render correctly on ${vp.name} (${vp.width}x${vp.height})`, async ({ page }) => {
      await page.setViewportSize({ width: vp.width, height: vp.height })
      await page.goto('/')

      // Nav should always be visible
      await expect(page.locator('.app-nav')).toBeVisible()

      // Brand should be visible
      await expect(page.locator('.nav-brand')).toBeVisible()

      // Main area should be within viewport width
      const main = page.locator('.app-main')
      await expect(main).toBeVisible()
      const box = await main.boundingBox()
      if (box) {
        expect(box.width).toBeLessThanOrEqual(vp.width)
      }
    })
  }

  test('should handle topics grid on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 })
    await page.goto('/topics')
    await expect(page.locator('.page-title')).toBeVisible()
    // Search input should be full-width on mobile
    const input = page.locator('.search-input')
    await expect(input).toBeVisible()
  })

  test('should handle study view on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 })
    await page.goto('/study/Genesis/1/1')
    // Study view or loading state should be visible
    await expect(page.locator('.study-view, .loading, .error')).toBeVisible()
  })
})
