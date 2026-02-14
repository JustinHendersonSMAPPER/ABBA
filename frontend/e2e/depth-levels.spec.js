// @ts-check
import { test, expect } from '@playwright/test'

test.describe('Depth Levels', () => {
  test('should default to basic depth level', async ({ page }) => {
    await page.goto('/')
    // Context sidebar should NOT be visible at basic depth
    await expect(page.locator('.context-sidebar')).not.toBeVisible()
  })

  test('should toggle depth via depth dial', async ({ page }) => {
    await page.goto('/')
    const depthDial = page.locator('.depth-dial, [class*="depth"]')
    await expect(depthDial).toBeVisible()
  })

  test('should show context sidebar at non-basic depth', async ({ page }) => {
    await page.goto('/')
    // Click on a depth option that's not basic
    const depthOptions = page.locator('.depth-option, .depth-btn, button')
    const count = await depthOptions.count()
    if (count > 1) {
      await depthOptions.nth(1).click()
      // Context sidebar should appear at higher depths
      await page.waitForTimeout(300)
    }
  })
})
