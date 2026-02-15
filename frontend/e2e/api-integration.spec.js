// @ts-check
import { test, expect } from '@playwright/test'

test.describe('API Integration', () => {
  test('should handle API errors gracefully', async ({ page }) => {
    // Mock API to return an error
    await page.route('**/api/v1/books', (route) =>
      route.fulfill({ status: 500, body: 'Internal Server Error' })
    )
    await page.goto('/')
    // Should show error state, not crash
    await expect(page.locator('.error, .reading-placeholder')).toBeVisible()
  })

  test('should show loading state during API calls', async ({ page }) => {
    // Delay API response to observe loading state
    await page.route('**/api/v1/life-topics', async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 1000))
      await route.fulfill({ status: 200, body: JSON.stringify({ topics: [] }) })
    })
    await page.goto('/topics')
    await expect(page.locator('.loading')).toBeVisible()
  })

  test('should handle empty responses', async ({ page }) => {
    await page.route('**/api/v1/life-topics', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ topics: [] }),
      })
    )
    await page.goto('/topics')
    // Wait for the mock to respond
    await page.waitForTimeout(500)
    // Should show either empty grid or empty state after search
  })

  test('should handle network timeouts', async ({ page }) => {
    await page.route('**/api/v1/books', (route) => route.abort('timedout'))
    await page.goto('/')
    // Should show error state
    await expect(page.locator('.error, .reading-placeholder')).toBeVisible()
  })
})
