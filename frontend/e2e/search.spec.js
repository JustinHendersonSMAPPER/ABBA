// @ts-check
import { test, expect } from '@playwright/test'

test.describe('Search', () => {
  test('should show search page with mode selector and input', async ({ page }) => {
    await page.goto('/search')
    await expect(page.locator('.search-title, h1')).toContainText('Search')
    await expect(page.locator('.mode-select')).toBeVisible()
    await expect(page.locator('.search-input')).toBeVisible()
    await expect(page.locator('.search-btn')).toBeVisible()
  })

  test('should have text, semantic, and strongs search modes', async ({ page }) => {
    await page.goto('/search')
    const select = page.locator('.mode-select')
    await expect(select.locator('option')).toHaveCount(3)
    await expect(select.locator('option[value="text"]')).toHaveText('Text Search')
    await expect(select.locator('option[value="semantic"]')).toHaveText('Semantic Search')
    await expect(select.locator('option[value="strongs"]')).toContainText("Strong's")
  })

  test('should update URL with search query and mode', async ({ page }) => {
    await page.goto('/search')
    await page.fill('.search-input', 'love')
    await page.click('.search-btn')
    await expect(page).toHaveURL(/q=love/)
    await expect(page).toHaveURL(/mode=text/)
  })

  test('should navigate to search from global nav search bar', async ({ page }) => {
    await page.goto('/')
    await page.fill('.nav-search-input', 'grace')
    await page.click('.nav-search-btn')
    await expect(page).toHaveURL(/\/search/)
    await expect(page).toHaveURL(/q=grace/)
  })

  test('should show loading state during search', async ({ page }) => {
    await page.route('**/api/v1/search/text*', async (route) => {
      await new Promise((r) => setTimeout(r, 500))
      await route.fulfill({ json: { results: [] } })
    })
    await page.goto('/search')
    await page.fill('.search-input', 'test')
    await page.click('.search-btn')
    await expect(page.locator('.status-msg')).toContainText('Searching')
  })

  test('should display search results as cards', async ({ page }) => {
    await page.route('**/api/v1/search/text*', async (route) => {
      await route.fulfill({
        json: {
          results: [
            { book_id: 'GEN', book_name: 'Genesis', chapter: 1, verse: 1, text: 'In the beginning God created...' },
            { book_id: 'JHN', book_name: 'John', chapter: 3, verse: 16, text: 'For God so loved the world...' },
          ],
        },
      })
    })
    await page.goto('/search?q=God&mode=text')
    await expect(page.locator('.result-card')).toHaveCount(2)
    await expect(page.locator('.results-count')).toContainText('2 results')
  })

  test('should show empty state when no results', async ({ page }) => {
    await page.route('**/api/v1/search/text*', async (route) => {
      await route.fulfill({ json: { results: [] } })
    })
    await page.goto('/search?q=xyznonexistent&mode=text')
    await expect(page.locator('.status-msg')).toContainText('No results found')
  })

  test('should handle API errors gracefully', async ({ page }) => {
    await page.route('**/api/v1/search/text*', async (route) => {
      await route.fulfill({ status: 500, body: 'Internal Server Error' })
    })
    await page.goto('/search?q=error&mode=text')
    await expect(page.locator('.error')).toBeVisible()
  })

  test('should link search results to study view', async ({ page }) => {
    await page.route('**/api/v1/search/text*', async (route) => {
      await route.fulfill({
        json: {
          results: [{ book_id: 'GEN', book_name: 'Genesis', chapter: 1, verse: 1, text: 'In the beginning...' }],
        },
      })
    })
    await page.goto('/search?q=beginning&mode=text')
    const link = page.locator('.result-ref').first()
    await expect(link).toHaveAttribute('href', /\/study\/GEN\/1\/1/)
  })
})
