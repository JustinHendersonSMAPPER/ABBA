// @ts-check
import { test, expect } from '@playwright/test'

test.describe('Semantic Domain Browser', () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => localStorage.setItem('abba-onboarded', 'true'))
  })

  test('should show page title "Semantic Domains" and subtitle', async ({ page }) => {
    await page.route('**/api/v1/semantic-domains*', async (route) => {
      await route.fulfill({ json: { domains: [] } })
    })
    await page.goto('/domains')
    await expect(page.locator('.browser-title')).toContainText('Semantic Domains')
    await expect(page.locator('.browser-subtitle')).toBeVisible()
  })

  test('should display top-level domains as cards', async ({ page }) => {
    await page.route('**/api/v1/semantic-domains', async (route) => {
      await route.fulfill({
        json: {
          domains: [
            { domain_code: '1', domain_name: 'Geographical Objects', description: 'Places and terrain', child_count: 5 },
            { domain_code: '2', domain_name: 'Natural Substances', child_count: 3 },
          ],
        },
      })
    })
    await page.goto('/domains')
    await expect(page.locator('.domain-card')).toHaveCount(2)
  })

  test('should show domain code and name on cards', async ({ page }) => {
    await page.route('**/api/v1/semantic-domains', async (route) => {
      await route.fulfill({
        json: {
          domains: [
            { domain_code: '1', domain_name: 'Geographical Objects', description: 'Places and terrain', child_count: 5 },
            { domain_code: '2', domain_name: 'Natural Substances', child_count: 3 },
          ],
        },
      })
    })
    await page.goto('/domains')
    await expect(page.locator('.domain-code').first()).toContainText('1')
    await expect(page.locator('.domain-name').first()).toContainText('Geographical Objects')
  })

  test('should show sub-domains when clicking a domain', async ({ page }) => {
    await page.route('**/api/v1/semantic-domains*', async (route) => {
      const url = route.request().url()
      if (url.includes('parent=1')) {
        await route.fulfill({
          json: {
            domains: [{ domain_code: '1.1', domain_name: 'Bodies of Water' }],
          },
        })
      } else {
        await route.fulfill({
          json: {
            domains: [
              { domain_code: '1', domain_name: 'Geographical Objects', description: 'Places and terrain', child_count: 5 },
            ],
          },
        })
      }
    })
    await page.goto('/domains')
    await page.click('.domain-card')
    await expect(page.locator('.domain-name')).toContainText('Bodies of Water')
  })

  test('should show breadcrumb navigation after drilling down', async ({ page }) => {
    await page.route('**/api/v1/semantic-domains*', async (route) => {
      const url = route.request().url()
      if (url.includes('parent=1')) {
        await route.fulfill({
          json: {
            domains: [{ domain_code: '1.1', domain_name: 'Bodies of Water' }],
          },
        })
      } else {
        await route.fulfill({
          json: {
            domains: [
              { domain_code: '1', domain_name: 'Geographical Objects', description: 'Places and terrain', child_count: 5 },
            ],
          },
        })
      }
    })
    await page.goto('/domains')
    await page.click('.domain-card')
    await expect(page.locator('.breadcrumbs')).toBeVisible()
    await expect(page.locator('.breadcrumbs')).toContainText('Geographical Objects')
  })

  test('should show words when clicking a leaf domain', async ({ page }) => {
    await page.route('**/api/v1/semantic-domains/*/words', async (route) => {
      await route.fulfill({
        json: {
          words: [{ strongs_number: 'H3220', word: 'yam', transliteration: 'yam', gloss: 'sea' }],
        },
      })
    })
    await page.route('**/api/v1/semantic-domains*', async (route) => {
      const url = route.request().url()
      if (url.includes('parent=1.1')) {
        await route.fulfill({ json: { domains: [] } })
      } else if (url.includes('parent=1')) {
        await route.fulfill({
          json: {
            domains: [{ domain_code: '1.1', domain_name: 'Bodies of Water', child_count: 0 }],
          },
        })
      } else {
        await route.fulfill({
          json: {
            domains: [
              { domain_code: '1', domain_name: 'Geographical Objects', child_count: 5 },
            ],
          },
        })
      }
    })
    await page.goto('/domains')
    await page.click('.domain-card')
    await expect(page.locator('.domain-name')).toContainText('Bodies of Water')
    await page.click('.domain-card')
    await expect(page.locator('.word-card')).toHaveCount(1)
    await expect(page.locator('.word-card')).toContainText('yam')
    await expect(page.locator('.word-card')).toContainText('sea')
  })

  test('should link word cards to lexicon page', async ({ page }) => {
    await page.route('**/api/v1/semantic-domains/*/words', async (route) => {
      await route.fulfill({
        json: {
          words: [{ strongs_number: 'H3220', word: 'yam', transliteration: 'yam', gloss: 'sea' }],
        },
      })
    })
    await page.route('**/api/v1/semantic-domains*', async (route) => {
      const url = route.request().url()
      if (url.includes('parent=1.1')) {
        await route.fulfill({ json: { domains: [] } })
      } else if (url.includes('parent=1')) {
        await route.fulfill({
          json: {
            domains: [{ domain_code: '1.1', domain_name: 'Bodies of Water', child_count: 0 }],
          },
        })
      } else {
        await route.fulfill({
          json: {
            domains: [
              { domain_code: '1', domain_name: 'Geographical Objects', child_count: 5 },
            ],
          },
        })
      }
    })
    await page.goto('/domains')
    await page.click('.domain-card')
    await expect(page.locator('.domain-name')).toContainText('Bodies of Water')
    await page.click('.domain-card')
    const wordCard = page.locator('.word-card').first()
    await expect(wordCard).toHaveAttribute('href', /\/lexicon\/H3220/)
  })
})
