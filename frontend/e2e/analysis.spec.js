// @ts-check
import { test, expect } from '@playwright/test'

test.describe('Analysis View', () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => localStorage.setItem('abba-onboarded', 'true'))
  })

  test('should show page title "Word Analysis" and subtitle', async ({ page }) => {
    await page.goto('/analysis')
    await expect(page.locator('.analysis-title')).toContainText('Word Analysis')
    await expect(page.locator('.analysis-subtitle')).toBeVisible()
  })

  test('should display two tabs: Word Frequency and Morphology Patterns', async ({ page }) => {
    await page.goto('/analysis')
    await expect(page.locator('.tab-btn')).toHaveCount(2)
    await expect(page.locator('.tab-btn').first()).toContainText('Word Frequency')
    await expect(page.locator('.tab-btn').last()).toContainText('Morphology Patterns')
  })

  test('should show frequency controls (pattern input, min freq input, Analyze button)', async ({ page }) => {
    await page.goto('/analysis')
    await expect(page.locator('.control-input').first()).toBeVisible()
    await expect(page.locator('.control-input.narrow')).toBeVisible()
    await expect(page.locator('.search-btn')).toBeVisible()
  })

  test('should display frequency results table', async ({ page }) => {
    await page.route('**/api/v1/analysis/frequency*', async (route) => {
      await route.fulfill({
        json: {
          results: [
            { strongs_number: 'H2617', word: 'chesed', gloss: 'lovingkindness', frequency: 248 },
            { strongs_number: 'H2603', word: 'chanan', gloss: 'to be gracious', frequency: 77 },
          ],
        },
      })
    })
    await page.goto('/analysis')
    await page.fill('.control-input >> nth=0', 'H26*')
    await page.click('.search-btn')
    await expect(page.locator('.results-table')).toBeVisible()
    await expect(page.locator('.results-table tbody tr')).toHaveCount(2)
    await expect(page.locator('.results-table')).toContainText('chesed')
    await expect(page.locator('.results-table')).toContainText('248')
  })

  test('should link frequency results to lexicon', async ({ page }) => {
    await page.route('**/api/v1/analysis/frequency*', async (route) => {
      await route.fulfill({
        json: {
          results: [
            { strongs_number: 'H2617', word: 'chesed', gloss: 'lovingkindness', frequency: 248 },
          ],
        },
      })
    })
    await page.goto('/analysis')
    await page.fill('.control-input >> nth=0', 'H26*')
    await page.click('.search-btn')
    const link = page.locator('.strongs-link').first()
    await expect(link).toBeVisible()
    await expect(link).toHaveAttribute('href', /\/lexicon\/H2617/)
  })

  test('should switch to morphology tab and show controls', async ({ page }) => {
    await page.goto('/analysis')
    await page.click('.tab-btn:has-text("Morphology")')
    await expect(page.locator('.control-select')).toBeVisible()
    await expect(page.locator('.search-btn')).toBeVisible()
  })

  test('should display morphology results as cards', async ({ page }) => {
    await page.route('**/api/v1/analysis/morphology*', async (route) => {
      await route.fulfill({
        json: {
          results: [
            {
              pattern: 'Qal Perfect 3ms',
              code: 'HVqp3ms',
              count: 5420,
              description: 'Simple active, completed action, 3rd person masculine singular',
            },
          ],
        },
      })
    })
    await page.goto('/analysis')
    await page.click('.tab-btn:has-text("Morphology")')
    const langSelect = page.locator('.control-select')
    await langSelect.selectOption('hebrew')
    await page.click('.search-btn')
    await expect(page.locator('.morph-card')).toHaveCount(1)
    await expect(page.locator('.morph-card')).toContainText('Qal Perfect 3ms')
    await expect(page.locator('.morph-card')).toContainText('5420')
  })
})
