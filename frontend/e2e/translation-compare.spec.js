// @ts-check
import { test, expect } from '@playwright/test'

test.describe('Translation Compare', () => {
  test('should show compare page with controls', async ({ page }) => {
    await page.goto('/compare')
    await expect(page.locator('.compare-title, h1')).toContainText('Compare')
    await expect(page.locator('select').first()).toBeVisible()
    await expect(page.locator('.compare-btn')).toBeVisible()
  })

  test('should have translation picker with checkboxes', async ({ page }) => {
    await page.goto('/compare')
    await expect(page.locator('.translation-picker')).toBeVisible()
    const chips = page.locator('.chip-label')
    await expect(chips).toHaveCount(6) // BSB, KJV, ESV, NIV, NLT, NASB
  })

  test('should populate book selector from API', async ({ page }) => {
    await page.route('**/api/v1/books', async (route) => {
      await route.fulfill({
        json: {
          books: [
            { id: 'GEN', name: 'Genesis', chapters: 50 },
            { id: 'EXO', name: 'Exodus', chapters: 40 },
          ],
        },
      })
    })
    await page.goto('/compare')
    const select = page.locator('select').first()
    await expect(select.locator('option')).toHaveCount(3) // disabled + 2 books
  })

  test('should compare button be disabled without required fields', async ({ page }) => {
    await page.goto('/compare')
    await expect(page.locator('.compare-btn')).toBeDisabled()
  })

  test('should display comparison table after loading', async ({ page }) => {
    await page.route('**/api/v1/books', async (route) => {
      await route.fulfill({
        json: { books: [{ id: 'GEN', name: 'Genesis', chapters: 50 }] },
      })
    })
    await page.route('**/api/v1/compare/**', async (route) => {
      await route.fulfill({
        json: {
          reference: 'Genesis 1:1',
          translations: [
            { name: 'BSB', translation_id: 'engbsb', text: 'In the beginning God created the heavens and the earth.' },
            { name: 'KJV', translation_id: 'engkjv', text: 'In the beginning God created the heaven and the earth.' },
          ],
          original_words: [
            { text: 'bereshith', gloss: 'in the beginning' },
            { text: 'bara', gloss: 'created' },
          ],
        },
      })
    })
    await page.goto('/compare')
    await page.selectOption('select:first-of-type', 'GEN')
    await page.selectOption('select:nth-of-type(2)', '1')
    await page.click('.compare-btn')
    await expect(page.locator('.compare-table')).toBeVisible()
    await expect(page.locator('.compare-table tbody tr')).toHaveCount(2)
  })

  test('should show original language words', async ({ page }) => {
    await page.route('**/api/v1/books', async (route) => {
      await route.fulfill({ json: { books: [{ id: 'GEN', name: 'Genesis', chapters: 50 }] } })
    })
    await page.route('**/api/v1/compare/**', async (route) => {
      await route.fulfill({
        json: {
          reference: 'Genesis 1:1',
          translations: [{ name: 'BSB', text: 'test' }],
          original_words: [{ text: 'bereshith', gloss: 'in the beginning' }],
        },
      })
    })
    await page.goto('/compare')
    await page.selectOption('select:first-of-type', 'GEN')
    await page.selectOption('select:nth-of-type(2)', '1')
    await page.click('.compare-btn')
    await expect(page.locator('.original-words')).toBeVisible()
  })

  test('should navigate to compare from nav', async ({ page }) => {
    await page.goto('/')
    await page.click('.nav-links >> text=Compare')
    await expect(page).toHaveURL('/compare')
  })
})
