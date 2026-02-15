// @ts-check
import { test, expect } from '@playwright/test'

test.describe('Lexicon / Word Study', () => {
  test('should show lexicon page with lookup bar', async ({ page }) => {
    await page.goto('/lexicon')
    await expect(page.locator('.lexicon-title, h1')).toContainText('Word Study')
    await expect(page.locator('.lookup-input')).toBeVisible()
    await expect(page.locator('.lookup-btn')).toBeVisible()
  })

  test('should disable lookup button when input is empty', async ({ page }) => {
    await page.goto('/lexicon')
    await expect(page.locator('.lookup-btn')).toBeDisabled()
  })

  test('should display word entry after lookup', async ({ page }) => {
    await page.route('**/api/v1/lexicon/**', async (route) => {
      await route.fulfill({
        json: {
          strongs_number: 'H2617',
          word: '\u05D7\u05E1\u05D3',
          transliteration: 'chesed',
          short_definition: 'lovingkindness',
          full_definition: 'Loyal love, covenant faithfulness, mercy, goodness',
        },
      })
    })
    await page.route('**/api/v1/word-explanations/**', async (route) => {
      await route.fulfill({
        json: { explanation: 'Chesed means loyal love that keeps covenant promises even when the other party fails.' },
      })
    })
    await page.route('**/api/v1/words/**/domains', async (route) => {
      await route.fulfill({ json: { domains: [{ domain_code: '25A', domain_name: 'Love' }] } })
    })
    await page.goto('/lexicon')
    await page.fill('.lookup-input', 'H2617')
    await page.click('.lookup-btn')
    await expect(page.locator('.word-original')).toContainText('\u05D7\u05E1\u05D3')
    await expect(page.locator('.word-translit')).toContainText('chesed')
  })

  test('should load word from URL parameter', async ({ page }) => {
    await page.route('**/api/v1/lexicon/**', async (route) => {
      await route.fulfill({
        json: { strongs_number: 'G26', word: '\u03B1\u03B3\u03B1\u03C0\u03B7', transliteration: 'agape', short_definition: 'love' },
      })
    })
    await page.route('**/api/v1/word-explanations/**', async (route) => {
      await route.fulfill({ json: null })
    })
    await page.route('**/api/v1/words/**/domains', async (route) => {
      await route.fulfill({ json: { domains: [] } })
    })
    await page.goto('/lexicon/G26')
    await expect(page.locator('.word-original')).toBeVisible()
  })

  test('should show explanation section', async ({ page }) => {
    await page.route('**/api/v1/lexicon/**', async (route) => {
      await route.fulfill({ json: { strongs_number: 'H2617', word: 'test', short_definition: 'test' } })
    })
    await page.route('**/api/v1/word-explanations/**', async (route) => {
      await route.fulfill({ json: { explanation: 'This word means loyal love.' } })
    })
    await page.route('**/api/v1/words/**/domains', async (route) => {
      await route.fulfill({ json: { domains: [] } })
    })
    await page.goto('/lexicon/H2617')
    await expect(page.locator('.explanation-text')).toContainText('loyal love')
  })

  test('should show semantic domain badges', async ({ page }) => {
    await page.route('**/api/v1/lexicon/**', async (route) => {
      await route.fulfill({ json: { strongs_number: 'H2617', word: 'test', short_definition: 'test' } })
    })
    await page.route('**/api/v1/word-explanations/**', async (route) => {
      await route.fulfill({ json: null })
    })
    await page.route('**/api/v1/words/**/domains', async (route) => {
      await route.fulfill({
        json: { domains: [{ domain_code: '25A', domain_name: 'Love' }, { domain_code: '34B', domain_name: 'Trust' }] },
      })
    })
    await page.goto('/lexicon/H2617')
    await expect(page.locator('.domain-badge')).toHaveCount(2)
  })

  test('should link to find all occurrences via search', async ({ page }) => {
    await page.route('**/api/v1/lexicon/**', async (route) => {
      await route.fulfill({ json: { strongs_number: 'H2617', word: 'test', short_definition: 'test' } })
    })
    await page.route('**/api/v1/word-explanations/**', async (route) => {
      await route.fulfill({ json: null })
    })
    await page.route('**/api/v1/words/**/domains', async (route) => {
      await route.fulfill({ json: { domains: [] } })
    })
    await page.goto('/lexicon/H2617')
    const link = page.locator('.action-link')
    await expect(link).toContainText('Find all occurrences')
    await expect(link).toHaveAttribute('href', /\/search.*mode=strongs/)
  })

  test('should navigate to lexicon from nav', async ({ page }) => {
    await page.goto('/')
    await page.click('.nav-links >> text=Words')
    await expect(page).toHaveURL('/lexicon')
  })
})
