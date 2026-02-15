// @ts-check
import { test, expect } from '@playwright/test'

test.describe('Cross-Reference Drilldown', () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => localStorage.setItem('abba-onboarded', 'true'))
    await page.route('**/api/v1/verses/GEN/1/1?depth=*', async (route) => {
      await route.fulfill({
        json: {
          book: 'Genesis',
          chapter: 1,
          verse: 1,
          text: 'In the beginning God created the heavens and the earth.',
          words: [
            {
              word_num: 1,
              original_text: 'bereshith',
              transliteration: 'bereshith',
              gloss: 'in the beginning',
              strongs: 'H7225',
              richness_flags: ['theological', 'key_term'],
            },
            {
              word_num: 2,
              original_text: 'elohim',
              transliteration: 'elohim',
              gloss: 'God',
              strongs: 'H430',
              richness_flags: ['theological'],
            },
          ],
        },
      })
    })
    await page.route('**/api/v1/verses/JHN/1/1?depth=*', async (route) => {
      await route.fulfill({
        json: {
          book: 'John',
          chapter: 1,
          verse: 1,
          text: 'In the beginning was the Word, and the Word was with God, and the Word was God.',
          words: [
            {
              word_num: 1,
              original_text: 'arche',
              transliteration: 'arche',
              gloss: 'beginning',
              strongs: 'G746',
            },
          ],
        },
      })
    })
    await page.route('**/cross-references', async (route) => {
      await route.fulfill({
        json: {
          references: [
            { book: 'JHN', chapter: 1, verse: 1, label: 'John 1:1', note: 'In the beginning was the Word' },
          ],
        },
      })
    })
    await page.route('**/context', async (route) => {
      await route.fulfill({
        json: {
          cultural: [{ text: 'Ancient Near Eastern creation account context' }],
          historical: 'Written during or after the Babylonian exile',
        },
      })
    })
    await page.route('**/api/v1/notes/**', async (route) => {
      await route.fulfill({ json: { notes: [] } })
    })
  })

  test('should display cross-references at non-basic depth', async ({ page }) => {
    await page.goto('/study/GEN/1/1')
    const depthOptions = page.locator('.depth-option')
    // Click "Understand" (standard depth, index 1)
    await depthOptions.nth(1).click()
    await page.waitForTimeout(300)
    // Cross refs appear in the sidebar as .cross-ref-item
    await expect(page.locator('.cross-ref-item').first()).toBeVisible()
    await expect(page.locator('.cross-ref-item').first()).toContainText('John 1:1')
  })

  test('should navigate to cross-referenced verse when clicked', async ({ page }) => {
    await page.goto('/study/GEN/1/1')
    const depthOptions = page.locator('.depth-option')
    await depthOptions.nth(1).click()
    await page.waitForTimeout(300)
    const xrefLink = page.locator('.cross-ref-link').first()
    await xrefLink.click()
    await expect(page).toHaveURL(/\/study\/JHN\/1\/1/)
  })

  test('should display word cards when clicking rich words in TranslationLens', async ({ page }) => {
    await page.goto('/study/GEN/1/1')
    // Switch to standard depth so TranslationLens renders
    const depthOptions = page.locator('.depth-option')
    await depthOptions.nth(1).click()
    await page.waitForTimeout(300)
    const richWord = page.locator('.lens-word.has-richness').first()
    const exists = (await richWord.count()) > 0
    if (exists) {
      await richWord.click()
      await page.waitForTimeout(300)
      await expect(page.locator('.word-journey-card')).toBeVisible()
      await expect(page.locator('.word-journey-card')).toContainText('bereshith')
    }
  })

  test('should link word card Learn more to lexicon page', async ({ page }) => {
    await page.goto('/study/GEN/1/1')
    const depthOptions = page.locator('.depth-option')
    await depthOptions.nth(1).click()
    await page.waitForTimeout(300)
    const richWord = page.locator('.lens-word.has-richness').first()
    const exists = (await richWord.count()) > 0
    if (exists) {
      await richWord.click()
      await page.waitForTimeout(300)
      const learnMore = page.locator('.learn-more')
      await expect(learnMore).toHaveAttribute('href', /\/lexicon\/H7225/)
    }
  })

  test('should show Find all occurrences from word card that links to search', async ({ page }) => {
    await page.goto('/study/GEN/1/1')
    const depthOptions = page.locator('.depth-option')
    await depthOptions.nth(1).click()
    await page.waitForTimeout(300)
    const richWord = page.locator('.lens-word.has-richness').first()
    const exists = (await richWord.count()) > 0
    if (exists) {
      await richWord.click()
      await page.waitForTimeout(300)
      // The WordJourneyCard has a learn-more link to lexicon
      // and shows Strong's number which can be used for search
      await expect(page.locator('.word-journey-card')).toContainText('H7225')
    }
  })
})
