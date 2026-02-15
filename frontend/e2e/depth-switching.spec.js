// @ts-check
import { test, expect } from '@playwright/test'

test.describe('Depth Switching', () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => localStorage.setItem('abba-onboarded', 'true'))
    await page.route('**/api/v1/verses/**', async (route) => {
      const url = route.request().url()
      if (url.includes('cross-references')) {
        await route.fulfill({
          json: {
            references: [
              { book: 'JHN', chapter: 1, verse: 1, label: 'John 1:1', note: 'In the beginning was the Word' },
            ],
          },
        })
      } else if (url.includes('context')) {
        await route.fulfill({
          json: {
            cultural: [{ text: 'Ancient Near Eastern creation account context' }],
            historical: 'Written during or after the Babylonian exile',
          },
        })
      } else {
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
              },
            ],
          },
        })
      }
    })
    await page.route('**/api/v1/syntax/**', async (route) => {
      await route.fulfill({
        json: {
          root_nodes: [
            { node_id: 'n1', node_type: 'sentence', text_content: 'In the beginning' },
          ],
        },
      })
    })
    await page.route('**/api/v1/variants/**', async (route) => {
      await route.fulfill({
        json: {
          variants: [
            {
              variant_id: 'v1',
              variant_type: 'spelling',
              significance: 'minor',
              base_text: 'bereshith',
              variant_text: 'bereshit',
            },
          ],
        },
      })
    })
    await page.route('**/api/v1/discourse/**', async (route) => {
      await route.fulfill({
        json: {
          units: [
            {
              discourse_id: 'd1',
              discourse_type: 'narrative',
              start_chapter: 1,
              start_verse: 1,
              end_chapter: 2,
              end_verse: 3,
              prominence: 2,
            },
          ],
        },
      })
    })
    await page.route('**/api/v1/notes/**', async (route) => {
      await route.fulfill({ json: { notes: [] } })
    })
  })

  test('should default to basic depth with no sidebar', async ({ page }) => {
    await page.goto('/study/GEN/1/1')
    await expect(page.locator('.context-sidebar')).not.toBeVisible()
  })

  test('should show context sidebar when switching to standard depth', async ({ page }) => {
    await page.goto('/study/GEN/1/1')
    const understandBtn = page.locator('.depth-option:has-text("Understand")')
    await understandBtn.click()
    await page.waitForTimeout(300)
    await expect(page.locator('.context-sidebar')).toBeVisible()
  })

  test('should show cross-references at deep depth', async ({ page }) => {
    await page.goto('/study/GEN/1/1')
    const depthOptions = page.locator('.depth-option')
    // Study = deep (index 2)
    await depthOptions.nth(2).click()
    await page.waitForTimeout(300)
    await expect(page.locator('.cross-ref-item')).toBeVisible()
    await expect(page.locator('.cross-ref-item').first()).toContainText('John 1:1')
  })

  test('should show syntax tree at scholarly depth', async ({ page }) => {
    await page.goto('/study/GEN/1/1')
    const depthOptions = page.locator('.depth-option')
    // Analyze = scholarly (index 3)
    await depthOptions.nth(3).click()
    await page.waitForTimeout(300)
    await expect(page.locator('.syntax-tree-view')).toBeVisible()
  })

  test('should show manuscript variants at scholarly depth', async ({ page }) => {
    await page.goto('/study/GEN/1/1')
    const depthOptions = page.locator('.depth-option')
    await depthOptions.nth(3).click()
    await page.waitForTimeout(300)
    await expect(page.locator('.manuscript-variants')).toBeVisible()
    await expect(page.locator('.variant-card').first()).toContainText('bereshit')
  })

  test('should show discourse units at scholarly depth', async ({ page }) => {
    await page.goto('/study/GEN/1/1')
    const depthOptions = page.locator('.depth-option')
    await depthOptions.nth(3).click()
    await page.waitForTimeout(300)
    await expect(page.locator('.discourse-view')).toBeVisible()
    await expect(page.locator('.discourse-unit').first()).toContainText('narrative')
  })
})
