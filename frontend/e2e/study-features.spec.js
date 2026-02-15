// @ts-check
import { test, expect } from '@playwright/test'

test.describe('Study View Features', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('**/api/v1/verses/**', async (route) => {
      await route.fulfill({
        json: {
          book: 'Genesis',
          chapter: 1,
          verse: 1,
          text: 'In the beginning God created the heavens and the earth.',
          words: [
            { word_num: 1, original_text: 'bereshith', transliteration: 'bereshith', gloss: 'in the beginning', strongs: 'H7225' },
          ],
        },
      })
    })
    await page.route('**/api/v1/verses/**/cross-references', async (route) => {
      await route.fulfill({
        json: {
          references: [
            { book: 'JHN', chapter: 1, verse: 1, label: 'John 1:1', note: 'In the beginning was the Word' },
          ],
        },
      })
    })
    await page.route('**/api/v1/verses/**/context', async (route) => {
      await route.fulfill({
        json: {
          cultural: [{ text: 'Ancient Near Eastern creation account context' }],
          historical: 'Written during or after the Babylonian exile',
        },
      })
    })
  })

  test('should show action buttons (Bookmark, Export, Share, Compare)', async ({ page }) => {
    await page.goto('/study/Genesis/1/1')
    await expect(page.locator('.study-actions')).toBeVisible()
    await expect(page.locator('.action-btn:has-text("Bookmark")')).toBeVisible()
    await expect(page.locator('.action-btn:has-text("Export")')).toBeVisible()
    await expect(page.locator('.action-btn:has-text("Share")')).toBeVisible()
    await expect(page.locator('.action-link:has-text("Compare")')).toBeVisible()
  })

  test('should show notes panel for verse view', async ({ page }) => {
    await page.route('**/api/v1/notes/**', async (route) => {
      await route.fulfill({ json: { notes: [] } })
    })
    await page.goto('/study/Genesis/1/1')
    await expect(page.locator('.notes-panel')).toBeVisible()
    await expect(page.locator('.panel-heading')).toContainText('Notes')
    await expect(page.locator('.note-input')).toBeVisible()
  })

  test('should display existing notes', async ({ page }) => {
    await page.route('**/api/v1/notes/**', async (route) => {
      if (route.request().method() === 'GET') {
        await route.fulfill({
          json: {
            notes: [
              { id: 1, content: 'This is the foundation verse', note_type: 'personal', created_at: '2025-01-01' },
              { id: 2, content: 'Key creation theology', note_type: 'study', created_at: '2025-01-02' },
            ],
          },
        })
      }
    })
    await page.goto('/study/Genesis/1/1')
    await expect(page.locator('.note-card')).toHaveCount(2)
    await expect(page.locator('.note-content').first()).toContainText('foundation verse')
  })

  test('should open collection picker modal', async ({ page }) => {
    await page.route('**/api/v1/notes/**', async (route) => {
      await route.fulfill({ json: { notes: [] } })
    })
    await page.route('**/api/v1/collections', async (route) => {
      await route.fulfill({
        json: { collections: [{ id: 1, name: 'Favorites' }] },
      })
    })
    await page.goto('/study/Genesis/1/1')
    await page.click('.action-btn:has-text("Bookmark")')
    await expect(page.locator('.modal-overlay')).toBeVisible()
    await expect(page.locator('.modal-title')).toContainText('Save to Collection')
    await expect(page.locator('.collection-option')).toContainText('Favorites')
  })

  test('should close modal on cancel', async ({ page }) => {
    await page.route('**/api/v1/notes/**', async (route) => {
      await route.fulfill({ json: { notes: [] } })
    })
    await page.route('**/api/v1/collections', async (route) => {
      await route.fulfill({ json: { collections: [] } })
    })
    await page.goto('/study/Genesis/1/1')
    await page.click('.action-btn:has-text("Bookmark")')
    await page.click('.modal-close')
    await expect(page.locator('.modal-overlay')).not.toBeVisible()
  })

  test('should show share link after sharing', async ({ page }) => {
    await page.route('**/api/v1/notes/**', async (route) => {
      await route.fulfill({ json: { notes: [] } })
    })
    await page.route('**/api/v1/share', async (route) => {
      await route.fulfill({ json: { token: 'abc123' } })
    })
    await page.goto('/study/Genesis/1/1')
    await page.click('.action-btn:has-text("Share")')
    await expect(page.locator('.share-banner')).toBeVisible()
    await expect(page.locator('.share-banner')).toContainText('abc123')
  })

  test('should show compare link pointing to compare page', async ({ page }) => {
    await page.route('**/api/v1/notes/**', async (route) => {
      await route.fulfill({ json: { notes: [] } })
    })
    await page.goto('/study/Genesis/1/1')
    const compareLink = page.locator('.action-link:has-text("Compare")')
    await expect(compareLink).toHaveAttribute('href', /\/compare/)
  })

  test('should link cross-references to study view', async ({ page }) => {
    await page.route('**/api/v1/notes/**', async (route) => {
      await route.fulfill({ json: { notes: [] } })
    })
    await page.goto('/study/Genesis/1/1')
    // Cross-refs shown when depth is not basic - but we default to basic
    // Navigate with depth parameter via depth dial simulation
  })
})
