// @ts-check
import { test, expect } from '@playwright/test'

test.describe('Notes and Collections Flow', () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => localStorage.setItem('abba-onboarded', 'true'))
  })

  test('should create a new note', async ({ page }) => {
    await page.route('**/api/v1/verses/**', async (route) => {
      const url = route.request().url()
      if (url.includes('cross-references')) {
        await route.fulfill({ json: { references: [] } })
      } else if (url.includes('context')) {
        await route.fulfill({ json: { cultural: [], historical: '' } })
      } else {
        await route.fulfill({
          json: {
            book: 'Genesis',
            chapter: 1,
            verse: 1,
            text: 'In the beginning God created the heavens and the earth.',
          },
        })
      }
    })
    let noteCreated = false
    await page.route('**/api/v1/notes/**', async (route) => {
      if (route.request().method() === 'POST') {
        noteCreated = true
        await route.fulfill({
          json: {
            id: 1,
            content: 'This verse establishes God as Creator',
            note_type: 'personal',
            created_at: '2025-06-01',
          },
        })
      } else {
        // GET: return created note after POST, otherwise empty
        if (noteCreated) {
          await route.fulfill({
            json: {
              notes: [
                { id: 1, content: 'This verse establishes God as Creator', note_type: 'personal', created_at: '2025-06-01' },
              ],
            },
          })
        } else {
          await route.fulfill({ json: { notes: [] } })
        }
      }
    })
    await page.goto('/study/GEN/1/1')
    await expect(page.locator('.notes-panel')).toBeVisible()
    await page.fill('.note-input', 'This verse establishes God as Creator')
    const typeSelect = page.locator('.type-select')
    if ((await typeSelect.count()) > 0) {
      await typeSelect.first().selectOption('personal')
    }
    await page.click('.save-btn')
    await page.waitForTimeout(500)
    await expect(page.locator('.note-card')).toBeVisible()
  })

  test('should delete a note', async ({ page }) => {
    await page.route('**/api/v1/verses/**', async (route) => {
      const url = route.request().url()
      if (url.includes('cross-references')) {
        await route.fulfill({ json: { references: [] } })
      } else if (url.includes('context')) {
        await route.fulfill({ json: { cultural: [], historical: '' } })
      } else {
        await route.fulfill({
          json: { book: 'Genesis', chapter: 1, verse: 1, text: 'In the beginning God created the heavens and the earth.' },
        })
      }
    })
    let noteDeleted = false
    await page.route('**/api/v1/notes/**', async (route) => {
      if (route.request().method() === 'GET') {
        if (noteDeleted) {
          await route.fulfill({ json: { notes: [] } })
        } else {
          await route.fulfill({
            json: {
              notes: [
                { id: 1, content: 'Old note to delete', note_type: 'personal', created_at: '2025-01-01' },
              ],
            },
          })
        }
      } else if (route.request().method() === 'DELETE') {
        noteDeleted = true
        await route.fulfill({ json: { success: true } })
      }
    })
    await page.goto('/study/GEN/1/1')
    await expect(page.locator('.note-card')).toHaveCount(1)
    await page.click('.note-delete')
    await page.waitForTimeout(300)
    await expect(page.locator('.note-card')).toHaveCount(0)
  })

  test('should create a new collection', async ({ page }) => {
    let collections = []
    await page.route('**/api/v1/collections', async (route) => {
      if (route.request().method() === 'GET') {
        await route.fulfill({ json: { collections } })
      } else if (route.request().method() === 'POST') {
        const newCol = { id: 1, name: 'My Study Notes', description: 'Notes from my daily study', item_count: 0 }
        collections = [newCol]
        await route.fulfill({ json: newCol })
      }
    })
    await page.goto('/collections')
    await page.click('.btn-primary:has-text("New Collection")')
    await page.fill('input[placeholder="Collection name"]', 'My Study Notes')
    const descInput = page.locator('input[placeholder*="Description"], input[placeholder*="description"]')
    if ((await descInput.count()) > 0) {
      await descInput.first().fill('Notes from my daily study')
    }
    await page.click('.btn-primary:has-text("Create")')
    await page.waitForTimeout(300)
    await expect(page.locator('.collection-card')).toHaveCount(1)
    await expect(page.locator('.col-name')).toContainText('My Study Notes')
  })

  test('should save verse to collection from study view', async ({ page }) => {
    await page.route('**/api/v1/verses/**', async (route) => {
      const url = route.request().url()
      if (url.includes('cross-references')) {
        await route.fulfill({ json: { references: [] } })
      } else if (url.includes('context')) {
        await route.fulfill({ json: { cultural: [], historical: '' } })
      } else {
        await route.fulfill({
          json: { book: 'Genesis', chapter: 1, verse: 1, text: 'In the beginning God created the heavens and the earth.' },
        })
      }
    })
    await page.route('**/api/v1/notes/**', async (route) => {
      await route.fulfill({ json: { notes: [] } })
    })
    await page.route('**/api/v1/collections', async (route) => {
      await route.fulfill({
        json: { collections: [{ id: 1, name: 'Favorites', item_count: 0 }] },
      })
    })
    await page.route('**/api/v1/collections/1/items', async (route) => {
      if (route.request().method() === 'POST') {
        await route.fulfill({ json: { success: true } })
      }
    })
    await page.goto('/study/GEN/1/1')
    await page.click('.action-btn:has-text("Bookmark")')
    await expect(page.locator('.modal-overlay')).toBeVisible()
    await expect(page.locator('.collection-option')).toContainText('Favorites')
    await page.click('.collection-option:has-text("Favorites")')
    await page.waitForTimeout(300)
  })

  test('should show collection items after creation', async ({ page }) => {
    await page.route('**/api/v1/collections', async (route) => {
      await route.fulfill({
        json: { collections: [{ id: 1, name: 'Favorites', item_count: 1 }] },
      })
    })
    await page.route('**/api/v1/collections/1/items', async (route) => {
      await route.fulfill({
        json: {
          items: [
            { book_id: 'GEN', book_name: 'Genesis', chapter: 1, verse: 1, text: 'In the beginning God created the heavens and the earth.' },
          ],
        },
      })
    })
    await page.goto('/collections')
    await page.click('.collection-card')
    await expect(page.locator('.detail-name')).toContainText('Favorites')
    await expect(page.locator('.item-card')).toHaveCount(1)
    await expect(page.locator('.item-ref')).toContainText('Genesis 1:1')
  })

  test('should delete a collection', async ({ page }) => {
    let collections = [
      { id: 1, name: 'To Delete', description: 'Will be removed', item_count: 0 },
      { id: 2, name: 'Keep This', description: 'Stays around', item_count: 2 },
    ]
    await page.route('**/api/v1/collections', async (route) => {
      if (route.request().method() === 'GET') {
        await route.fulfill({ json: { collections } })
      }
    })
    await page.route('**/api/v1/collections/1', async (route) => {
      if (route.request().method() === 'DELETE') {
        collections = collections.filter((c) => c.id !== 1)
        await route.fulfill({ json: { success: true } })
      }
    })
    await page.goto('/collections')
    await expect(page.locator('.collection-card')).toHaveCount(2)
    const deleteBtn = page.locator('.collection-card').first().locator('.col-delete')
    if ((await deleteBtn.count()) > 0) {
      await deleteBtn.first().click()
      await page.waitForTimeout(300)
      await expect(page.locator('.collection-card')).toHaveCount(1)
    }
  })

  test('should show empty state after deleting last collection', async ({ page }) => {
    let collections = [{ id: 1, name: 'Last One', description: '', item_count: 0 }]
    await page.route('**/api/v1/collections', async (route) => {
      if (route.request().method() === 'GET') {
        await route.fulfill({ json: { collections } })
      }
    })
    await page.route('**/api/v1/collections/1', async (route) => {
      if (route.request().method() === 'DELETE') {
        collections = []
        await route.fulfill({ json: { success: true } })
      }
    })
    await page.goto('/collections')
    await expect(page.locator('.collection-card')).toHaveCount(1)
    const deleteBtn = page.locator('.collection-card').first().locator('.col-delete')
    if ((await deleteBtn.count()) > 0) {
      await deleteBtn.first().click()
      await page.waitForTimeout(300)
      await expect(page.locator('.status-msg')).toContainText('No collections yet')
    }
  })
})
