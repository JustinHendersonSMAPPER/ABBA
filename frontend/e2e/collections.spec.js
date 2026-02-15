// @ts-check
import { test, expect } from '@playwright/test'

test.describe('Collections', () => {
  test('should show collections page with create button', async ({ page }) => {
    await page.route('**/api/v1/collections', async (route) => {
      if (route.request().method() === 'GET') {
        await route.fulfill({ json: { collections: [] } })
      }
    })
    await page.goto('/collections')
    await expect(page.locator('.collections-title, h1')).toContainText('Collections')
    await expect(page.locator('.btn-primary')).toContainText('New Collection')
  })

  test('should show create form when clicking New Collection', async ({ page }) => {
    await page.route('**/api/v1/collections', async (route) => {
      await route.fulfill({ json: { collections: [] } })
    })
    await page.goto('/collections')
    await page.click('.btn-primary')
    await expect(page.locator('.create-form')).toBeVisible()
    await expect(page.locator('input[placeholder="Collection name"]')).toBeVisible()
  })

  test('should display existing collections as cards', async ({ page }) => {
    await page.route('**/api/v1/collections', async (route) => {
      if (route.request().method() === 'GET') {
        await route.fulfill({
          json: {
            collections: [
              { id: 1, name: 'Favorite Psalms', description: 'My favorite psalms', item_count: 5 },
              { id: 2, name: 'Study Notes', description: '', item_count: 3 },
            ],
          },
        })
      }
    })
    await page.goto('/collections')
    await expect(page.locator('.collection-card')).toHaveCount(2)
    await expect(page.locator('.col-name').first()).toContainText('Favorite Psalms')
    await expect(page.locator('.col-count').first()).toContainText('5 verses')
  })

  test('should show collection items when clicking a collection', async ({ page }) => {
    await page.route('**/api/v1/collections', async (route) => {
      await route.fulfill({
        json: { collections: [{ id: 1, name: 'My Collection', item_count: 1 }] },
      })
    })
    await page.route('**/api/v1/collections/1/items', async (route) => {
      await route.fulfill({
        json: {
          items: [{ book_id: 'PSA', book_name: 'Psalms', chapter: 23, verse: 1, text: 'The Lord is my shepherd...' }],
        },
      })
    })
    await page.goto('/collections')
    await page.click('.collection-card')
    await expect(page.locator('.detail-name')).toContainText('My Collection')
    await expect(page.locator('.item-card')).toHaveCount(1)
    await expect(page.locator('.item-ref')).toContainText('Psalms 23:1')
  })

  test('should navigate back from collection detail', async ({ page }) => {
    await page.route('**/api/v1/collections', async (route) => {
      await route.fulfill({
        json: { collections: [{ id: 1, name: 'Test', item_count: 0 }] },
      })
    })
    await page.route('**/api/v1/collections/1/items', async (route) => {
      await route.fulfill({ json: { items: [] } })
    })
    await page.goto('/collections')
    await page.click('.collection-card')
    await expect(page.locator('.back-btn')).toBeVisible()
    await page.click('.back-btn')
    await expect(page.locator('.collections-grid')).toBeVisible()
  })

  test('should show empty state when no collections', async ({ page }) => {
    await page.route('**/api/v1/collections', async (route) => {
      await route.fulfill({ json: { collections: [] } })
    })
    await page.goto('/collections')
    await expect(page.locator('.status-msg')).toContainText('No collections yet')
  })

  test('should navigate to collections from nav', async ({ page }) => {
    await page.goto('/')
    await page.click('.nav-links >> text=Collections')
    await expect(page).toHaveURL('/collections')
  })
})
