// @ts-check
import { test, expect } from '@playwright/test'

test.describe('Shared View', () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => localStorage.setItem('abba-onboarded', 'true'))
  })

  test('should display shared verse content', async ({ page }) => {
    await page.route('**/api/v1/share/abc123', async (route) => {
      await route.fulfill({
        json: {
          title: 'Genesis 1:1',
          share_type: 'verse',
          content: { text: 'In the beginning...', book: 'GEN', chapter: 1, verse: 1 },
          created_at: '2025-06-01',
        },
      })
    })
    await page.goto('/shared/abc123')
    await expect(page.locator('.shared-verse')).toContainText('In the beginning...')
  })

  test('should show "Shared" badge and title', async ({ page }) => {
    await page.route('**/api/v1/share/abc123', async (route) => {
      await route.fulfill({
        json: {
          title: 'Genesis 1:1',
          share_type: 'verse',
          content: { text: 'In the beginning...', book: 'GEN', chapter: 1, verse: 1 },
          created_at: '2025-06-01',
        },
      })
    })
    await page.goto('/shared/abc123')
    await expect(page.locator('.shared-badge')).toContainText('Shared')
    await expect(page.locator('.shared-title')).toContainText('Genesis 1:1')
  })

  test('should show "Open in Study View" link that points to /study/GEN/1/1', async ({ page }) => {
    await page.route('**/api/v1/share/abc123', async (route) => {
      await route.fulfill({
        json: {
          title: 'Genesis 1:1',
          share_type: 'verse',
          content: { text: 'In the beginning...', book: 'GEN', chapter: 1, verse: 1 },
          created_at: '2025-06-01',
        },
      })
    })
    await page.goto('/shared/abc123')
    const studyLink = page.locator('.open-link')
    await expect(studyLink).toContainText('Open in Study View')
    await expect(studyLink).toHaveAttribute('href', /\/study\/GEN\/1\/1/)
  })

  test('should show "not found" when share token is invalid', async ({ page }) => {
    await page.route('**/api/v1/share/invalid-token', async (route) => {
      await route.fulfill({ status: 404, json: { error: 'Not found' } })
    })
    await page.goto('/shared/invalid-token')
    await expect(page.locator('.error')).toBeVisible()
  })

  test('should show loading state initially', async ({ page }) => {
    await page.route('**/api/v1/share/abc123', async (route) => {
      await new Promise((r) => setTimeout(r, 500))
      await route.fulfill({
        json: {
          title: 'Genesis 1:1',
          share_type: 'verse',
          content: { text: 'In the beginning...', book: 'GEN', chapter: 1, verse: 1 },
          created_at: '2025-06-01',
        },
      })
    })
    await page.goto('/shared/abc123')
    await expect(page.locator('.loading')).toBeVisible()
  })
})
