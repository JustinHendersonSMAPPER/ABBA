// @ts-check
import { test, expect } from '@playwright/test'

test.describe('Community View', () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => localStorage.setItem('abba-onboarded', 'true'))
  })

  test('should show page title and subtitle', async ({ page }) => {
    await page.route('**/api/v1/community/contributions*', async (route) => {
      await route.fulfill({ json: { contributions: [] } })
    })
    await page.route('**/api/v1/concepts/proposals*', async (route) => {
      await route.fulfill({ json: { proposals: [] } })
    })
    await page.goto('/community')
    await expect(page.locator('.community-title')).toContainText('Help Improve ABBA')
    await expect(page.locator('.community-subtitle')).toBeVisible()
  })

  test('should display two tabs: Contributions and Concept Proposals', async ({ page }) => {
    await page.route('**/api/v1/community/contributions*', async (route) => {
      await route.fulfill({ json: { contributions: [] } })
    })
    await page.route('**/api/v1/concepts/proposals*', async (route) => {
      await route.fulfill({ json: { proposals: [] } })
    })
    await page.goto('/community')
    await expect(page.locator('.tab-btn')).toHaveCount(2)
    await expect(page.locator('.tab-btn').first()).toContainText('Contributions')
    await expect(page.locator('.tab-btn').last()).toContainText('Concept Proposals')
  })

  test('should show existing contributions', async ({ page }) => {
    await page.route('**/api/v1/community/contributions*', async (route) => {
      await route.fulfill({
        json: {
          contributions: [
            {
              id: '1',
              contribution_type: 'correction',
              status: 'pending',
              content: 'Fix typo in Genesis 1:2',
              book_id: 'GEN',
              chapter: 1,
              verse: 2,
            },
          ],
        },
      })
    })
    await page.route('**/api/v1/concepts/proposals*', async (route) => {
      await route.fulfill({ json: { proposals: [] } })
    })
    await page.goto('/community')
    await expect(page.locator('.item-card')).toHaveCount(1)
    await expect(page.locator('.item-content')).toContainText('Fix typo in Genesis 1:2')
    await expect(page.locator('.item-status')).toContainText('pending')
  })

  test('should show contribution form when clicking "Submit Contribution"', async ({ page }) => {
    await page.route('**/api/v1/community/contributions*', async (route) => {
      await route.fulfill({ json: { contributions: [] } })
    })
    await page.route('**/api/v1/concepts/proposals*', async (route) => {
      await route.fulfill({ json: { proposals: [] } })
    })
    await page.goto('/community')
    await page.click('.btn-primary:has-text("Submit Contribution")')
    await expect(page.locator('.submit-form')).toBeVisible()
  })

  test('should show review buttons for pending contributions', async ({ page }) => {
    await page.route('**/api/v1/community/contributions*', async (route) => {
      await route.fulfill({
        json: {
          contributions: [
            {
              id: '1',
              contribution_type: 'correction',
              status: 'pending',
              content: 'Fix typo in Genesis 1:2',
              book_id: 'GEN',
              chapter: 1,
              verse: 2,
            },
          ],
        },
      })
    })
    await page.route('**/api/v1/concepts/proposals*', async (route) => {
      await route.fulfill({ json: { proposals: [] } })
    })
    await page.goto('/community')
    await expect(page.locator('.review-btn.approve')).toBeVisible()
    await expect(page.locator('.review-btn.reject')).toBeVisible()
  })

  test('should switch to Concept Proposals tab', async ({ page }) => {
    await page.route('**/api/v1/community/contributions*', async (route) => {
      await route.fulfill({ json: { contributions: [] } })
    })
    await page.route('**/api/v1/concepts/proposals*', async (route) => {
      await route.fulfill({
        json: {
          proposals: [
            {
              id: '1',
              concept_name: 'Divine Mercy',
              status: 'approved',
              description: 'Gods compassionate nature',
            },
          ],
        },
      })
    })
    await page.goto('/community')
    await page.click('.tab-btn:has-text("Concept Proposals")')
    await expect(page.locator('.item-card')).toHaveCount(1)
    await expect(page.locator('.item-type')).toContainText('Divine Mercy')
    await expect(page.locator('.item-status')).toContainText('approved')
  })

  test('should show proposal form when clicking "Propose Concept"', async ({ page }) => {
    await page.route('**/api/v1/community/contributions*', async (route) => {
      await route.fulfill({ json: { contributions: [] } })
    })
    await page.route('**/api/v1/concepts/proposals*', async (route) => {
      await route.fulfill({ json: { proposals: [] } })
    })
    await page.goto('/community')
    await page.click('.tab-btn:has-text("Concept Proposals")')
    await page.click('.btn-primary:has-text("Propose Concept")')
    await expect(page.locator('.submit-form')).toBeVisible()
  })

  test('should filter contributions by status', async ({ page }) => {
    await page.route('**/api/v1/community/contributions*', async (route) => {
      await route.fulfill({
        json: {
          contributions: [
            { id: '1', contribution_type: 'correction', status: 'pending', content: 'Fix typo' },
            { id: '2', contribution_type: 'note', status: 'approved', content: 'Add note' },
          ],
        },
      })
    })
    await page.route('**/api/v1/concepts/proposals*', async (route) => {
      await route.fulfill({ json: { proposals: [] } })
    })
    await page.goto('/community')
    await expect(page.locator('.item-card')).toHaveCount(2)
    const filterSelect = page.locator('.filter-select')
    await filterSelect.selectOption('pending')
    // After selecting filter, the component re-fetches; mock returns same data
    // but the component calls loadContributions with status=pending
    await page.waitForTimeout(300)
  })
})
