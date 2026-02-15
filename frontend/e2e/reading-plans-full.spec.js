// @ts-check
import { test, expect } from '@playwright/test'

test.describe('Reading Plans Full Flow', () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => localStorage.setItem('abba-onboarded', 'true'))
  })

  test('should show page title "Reading Plans"', async ({ page }) => {
    await page.route('**/api/v1/reading-plans', async (route) => {
      await route.fulfill({ json: { plans: [] } })
    })
    await page.goto('/plans')
    await expect(page.locator('.page-title')).toContainText('Reading Plans')
  })

  test('should display plan cards', async ({ page }) => {
    await page.route('**/api/v1/reading-plans', async (route) => {
      await route.fulfill({
        json: {
          plans: [
            {
              id: 'psalms-30',
              name: '30 Days in Psalms',
              description: 'A month-long journey through the Psalms',
              duration: 30,
              category: 'devotional',
            },
          ],
        },
      })
    })
    await page.goto('/plans')
    await expect(page.locator('.plan-card')).toHaveCount(1)
  })

  test('should show plan name, description, duration', async ({ page }) => {
    await page.route('**/api/v1/reading-plans', async (route) => {
      await route.fulfill({
        json: {
          plans: [
            {
              id: 'psalms-30',
              name: '30 Days in Psalms',
              description: 'A month-long journey through the Psalms',
              duration: 30,
              category: 'devotional',
            },
          ],
        },
      })
    })
    await page.goto('/plans')
    await expect(page.locator('.plan-name')).toContainText('30 Days in Psalms')
    await expect(page.locator('.plan-desc')).toContainText('A month-long journey through the Psalms')
    await expect(page.locator('.plan-meta')).toContainText('30')
  })

  test('should show plan detail when clicking a card', async ({ page }) => {
    await page.route('**/api/v1/reading-plans', async (route) => {
      const url = route.request().url()
      if (!url.includes('psalms-30')) {
        await route.fulfill({
          json: {
            plans: [
              {
                id: 'psalms-30',
                name: '30 Days in Psalms',
                description: 'A month-long journey through the Psalms',
                duration: 30,
                category: 'devotional',
              },
            ],
          },
        })
      }
    })
    await page.route('**/api/v1/reading-plans/psalms-30', async (route) => {
      await route.fulfill({
        json: {
          id: 'psalms-30',
          name: '30 Days in Psalms',
          entries: [
            {
              day: 1,
              title: 'Day 1: The Blessed Life',
              readings: [{ book: 'PSA', chapter: 1, reference: 'Psalm 1' }],
              reflection: 'What does it mean to delight in Gods law?',
            },
          ],
        },
      })
    })
    await page.goto('/plans')
    await page.click('.plan-card')
    await expect(page.locator('.detail-title')).toContainText('30 Days in Psalms')
  })

  test('should show entries with day numbers, titles, and reading links', async ({ page }) => {
    await page.route('**/api/v1/reading-plans', async (route) => {
      const url = route.request().url()
      if (!url.includes('psalms-30')) {
        await route.fulfill({
          json: {
            plans: [
              {
                id: 'psalms-30',
                name: '30 Days in Psalms',
                description: 'A month-long journey',
                duration: 30,
                category: 'devotional',
              },
            ],
          },
        })
      }
    })
    await page.route('**/api/v1/reading-plans/psalms-30', async (route) => {
      await route.fulfill({
        json: {
          id: 'psalms-30',
          name: '30 Days in Psalms',
          entries: [
            {
              day: 1,
              title: 'Day 1: The Blessed Life',
              readings: [{ book: 'PSA', chapter: 1, reference: 'Psalm 1' }],
              reflection: 'What does it mean to delight in Gods law?',
            },
          ],
        },
      })
    })
    await page.goto('/plans')
    await page.click('.plan-card')
    await expect(page.locator('.plan-entry')).toHaveCount(1)
    await expect(page.locator('.entry-day')).toContainText('Day 1')
    await expect(page.locator('.entry-content')).toContainText('The Blessed Life')
    await expect(page.locator('.entry-link')).toContainText('Psalm 1')
  })

  test('should navigate back from detail to plan list', async ({ page }) => {
    await page.route('**/api/v1/reading-plans', async (route) => {
      const url = route.request().url()
      if (!url.includes('psalms-30')) {
        await route.fulfill({
          json: {
            plans: [
              {
                id: 'psalms-30',
                name: '30 Days in Psalms',
                description: 'A month-long journey',
                duration: 30,
                category: 'devotional',
              },
            ],
          },
        })
      }
    })
    await page.route('**/api/v1/reading-plans/psalms-30', async (route) => {
      await route.fulfill({
        json: {
          id: 'psalms-30',
          name: '30 Days in Psalms',
          entries: [],
        },
      })
    })
    await page.goto('/plans')
    await page.click('.plan-card')
    await expect(page.locator('.plan-detail')).toBeVisible()
    await page.click('.back-btn')
    await expect(page.locator('.plan-card')).toBeVisible()
  })
})
