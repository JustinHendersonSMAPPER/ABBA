// @ts-check
import { test, expect } from '@playwright/test'

test.describe('More Dropdown and Mobile Navigation', () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => localStorage.setItem('abba-onboarded', 'true'))
  })

  test('should show More button in navigation', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('.nav-more-btn')).toBeVisible()
  })

  test('should show dropdown with additional links when clicking More', async ({ page }) => {
    await page.goto('/')
    await page.click('.nav-more-btn')
    await page.waitForTimeout(300)
    const dropdown = page.locator('.nav-dropdown')
    await expect(dropdown).toBeVisible()
    await expect(dropdown.getByText('Compare')).toBeVisible()
    await expect(dropdown.getByText('Words')).toBeVisible()
    await expect(dropdown.getByText('Domains')).toBeVisible()
    await expect(dropdown.getByText('Collections')).toBeVisible()
    await expect(dropdown.getByText('Community')).toBeVisible()
    await expect(dropdown.getByText('Analysis')).toBeVisible()
  })

  test('should navigate to Domains from More dropdown', async ({ page }) => {
    await page.goto('/')
    await page.click('.nav-more-btn')
    await page.waitForTimeout(200)
    await page.locator('.nav-dropdown').getByText('Domains').click()
    await expect(page).toHaveURL(/\/domains/)
  })

  test('should navigate to Community from More dropdown', async ({ page }) => {
    await page.goto('/')
    await page.click('.nav-more-btn')
    await page.waitForTimeout(200)
    await page.locator('.nav-dropdown').getByText('Community').click()
    await expect(page).toHaveURL(/\/community/)
  })

  test('should navigate to Analysis from More dropdown', async ({ page }) => {
    await page.goto('/')
    await page.click('.nav-more-btn')
    await page.waitForTimeout(200)
    await page.locator('.nav-dropdown').getByText('Analysis').click()
    await expect(page).toHaveURL(/\/analysis/)
  })

  test('should show hamburger menu on mobile viewport', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 })
    await page.goto('/')
    await expect(page.locator('.nav-hamburger')).toBeVisible()
  })

  test('should toggle mobile nav open and closed when clicking hamburger', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 })
    await page.goto('/')
    const hamburger = page.locator('.nav-hamburger')
    await expect(hamburger).toBeVisible()
    // Open mobile nav
    await hamburger.click()
    await page.waitForTimeout(300)
    await expect(page.locator('.nav-links.open')).toBeVisible()
    // Close mobile nav
    await hamburger.click()
    await page.waitForTimeout(300)
    await expect(page.locator('.nav-links.open')).not.toBeVisible()
  })
})
