// @ts-check
import { test, expect } from '@playwright/test'

test.describe('Onboarding Overlay', () => {
  test('should show onboarding on first visit', async ({ page }) => {
    await page.addInitScript(() => localStorage.clear())
    await page.goto('/')
    await expect(page.locator('.onboarding-overlay')).toBeVisible()
  })

  test('should show step 1 with Welcome to ABBA title', async ({ page }) => {
    await page.addInitScript(() => localStorage.clear())
    await page.goto('/')
    await expect(page.locator('.onboarding-overlay')).toBeVisible()
    await expect(page.locator('.step-title')).toContainText('Welcome to ABBA')
  })

  test('should advance to step 2 with Start with What Matters when clicking Next', async ({ page }) => {
    await page.addInitScript(() => localStorage.clear())
    await page.goto('/')
    await expect(page.locator('.onboarding-overlay')).toBeVisible()
    await page.click('.nav-btn:has-text("Next")')
    await page.waitForTimeout(300)
    await expect(page.locator('.step-title')).toContainText('Start with What Matters')
  })

  test('should advance to step 3 with Or Just Read when clicking Next again', async ({ page }) => {
    await page.addInitScript(() => localStorage.clear())
    await page.goto('/')
    await expect(page.locator('.onboarding-overlay')).toBeVisible()
    await page.click('.nav-btn:has-text("Next")')
    await page.waitForTimeout(300)
    await page.click('.nav-btn:has-text("Next")')
    await page.waitForTimeout(300)
    await expect(page.locator('.step-title')).toContainText('Or Just Read')
  })

  test('should dismiss onboarding and set localStorage when clicking Get Started', async ({ page }) => {
    await page.addInitScript(() => localStorage.clear())
    await page.goto('/')
    await expect(page.locator('.onboarding-overlay')).toBeVisible()
    // Advance through steps to reach Get Started button
    await page.click('.nav-btn:has-text("Next")')
    await page.waitForTimeout(200)
    await page.click('.nav-btn:has-text("Next")')
    await page.waitForTimeout(200)
    await page.click('.nav-btn:has-text("Get Started")')
    await page.waitForTimeout(300)
    await expect(page.locator('.onboarding-overlay')).not.toBeVisible()
    const onboarded = await page.evaluate(() => localStorage.getItem('abba-onboarded'))
    expect(onboarded).toBe('true')
  })

  test('should NOT show onboarding on subsequent visit', async ({ page }) => {
    await page.addInitScript(() => localStorage.setItem('abba-onboarded', 'true'))
    await page.goto('/')
    await expect(page.locator('.onboarding-overlay')).not.toBeVisible()
  })
})
