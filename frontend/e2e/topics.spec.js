// @ts-check
import { test, expect } from '@playwright/test'

test.describe('Life Topics', () => {
  test('should display page title and search bar', async ({ page }) => {
    await page.goto('/topics')
    await expect(page.locator('.page-title')).toHaveText('Life Topics')
    await expect(page.locator('.search-input')).toBeVisible()
  })

  test('should show loading state initially', async ({ page }) => {
    await page.goto('/topics')
    // Either loading or topic grid should appear
    const loadingOrGrid = page.locator('.loading, .topic-grid, .error')
    await expect(loadingOrGrid.first()).toBeVisible()
  })

  test('should filter topics when searching', async ({ page }) => {
    await page.goto('/topics')
    const searchInput = page.locator('.search-input')
    await searchInput.fill('anxiety')
    // Should filter the grid or show empty state
    await expect(page.locator('.topic-grid, .empty-state')).toBeVisible()
  })

  test('should show empty state for no matches', async ({ page }) => {
    await page.goto('/topics')
    // Wait for initial load
    await page.waitForTimeout(500)
    const searchInput = page.locator('.search-input')
    await searchInput.fill('zzzznonexistenttopic')
    await expect(page.locator('.empty-state')).toBeVisible()
  })
})
