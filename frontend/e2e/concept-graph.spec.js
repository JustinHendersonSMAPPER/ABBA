// @ts-check
import { test, expect } from '@playwright/test'

test.describe('Concept Explorer with Graph', () => {
  test('should show graph after concept discovery', async ({ page }) => {
    await page.route('**/api/v1/discover*', async (route) => {
      await route.fulfill({
        json: {
          matched_concepts: [{ name: 'grace', verse_count: 42, description: "God's unmerited favor" }],
          matched_life_topics: [],
          suggested_searches: [],
        },
      })
    })
    await page.route('**/api/v1/graph/**', async (route) => {
      await route.fulfill({
        json: {
          center_concept: 'grace',
          nodes: [
            { name: 'grace', is_center: true },
            { name: 'mercy', is_center: false },
            { name: 'forgiveness', is_center: false },
          ],
          relationships: [
            { source_concept: 'grace', target_concept: 'mercy', relationship_type: 'related', weight: 0.85 },
            { source_concept: 'grace', target_concept: 'forgiveness', relationship_type: 'enables', weight: 0.75 },
          ],
        },
      })
    })
    await page.goto('/discover')
    await page.fill('.search-input', 'grace')
    await page.click('.search-btn')
    await expect(page.locator('.concept-graph-view')).toBeVisible()
    await expect(page.locator('.graph-title')).toContainText('grace')
  })

  test('should show SVG graph with nodes and edges', async ({ page }) => {
    await page.route('**/api/v1/discover*', async (route) => {
      await route.fulfill({
        json: { matched_concepts: [{ name: 'love', verse_count: 100 }], matched_life_topics: [], suggested_searches: [] },
      })
    })
    await page.route('**/api/v1/graph/**', async (route) => {
      await route.fulfill({
        json: {
          center_concept: 'love',
          nodes: [{ name: 'love', is_center: true }, { name: 'joy', is_center: false }],
          relationships: [{ source_concept: 'love', target_concept: 'joy', relationship_type: 'related', weight: 0.7 }],
        },
      })
    })
    await page.goto('/discover')
    await page.fill('.search-input', 'love')
    await page.click('.search-btn')
    await expect(page.locator('.graph-svg')).toBeVisible()
    await expect(page.locator('.graph-node')).toHaveCount(2)
    await expect(page.locator('.graph-edge')).toHaveCount(1)
  })

  test('should show relationship table below graph', async ({ page }) => {
    await page.route('**/api/v1/discover*', async (route) => {
      await route.fulfill({
        json: { matched_concepts: [{ name: 'faith', verse_count: 80 }], matched_life_topics: [], suggested_searches: [] },
      })
    })
    await page.route('**/api/v1/graph/**', async (route) => {
      await route.fulfill({
        json: {
          center_concept: 'faith',
          nodes: [{ name: 'faith', is_center: true }, { name: 'trust', is_center: false }],
          relationships: [{ source_concept: 'faith', target_concept: 'trust', relationship_type: 'synonym', weight: 0.9 }],
        },
      })
    })
    await page.goto('/discover')
    await page.fill('.search-input', 'faith')
    await page.click('.search-btn')
    await expect(page.locator('.relationship-list')).toBeVisible()
    await expect(page.locator('.rel-row')).toHaveCount(1)
  })

  test('should click concept card to load its graph', async ({ page }) => {
    let graphCalls = 0
    await page.route('**/api/v1/discover*', async (route) => {
      await route.fulfill({
        json: {
          matched_concepts: [
            { name: 'grace', verse_count: 42 },
            { name: 'mercy', verse_count: 30 },
          ],
          matched_life_topics: [],
          suggested_searches: [],
        },
      })
    })
    await page.route('**/api/v1/graph/**', async (route) => {
      graphCalls++
      const concept = route.request().url().includes('mercy') ? 'mercy' : 'grace'
      await route.fulfill({
        json: {
          center_concept: concept,
          nodes: [{ name: concept, is_center: true }],
          relationships: [],
        },
      })
    })
    await page.goto('/discover')
    await page.fill('.search-input', 'grace')
    await page.click('.search-btn')
    // Wait for initial graph load (auto for first concept)
    await expect(page.locator('.graph-title')).toContainText('grace')
    // Click second concept to change graph
    await page.click('.concept-card:nth-child(2) .concept-link')
    await expect(page.locator('.graph-title')).toContainText('mercy')
  })
})
