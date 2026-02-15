// @ts-check
import { test, expect } from '@playwright/test'

test.describe('Export and Audio', () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => localStorage.setItem('abba-onboarded', 'true'))
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
    await page.route('**/api/v1/notes/**', async (route) => {
      await route.fulfill({ json: { notes: [] } })
    })
    await page.route('**/api/v1/audio/GEN/1*', async (route) => {
      await route.fulfill({
        json: { url: 'https://example.com/audio.mp3', format: 'mp3', duration: 180 },
      })
    })
    await page.route('**/api/v1/export/verse/engbsb/GEN/1/1*', async (route) => {
      await route.fulfill({
        contentType: 'text/markdown',
        body: '# Genesis 1:1\n\nIn the beginning God created the heavens and the earth.',
      })
    })
  })

  test('should show Audio button for verse view', async ({ page }) => {
    await page.goto('/study/GEN/1/1')
    await expect(page.locator('.action-btn:has-text("Audio")')).toBeVisible()
  })

  test('should toggle audio player when clicking Audio', async ({ page }) => {
    await page.goto('/study/GEN/1/1')
    await page.click('.action-btn:has-text("Audio")')
    await page.waitForTimeout(300)
    await expect(page.locator('.audio-player')).toBeVisible()
  })

  test('should show audio player with controls element', async ({ page }) => {
    await page.goto('/study/GEN/1/1')
    await page.click('.action-btn:has-text("Audio")')
    await page.waitForTimeout(300)
    const audioElement = page.locator('.audio-player audio, .audio-element')
    await expect(audioElement).toBeVisible()
  })

  test('should show export button and trigger export', async ({ page }) => {
    await page.goto('/study/GEN/1/1')
    const exportBtn = page.locator('.action-btn:has-text("Export")')
    await expect(exportBtn).toBeVisible()
    // Verify the export endpoint is called when button is clicked
    const exportCalled = page.waitForResponse((r) => r.url().includes('/export/'))
    await exportBtn.click()
    const response = await exportCalled
    expect(response.status()).toBe(200)
  })

  test('should hide audio player when clicking Audio again', async ({ page }) => {
    await page.goto('/study/GEN/1/1')
    const audioBtn = page.locator('.action-btn:has-text("Audio")')
    // Open audio player
    await audioBtn.click()
    await page.waitForTimeout(300)
    await expect(page.locator('.audio-player')).toBeVisible()
    // Close audio player by toggling
    await audioBtn.click()
    await page.waitForTimeout(300)
    await expect(page.locator('.audio-player')).not.toBeVisible()
  })
})
