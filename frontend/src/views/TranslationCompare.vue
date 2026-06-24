<template>
  <div class="compare-view">
    <header class="compare-header">
      <h1 class="compare-title">Compare Translations</h1>
      <p class="compare-subtitle">See how different translations render the same verse</p>
    </header>

    <div class="compare-controls">
      <select v-model="selectedBook" class="control-select" @change="onBookChange">
        <option value="" disabled>Book</option>
        <option v-for="book in books" :key="book.book_id" :value="book.name">{{ book.name }}</option>
      </select>
      <select v-model="selectedChapter" class="control-select" @change="selectedVerse = '1'">
        <option value="" disabled>Ch.</option>
        <option v-for="ch in chapterCount" :key="ch" :value="String(ch)">{{ ch }}</option>
      </select>
      <input
        v-model="selectedVerse"
        type="number"
        min="1"
        placeholder="Verse"
        class="verse-input"
      />
      <button class="compare-btn" @click="loadComparison" :disabled="!canCompare">Compare</button>
    </div>

    <div class="translation-picker">
      <label class="picker-label">Translations to compare:</label>
      <div class="translation-chips">
        <label
          v-for="t in availableTranslations"
          :key="t.id"
          class="chip-label"
          :class="{ active: selectedTranslations.includes(t.id) }"
        >
          <input type="checkbox" :value="t.id" v-model="selectedTranslations" class="chip-check" />
          {{ t.name || t.id }}
        </label>
      </div>
    </div>

    <div v-if="api.loading.value" class="status-msg">Loading comparison...</div>
    <div v-else-if="api.error.value" class="status-msg error">{{ api.error.value }}</div>

    <div v-if="comparison" class="comparison-results">
      <h2 class="verse-ref">{{ comparison.reference || `${selectedBook} ${selectedChapter}:${selectedVerse}` }}</h2>

      <div v-if="comparison.original_words && comparison.original_words.length" class="original-words">
        <h3 class="section-label">Original Language</h3>
        <div class="word-row">
          <span v-for="(w, i) in comparison.original_words" :key="i" class="orig-word">
            <span class="orig-text">{{ w.original_text || w.transliteration || '' }}</span>
            <span class="orig-gloss">{{ w.english_gloss || '' }}</span>
          </span>
        </div>
      </div>

      <table class="compare-table">
        <thead>
          <tr>
            <th>Translation</th>
            <th>Text</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(t, i) in comparison.translations || []" :key="i">
            <td class="trans-id">{{ t.name || t.translation_id }}</td>
            <td class="trans-text">{{ t.text }}</td>
          </tr>
        </tbody>
      </table>

      <div v-if="comparison.divergences && comparison.divergences.length" class="divergences">
        <h3 class="section-label">Translation Divergences</h3>
        <div v-for="(d, i) in comparison.divergences" :key="i" class="divergence-item">
          <span class="div-word">{{ d.word || d.original_word }}</span>
          <span class="div-note">{{ d.note || d.explanation }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useApi } from '../composables/useApi'
import type { BookInfo, TranslationInfo } from '../types/api'

interface ComparisonWord {
  original_text?: string
  transliteration?: string
  english_gloss?: string
}

interface ComparisonTranslation {
  name?: string
  translation_id?: string
  text: string
}

interface Divergence {
  word?: string
  original_word?: string
  note?: string
  explanation?: string
}

interface ComparisonResult {
  reference?: string
  original_words?: ComparisonWord[]
  translations?: ComparisonTranslation[]
  divergences?: Divergence[]
}

const api = useApi()

const books = ref<BookInfo[]>([])
const selectedBook = ref('')
const selectedChapter = ref('')
const selectedVerse = ref('1')
const chapterCount = ref(0)
const selectedTranslations = ref<string[]>(['BSB'])
const comparison = ref<ComparisonResult | null>(null)
const availableTranslations = ref<TranslationInfo[]>([])

const canCompare = computed(() =>
  selectedBook.value && selectedChapter.value && selectedVerse.value && selectedTranslations.value.length >= 2
)

onMounted(async () => {
  const [booksResult, translationsResult] = await Promise.all([
    api.getBooks(),
    api.getTranslations(),
  ])

  if (booksResult) {
    if (Array.isArray(booksResult)) {
      books.value = booksResult as BookInfo[]
    } else {
      books.value = ((booksResult as Record<string, unknown>).books as BookInfo[]) || []
    }
  }

  if (translationsResult && translationsResult.length > 0) {
    availableTranslations.value = translationsResult
    // Default: BSB plus eng_kjv if present
    const ids = translationsResult.map(t => t.id)
    selectedTranslations.value = ['BSB', ...(ids.includes('eng_kjv') ? ['eng_kjv'] : [])].filter(id => ids.includes(id))
    if (selectedTranslations.value.length === 0) {
      selectedTranslations.value = [ids[0]]
    }
  } else {
    // Fallback to a single BSB stub when the endpoint is unavailable
    availableTranslations.value = [{ id: 'BSB', name: 'BSB', language: 'eng' }]
  }
})

function onBookChange(): void {
  selectedChapter.value = ''
  const book = books.value.find((b: BookInfo) => b.name === selectedBook.value)
  chapterCount.value = book ? book.chapter_count : 0
}

async function loadComparison(): Promise<void> {
  if (!canCompare.value) return
  comparison.value = null
  const data = await api.compareTranslations(
    selectedBook.value,
    selectedChapter.value,
    selectedVerse.value,
    selectedTranslations.value
  )
  if (data) comparison.value = data as ComparisonResult
}
</script>

<style scoped>
.compare-header { margin-bottom: 1rem; }
.compare-title { font-family: var(--font-ui); font-size: 1.5rem; margin-bottom: 0.25rem; }
.compare-subtitle { font-size: 0.9rem; opacity: 0.6; }

.compare-controls {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1rem;
  flex-wrap: wrap;
  align-items: center;
}
.control-select, .verse-input {
  padding: 0.4rem 0.6rem;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  font-family: var(--font-ui);
  font-size: 0.9rem;
  background: var(--color-surface);
  color: var(--color-text);
}
.verse-input { width: 70px; }
.compare-btn {
  padding: 0.4rem 1rem;
  background: var(--color-accent);
  color: white;
  border: none;
  border-radius: 4px;
  font-family: var(--font-ui);
  font-weight: 600;
  cursor: pointer;
}
.compare-btn:disabled { opacity: 0.5; cursor: not-allowed; }

.translation-picker { margin-bottom: 1.5rem; }
.picker-label { font-family: var(--font-ui); font-size: 0.85rem; opacity: 0.7; display: block; margin-bottom: 0.5rem; }
.translation-chips { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.chip-label {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.3rem 0.6rem;
  border: 1px solid var(--color-border);
  border-radius: 16px;
  font-family: var(--font-ui);
  font-size: 0.8rem;
  cursor: pointer;
  transition: background 0.15s;
}
.chip-label.active { background: var(--color-accent); color: white; border-color: var(--color-accent); }
.chip-check { display: none; }

.verse-ref { font-family: var(--font-ui); font-size: 1.25rem; margin-bottom: 1rem; }
.section-label { font-family: var(--font-ui); font-size: 0.85rem; color: var(--color-accent); margin-bottom: 0.5rem; margin-top: 1rem; }

.original-words { margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid var(--color-border); }
.word-row { display: flex; flex-wrap: wrap; gap: 0.75rem; }
.orig-word { text-align: center; }
.orig-text { display: block; font-size: 1.1rem; font-weight: 600; }
.orig-gloss { display: block; font-size: 0.75rem; opacity: 0.6; }

.compare-table { width: 100%; border-collapse: collapse; }
.compare-table th { font-family: var(--font-ui); font-size: 0.8rem; text-align: left; padding: 0.5rem; border-bottom: 2px solid var(--color-border); color: var(--color-accent); }
.compare-table td { padding: 0.5rem; border-bottom: 1px solid var(--color-border); vertical-align: top; }
.trans-id { font-family: var(--font-ui); font-weight: 600; font-size: 0.8rem; opacity: 0.6; min-width: 60px; }
.trans-text { font-family: var(--font-reading); line-height: 1.6; }

.divergences { margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--color-border); }
.divergence-item { padding: 0.35rem 0; font-size: 0.9rem; }
.div-word { font-weight: 600; margin-right: 0.5rem; }
.div-note { opacity: 0.7; }

.status-msg { font-family: var(--font-ui); font-size: 0.9rem; opacity: 0.6; padding: 1rem 0; }
.error { color: #c0392b; opacity: 1; }
</style>
