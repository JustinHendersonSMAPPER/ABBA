<template>
  <div class="reading-pane">
    <div class="reading-controls">
      <select v-model="selectedBook" class="control-select" @change="onBookChange">
        <option value="" disabled>Book</option>
        <option v-for="book in books" :key="book.id" :value="book.id">
          {{ book.name }}
        </option>
      </select>

      <select v-model="selectedChapter" class="control-select" @change="loadChapter">
        <option value="" disabled>Ch.</option>
        <option v-for="ch in chapterCount" :key="ch" :value="ch">
          {{ ch }}
        </option>
      </select>

      <LiteraryModeIndicator
        v-if="depth !== 'basic' && chapterData"
        :genre="chapterData.genre || ''"
        :literary-structures="chapterData.literary_structures || []"
      />
    </div>

    <div v-if="api.loading.value" class="loading">Loading...</div>
    <div v-else-if="api.error.value" class="error">{{ api.error.value }}</div>

    <div v-else-if="chapterData" class="reading-text">
      <div v-for="verse in chapterData.verses" :key="verse.number" class="verse-block">
        <sup class="verse-num">{{ verse.number }}</sup>
        <TranslationLens
          v-if="depth !== 'basic' && verse.words"
          :words="verse.words"
          :rich-flags="verse.richness_flags || []"
          @word-click="onWordClick"
        />
        <span v-else class="verse-text">{{ verse.text }}</span>
      </div>
    </div>

    <p v-else class="reading-placeholder">
      Select a book and chapter to begin reading.
    </p>

    <WordJourneyCard
      v-if="selectedWord"
      :detail="selectedWord"
      class="floating-card"
      @close="selectedWord = null"
    />
  </div>
</template>

<script setup>
import { ref, watch, onMounted } from 'vue'
import { useApi } from '../composables/useApi.js'
import TranslationLens from '../components/TranslationLens.vue'
import WordJourneyCard from '../components/WordJourneyCard.vue'
import LiteraryModeIndicator from '../components/LiteraryModeIndicator.vue'

const props = defineProps({
  depth: { type: String, default: 'basic' },
})

const api = useApi()

const books = ref([])
const selectedBook = ref('')
const selectedChapter = ref('')
const chapterCount = ref(0)
const chapterData = ref(null)
const selectedWord = ref(null)

onMounted(async () => {
  const result = await api.getBooks()
  if (result) {
    books.value = result.books || result
  }
})

function onBookChange() {
  selectedChapter.value = ''
  chapterData.value = null
  const book = books.value.find((b) => b.id === selectedBook.value)
  chapterCount.value = book ? book.chapters : 0
}

async function loadChapter() {
  if (!selectedBook.value || !selectedChapter.value) return
  const result = await api.getChapter(selectedBook.value, selectedChapter.value, props.depth)
  if (result) {
    chapterData.value = result
  }
}

watch(() => props.depth, () => {
  if (selectedBook.value && selectedChapter.value) {
    loadChapter()
  }
})

function onWordClick(word, flags) {
  if (flags && flags.strongs) {
    selectedWord.value = {
      original: flags.original || word,
      transliteration: flags.transliteration || '',
      gloss: flags.gloss || word,
      strongs: flags.strongs,
      morphology: flags.morphology || '',
      semantic_domain: flags.semantic_domain || '',
      occurrences: flags.occurrences,
    }
  }
}
</script>

<style scoped>
.reading-controls {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 1.5rem;
  font-family: var(--font-ui);
}

.control-select {
  padding: 0.4rem 0.6rem;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  background: white;
  font-size: 0.9rem;
  font-family: var(--font-ui);
  color: var(--color-text);
}

.loading,
.error,
.reading-placeholder {
  font-family: var(--font-ui);
  font-size: 0.9rem;
  opacity: 0.6;
  padding: 2rem 0;
}

.error {
  color: #c0392b;
  opacity: 1;
}

.verse-block {
  display: inline;
}

.verse-num {
  font-size: 0.7em;
  font-weight: 600;
  color: var(--color-accent);
  margin-right: 0.2em;
  font-family: var(--font-ui);
}

.verse-text {
  font-family: var(--font-reading);
}

.floating-card {
  position: fixed;
  bottom: 2rem;
  right: 2rem;
  z-index: 30;
}
</style>
