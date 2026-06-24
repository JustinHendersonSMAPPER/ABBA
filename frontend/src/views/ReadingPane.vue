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

      <button v-if="selectedChapter" class="control-btn" @click="toggleAudio">Audio</button>

      <LiteraryModeIndicator
        v-if="depth !== 'basic' && chapterData"
        :genre="chapterData.genre || ''"
        :literary-structures="chapterData.literary_structures || []"
      />
    </div>

    <AudioPlayer v-if="audioData" :audio="audioData" />

    <div v-if="books.length === 0 && !api.loading.value && !api.error.value" class="skeleton-controls">
      <div class="skeleton-bar" style="width: 90px;"></div>
      <div class="skeleton-bar" style="width: 60px;"></div>
    </div>
    <LoadingState v-if="api.loading.value" label="Loading…" />
    <div v-else-if="api.error.value" class="error">{{ api.error.value }}</div>

    <div v-else-if="chapterData" class="reading-text">
      <template v-for="verse in chapterData.verses" :key="verse.number">
        <h3 v-if="getPassageTitle(verse.number)" class="passage-heading">{{ getPassageTitle(verse.number) }}</h3>
        <div class="verse-block">
          <button class="verse-num verse-link" @click="router.push('/study/' + selectedBook + '/' + selectedChapter + '/' + verse.number)">{{ verse.number }}</button>
          <TranslationLens
            v-if="depth !== 'basic' && verse.words"
            :words="verse.words"
            :rich-flags="(verse.richness_flags || []).map(f => typeof f === 'string' ? { richness: 0, explanation: f } : f)"
            @word-click="onWordClick"
          />
          <span v-else class="verse-text">{{ verse.text }}</span>
        </div>
      </template>
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

<script setup lang="ts">
import { ref, watch, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useApi } from '../composables/useApi'
import type { BookInfo, ChapterData, AudioResource, PassageInfo, GenreShift } from '../types/api'
import TranslationLens from '../components/TranslationLens.vue'
import WordJourneyCard from '../components/WordJourneyCard.vue'
import LiteraryModeIndicator from '../components/LiteraryModeIndicator.vue'
import AudioPlayer from '../components/AudioPlayer.vue'
import LoadingState from '../components/LoadingState.vue'

const props = defineProps({
  depth: { type: String, default: 'basic' },
})

const api = useApi()
const router = useRouter()

const books = ref<BookInfo[]>([])
const selectedBook = ref<string>('')
const selectedChapter = ref<string>('')
const chapterCount = ref<number>(0)
const chapterData = ref<ChapterData | null>(null)
interface WordDetail {
  text?: string
  original?: string
  transliteration?: string
  gloss?: string
  strongs?: string
  morphology?: string
  semantic_domain?: string
  occurrences?: number
}

const selectedWord = ref<WordDetail | null>(null)
const audioData = ref<AudioResource | null>(null)
const passages = ref<PassageInfo[]>([])
const genreShifts = ref<GenreShift[]>([])

onMounted(async () => {
  const result = await api.getBooks()
  if (result) {
    if (Array.isArray(result)) {
      books.value = result as BookInfo[]
    } else {
      books.value = (result as Record<string, unknown>).books as BookInfo[] || []
    }
  }
  // Default to John 1 on first load
  if (!selectedBook.value) {
    const defaultBook = books.value.find((b) => b.id === 'JHN') || books.value[0]
    if (defaultBook) {
      selectedBook.value = defaultBook.id
      onBookChange()
      selectedChapter.value = '1'
      await loadChapter()
    }
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
  if (props.depth !== 'basic') {
    const [passageData, genreData, audio] = await Promise.all([
      api.getPassages(selectedBook.value, selectedChapter.value),
      api.getGenreShifts(selectedBook.value),
      api.getAudioResource(selectedBook.value, selectedChapter.value),
    ])
    passages.value = (passageData as PassageInfo[]) || []
    genreShifts.value = (genreData as GenreShift[]) || []
    if (audio) audioData.value = audio
  }
}

watch(() => props.depth, () => {
  if (selectedBook.value && selectedChapter.value) {
    loadChapter()
  }
})

function onWordClick(word: string | Record<string, unknown>, flags: Record<string, unknown>) {
  const wordStr = typeof word === 'string' ? word : (word as Record<string, unknown>).text as string || ''
  const strongs = flags.strongs as string | undefined
  if (flags && strongs) {
    selectedWord.value = {
      original: (flags.original as string | undefined) || wordStr,
      transliteration: (flags.transliteration as string | undefined) || '',
      gloss: (flags.gloss as string | undefined) || wordStr,
      strongs: strongs,
      morphology: (flags.morphology as string | undefined) || '',
      semantic_domain: (flags.semantic_domain as string | undefined) || '',
      occurrences: flags.occurrences as number | undefined,
    }
  }
}

function getPassageTitle(verseNum: number): string | null {
  const passage = passages.value.find(p => p.start_verse === verseNum)
  return passage ? passage.title : null
}

async function toggleAudio(): Promise<void> {
  if (audioData.value) { audioData.value = null; return }
  const data = await api.getAudioResource(selectedBook.value, selectedChapter.value)
  if (data) audioData.value = data
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
  background: var(--color-surface);
  font-size: 0.9rem;
  font-family: var(--font-ui);
  color: var(--color-text);
}

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
  margin-bottom: 0.35em;
}

.verse-num {
  font-size: 0.72em;
  font-weight: 700;
  color: var(--color-accent);
  margin-right: 0.25em;
  font-family: var(--font-ui);
  opacity: 0.85;
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

.verse-link {
  cursor: pointer;
  background: none;
  border: none;
  padding: 0;
  vertical-align: super;
  line-height: 1;
  font: inherit;
}
.verse-link:hover { color: var(--color-accent); text-decoration: underline; }
.passage-heading { font-family: var(--font-ui); font-size: 0.9rem; font-weight: 600; color: var(--color-accent); margin: 1rem 0 0.5rem; display: block; }
.control-btn { padding: 0.3rem 0.6rem; border: 1px solid var(--color-border); border-radius: 4px; background: var(--color-surface); font-family: var(--font-ui); font-size: 0.8rem; cursor: pointer; color: var(--color-text); }
.control-btn:hover { border-color: var(--color-accent); color: var(--color-accent); }

.skeleton-controls {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1.5rem;
}

.skeleton-bar {
  height: 34px;
  background: var(--color-border);
  border-radius: 4px;
  animation: skeleton-pulse 1.4s ease-in-out infinite;
}

@keyframes skeleton-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}
</style>
