<template>
  <div class="study-view">
    <div v-if="api.loading.value" class="loading">Loading verse...</div>
    <div v-else-if="api.error.value" class="error">{{ api.error.value }}</div>

    <template v-else-if="verseData">
      <header class="study-header">
        <h1 class="study-ref">
          {{ verseData.book || book }} {{ verseData.chapter || chapter }}<template v-if="verse">:{{ verseData.verse || verse }}</template>
        </h1>
        <LiteraryModeIndicator
          v-if="verseData.genre || verseData.literary_structures"
          :genre="verseData.genre || ''"
          :literary-structures="verseData.literary_structures || []"
        />
      </header>

      <div class="study-actions">
        <button class="action-btn" @click="addToCollection" title="Save to collection">Bookmark</button>
        <button class="action-btn" @click="doExport('markdown')" title="Export as markdown">Export</button>
        <button class="action-btn" @click="doShare" title="Create shareable link">Share</button>
        <button v-if="verse" class="action-btn" @click="loadAudio" title="Listen to this chapter">Audio</button>
        <router-link
          v-if="verse"
          :to="{ name: 'compare', query: { book: book, chapter: chapter, verse: verse } }"
          class="action-btn action-link"
        >Compare</router-link>
      </div>

      <AudioPlayer v-if="audioData" :audio="audioData" />

      <div v-if="shareUrl" class="share-banner">
        Shareable link: <a :href="shareUrl" target="_blank">{{ shareUrl }}</a>
      </div>

      <section class="study-text">
        <TranslationLens
          v-if="depth !== 'basic' && verseData.words"
          :words="verseData.words"
          :rich-flags="(verseData.richness_flags || []).map(f => typeof f === 'string' ? { richness: 0, explanation: f } : f)"
          @word-click="onWordClick"
        />
        <p v-else class="verse-text">{{ verseData.text }}</p>
      </section>

      <WordJourneyCard
        v-if="selectedWord"
        :detail="selectedWord"
        class="word-card-inline"
        @close="selectedWord = null"
      />

      <section v-if="depth !== 'basic' && verseData.parallel_translations" class="study-section">
        <h2 class="section-heading">Translations</h2>
        <div v-for="(trans, i) in verseData.parallel_translations" :key="i" class="translation-row">
          <span class="trans-name">{{ trans.name }}</span>
          <span class="trans-text">{{ trans.text }}</span>
        </div>
      </section>

      <SemanticDomainBadge
        v-if="verseData.semantic_domains && verseData.semantic_domains.length"
        :domains="verseData.semantic_domains"
      />

      <SyntaxTreeView v-if="verseData.syntax_tree" :tree="verseData.syntax_tree" />

      <DiscourseView
        v-if="verseData.discourse_units && verseData.discourse_units.length"
        :units="verseData.discourse_units"
      />

      <ManuscriptVariants
        v-if="verseData.manuscript_variants && verseData.manuscript_variants.length"
        :variants="verseData.manuscript_variants"
      />

      <section v-if="contextData" class="study-section">
        <h2 class="section-heading">Context</h2>
        <div v-if="contextData.cultural && contextData.cultural.length">
          <h3 class="sub-heading">Cultural Background</h3>
          <p v-for="(item, i) in contextData.cultural" :key="'c-' + i" class="context-text">
            {{ typeof item === 'object' && item !== null ? (item as { text?: string }).text || '' : String(item) }}
          </p>
        </div>
        <div v-if="contextData.historical">
          <h3 class="sub-heading">Historical Setting</h3>
          <p class="context-text">{{ contextData.historical }}</p>
        </div>
      </section>

      <section v-if="crossRefs && crossRefs.length" class="study-section">
        <h2 class="section-heading">Cross-References</h2>
        <ul class="cross-ref-list">
          <li v-for="(ref, i) in crossRefs" :key="i">
            <router-link
              v-if="ref.book && ref.chapter && ref.verse"
              :to="`/study/${ref.book}/${ref.chapter}/${ref.verse}`"
              class="ref-link"
            >
              {{ ref.label || `${ref.book} ${ref.chapter}:${ref.verse}` }}
            </router-link>
            <span v-else>{{ ref.label || ref }}</span>
            <span v-if="ref.note" class="ref-note"> -- {{ ref.note }}</span>
          </li>
        </ul>
      </section>

      <section v-if="verse && (depth === 'deep' || depth === 'scholarly')" class="study-section feedback-section">
        <h2 class="section-heading">Was this helpful?</h2>
        <p class="feedback-prompt">Help improve ABBA's verse-concept mapping</p>
        <div class="feedback-buttons">
          <button class="feedback-btn" :class="{ active: feedbackGiven === 'relevant' }" @click="sendFeedback('relevant')">Relevant</button>
          <button class="feedback-btn" :class="{ active: feedbackGiven === 'partial' }" @click="sendFeedback('partial')">Partially</button>
          <button class="feedback-btn" :class="{ active: feedbackGiven === 'irrelevant' }" @click="sendFeedback('irrelevant')">Not relevant</button>
        </div>
        <span v-if="feedbackGiven" class="feedback-thanks">Thanks for your feedback</span>
      </section>

      <NotesPanel
        v-if="verse"
        :bookId="book"
        :chapter="chapter"
        :verse="verse"
      />
    </template>

    <p v-else class="empty-state">No verse selected.</p>

    <div v-if="showCollectionPicker" class="modal-overlay" @click.self="showCollectionPicker = false">
      <div class="modal-box">
        <h3 class="modal-title">Save to Collection</h3>
        <div v-if="collections.length" class="collection-options">
          <button
            v-for="col in collections"
            :key="col.id"
            class="collection-option"
            @click="saveToCollection(col.id)"
          >{{ col.name }}</button>
        </div>
        <p v-else class="status-msg">No collections yet. Create one from the Collections page.</p>
        <button class="modal-close" @click="showCollectionPicker = false">Cancel</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { useApi } from '../composables/useApi'
import { useContextStore } from '../stores/context'
import type { VerseData, ContextData, CrossReference, AudioResource, CollectionInfo } from '../types/api'
import TranslationLens from '../components/TranslationLens.vue'
import WordJourneyCard from '../components/WordJourneyCard.vue'
import LiteraryModeIndicator from '../components/LiteraryModeIndicator.vue'
import SyntaxTreeView from '../components/SyntaxTreeView.vue'
import ManuscriptVariants from '../components/ManuscriptVariants.vue'
import DiscourseView from '../components/DiscourseView.vue'
import SemanticDomainBadge from '../components/SemanticDomainBadge.vue'
import NotesPanel from '../components/NotesPanel.vue'
import AudioPlayer from '../components/AudioPlayer.vue'

interface WordDetail {
  original?: string
  transliteration?: string
  gloss?: string
  strongs?: string
  morphology?: string
  semantic_domain?: string
  occurrences?: number
}

const props = defineProps<{
  depth?: string
}>()

const route = useRoute()
const api = useApi()
const contextStore = useContextStore()

const verseData = ref<VerseData | null>(null)
const contextData = ref<ContextData | null>(null)
const crossRefs = ref<CrossReference[]>([])
const selectedWord = ref<WordDetail | null>(null)
const shareUrl = ref<string | null>(null)
const showCollectionPicker = ref<boolean>(false)
const collections = ref<CollectionInfo[]>([])
const audioData = ref<AudioResource | null>(null)
const feedbackGiven = ref<string | null>(null)

const depth = computed<string>(() => props.depth ?? 'basic')
const book = computed<string>(() => (route.params.book as string) || '')
const chapter = computed<string>(() => (route.params.chapter as string) || '')
const verse = computed<string>(() => (route.params.verse as string) || '')

async function loadVerse(): Promise<void> {
  if (!book.value || !chapter.value) return

  verseData.value = null
  contextData.value = null
  crossRefs.value = []
  selectedWord.value = null
  shareUrl.value = null
  audioData.value = null
  feedbackGiven.value = null
  contextStore.clear()

  if (verse.value) {
    const result = await api.getVerse(book.value, chapter.value, verse.value, depth.value)
    if (result) verseData.value = result
  } else {
    const result = await api.getChapter(book.value, chapter.value, depth.value)
    if (result) verseData.value = result as unknown as VerseData
  }

  if (verse.value && depth.value !== 'basic') {
    const fetches: Promise<unknown>[] = [
      api.getContext(book.value, chapter.value, verse.value),
      api.getCrossReferences(book.value, chapter.value, verse.value),
    ]

    // At deep/scholarly depth, fetch scholarly data independently
    const isScholarly = depth.value === 'deep' || depth.value === 'scholarly'
    if (isScholarly) {
      fetches.push(
        api.getSyntaxTree(book.value, chapter.value, verse.value),
        api.getDiscourseUnits(book.value, chapter.value, verse.value),
        api.getManuscriptVariants(book.value, chapter.value, verse.value),
      )
    }

    const results = await Promise.all(fetches)
    const [ctx, refs] = results as [ContextData | null, { references?: CrossReference[] } | null]
    if (ctx) contextData.value = ctx
    if (refs) crossRefs.value = (refs as Record<string, unknown>).references as CrossReference[] || refs as unknown as CrossReference[] || []

    // Merge scholarly data into verseData so existing v-if guards render them
    if (isScholarly && verseData.value) {
      const [, , syntaxResult, discourseResult, variantsResult] = results as [unknown, unknown, Record<string, unknown> | null, Record<string, unknown> | null, Record<string, unknown> | null]
      if (syntaxResult && !verseData.value.syntax_tree) {
        verseData.value.syntax_tree = syntaxResult as unknown as VerseData['syntax_tree']
      }
      if (discourseResult && !verseData.value.discourse_units) {
        verseData.value.discourse_units = ((discourseResult as Record<string, unknown>).units || discourseResult || []) as VerseData['discourse_units']
      }
      if (variantsResult && !verseData.value.manuscript_variants) {
        verseData.value.manuscript_variants = ((variantsResult as Record<string, unknown>).variants || variantsResult || []) as VerseData['manuscript_variants']
      }
    }

    // Push context data to the shared store for the sidebar
    contextStore.setContext({
      cultural: (contextData.value?.cultural as Array<{title?: string; text?: string}> || []).map(c =>
        typeof c === 'object' && c !== null ? c : { text: String(c) }
      ),
      crossRefs: crossRefs.value,
      literary: verseData.value?.literary_structures || [],
      reference: `${book.value} ${chapter.value}:${verse.value}`,
    })
  }
}

function onWordClick(word: string | Record<string, unknown>, flags: Record<string, unknown>): void {
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

async function doExport(format: string): Promise<void> {
  if (!verse.value) return
  const data = await api.exportVerse('engbsb', book.value, chapter.value, verse.value, format)
  if (data) {
    const text = typeof data === 'string' ? data : JSON.stringify(data, null, 2)
    const blob = new Blob([text], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${book.value}_${chapter.value}_${verse.value}.${format === 'markdown' ? 'md' : 'json'}`
    a.click()
    URL.revokeObjectURL(url)
  }
}

async function doShare(): Promise<void> {
  if (!verse.value) return
  const refStr = `${book.value} ${chapter.value}:${verse.value}`
  const data = await api.createShare('verse', refStr, { book: book.value, chapter: chapter.value, verse: verse.value })
  if (data && data.token) {
    shareUrl.value = `${window.location.origin}/shared/${data.token}`
  }
}

async function loadAudio(): Promise<void> {
  if (audioData.value) {
    audioData.value = null
    return
  }
  const data = await api.getAudioResource(book.value, chapter.value)
  if (data) audioData.value = data
}

async function sendFeedback(feedbackType: string): Promise<void> {
  if (!verse.value || feedbackGiven.value) return
  const verseId = `${book.value}.${chapter.value}.${verse.value}`
  const conceptName = verseData.value?.primary_concept || 'general'
  await api.submitConceptFeedback(conceptName, verseId, feedbackType)
  feedbackGiven.value = feedbackType
}

async function addToCollection(): Promise<void> {
  const data = await api.getCollections()
  collections.value = (data as unknown as { collections?: CollectionInfo[] })?.collections || data || []
  showCollectionPicker.value = true
}

async function saveToCollection(collectionId: string): Promise<void> {
  await api.addToCollection(collectionId, book.value, chapter.value, verse.value)
  showCollectionPicker.value = false
}

onMounted(loadVerse)

watch(
  () => [route.params.book, route.params.chapter, route.params.verse, depth.value],
  loadVerse
)
</script>

<style scoped>
.study-header {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 0.5rem;
  flex-wrap: wrap;
}

.study-ref {
  font-family: var(--font-ui);
  font-size: 1.5rem;
}

.study-actions {
  display: flex;
  gap: 0.4rem;
  margin-bottom: 1.25rem;
  flex-wrap: wrap;
}

.action-btn {
  padding: 0.25rem 0.6rem;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  background: var(--color-surface);
  color: var(--color-text);
  font-family: var(--font-ui);
  font-size: 0.8rem;
  cursor: pointer;
  text-decoration: none;
}

.action-btn:hover {
  border-color: var(--color-accent);
  color: var(--color-accent);
}

.action-link {
  display: inline-flex;
  align-items: center;
}

.share-banner {
  font-family: var(--font-ui);
  font-size: 0.85rem;
  background: rgba(74, 111, 165, 0.08);
  padding: 0.5rem 0.75rem;
  border-radius: 6px;
  margin-bottom: 1rem;
}

.share-banner a {
  color: var(--color-accent);
}

.study-text {
  font-size: 1.15rem;
  line-height: 1.9;
  margin-bottom: 1.5rem;
}

.verse-text {
  font-family: var(--font-reading);
}

.word-card-inline {
  margin-bottom: 1.5rem;
}

.study-section {
  margin-bottom: 1.5rem;
  padding-top: 1rem;
  border-top: 1px solid var(--color-border);
}

.section-heading {
  font-family: var(--font-ui);
  font-size: 1rem;
  margin-bottom: 0.75rem;
  color: var(--color-accent);
}

.sub-heading {
  font-family: var(--font-ui);
  font-size: 0.85rem;
  font-weight: 600;
  margin-bottom: 0.35rem;
  margin-top: 0.5rem;
}

.context-text {
  font-size: 0.9rem;
  line-height: 1.6;
  margin-bottom: 0.5rem;
}

.translation-row {
  display: flex;
  gap: 0.75rem;
  padding: 0.35rem 0;
  font-size: 0.9rem;
  border-bottom: 1px solid rgba(0, 0, 0, 0.04);
}

.trans-name {
  min-width: 50px;
  font-family: var(--font-ui);
  font-weight: 600;
  font-size: 0.8rem;
  opacity: 0.5;
}

.cross-ref-list {
  list-style: none;
}

.cross-ref-list li {
  padding: 0.25rem 0;
  font-size: 0.9rem;
}

.ref-link {
  color: var(--color-accent);
  text-decoration: none;
}

.ref-link:hover {
  text-decoration: underline;
}

.ref-note {
  opacity: 0.6;
  font-size: 0.85rem;
}

.feedback-section { text-align: center; }
.feedback-prompt { font-size: 0.85rem; opacity: 0.6; margin-bottom: 0.5rem; }
.feedback-buttons { display: flex; gap: 0.5rem; justify-content: center; }
.feedback-btn {
  padding: 0.3rem 0.75rem;
  border: 1px solid var(--color-border);
  border-radius: 16px;
  background: none;
  font-family: var(--font-ui);
  font-size: 0.8rem;
  cursor: pointer;
  color: var(--color-text);
}
.feedback-btn:hover { border-color: var(--color-accent); color: var(--color-accent); }
.feedback-btn.active { background: var(--color-accent); color: white; border-color: var(--color-accent); }
.feedback-thanks { display: block; margin-top: 0.5rem; font-size: 0.8rem; opacity: 0.6; font-family: var(--font-ui); }

.loading,
.error,
.empty-state {
  font-family: var(--font-ui);
  font-size: 0.9rem;
  opacity: 0.6;
  padding: 2rem 0;
}

.error {
  color: #c0392b;
  opacity: 1;
}

/* Collection picker modal */
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.4);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
}

.modal-box {
  background: var(--color-surface);
  border-radius: 8px;
  padding: 1.5rem;
  max-width: 360px;
  width: 90%;
}

.modal-title {
  font-family: var(--font-ui);
  font-size: 1rem;
  margin-bottom: 0.75rem;
}

.collection-options {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
  margin-bottom: 0.75rem;
}

.collection-option {
  padding: 0.5rem 0.75rem;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  background: none;
  text-align: left;
  font-family: var(--font-ui);
  font-size: 0.9rem;
  cursor: pointer;
  color: var(--color-text);
}

.collection-option:hover {
  border-color: var(--color-accent);
  color: var(--color-accent);
}

.modal-close {
  display: block;
  margin-top: 0.5rem;
  padding: 0.3rem 0.75rem;
  background: none;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  font-family: var(--font-ui);
  cursor: pointer;
  color: var(--color-text);
}

.status-msg {
  font-family: var(--font-ui);
  font-size: 0.85rem;
  opacity: 0.5;
  font-style: italic;
}
</style>
