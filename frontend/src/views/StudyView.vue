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
        <router-link
          v-if="verse"
          :to="{ name: 'compare', query: { book: book, chapter: chapter, verse: verse } }"
          class="action-btn action-link"
        >Compare</router-link>
      </div>

      <div v-if="shareUrl" class="share-banner">
        Shareable link: <a :href="shareUrl" target="_blank">{{ shareUrl }}</a>
      </div>

      <section class="study-text">
        <TranslationLens
          v-if="depth !== 'basic' && verseData.words"
          :words="verseData.words"
          :rich-flags="verseData.richness_flags || []"
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
            {{ item.text || item }}
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

<script setup>
import { ref, computed, watch, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { useApi } from '../composables/useApi.js'
import { useContextStore } from '../stores/context.js'
import TranslationLens from '../components/TranslationLens.vue'
import WordJourneyCard from '../components/WordJourneyCard.vue'
import LiteraryModeIndicator from '../components/LiteraryModeIndicator.vue'
import SyntaxTreeView from '../components/SyntaxTreeView.vue'
import ManuscriptVariants from '../components/ManuscriptVariants.vue'
import DiscourseView from '../components/DiscourseView.vue'
import SemanticDomainBadge from '../components/SemanticDomainBadge.vue'
import NotesPanel from '../components/NotesPanel.vue'

const props = defineProps({
  depth: { type: String, default: 'basic' },
})

const route = useRoute()
const api = useApi()
const contextStore = useContextStore()

const verseData = ref(null)
const contextData = ref(null)
const crossRefs = ref([])
const selectedWord = ref(null)
const shareUrl = ref(null)
const showCollectionPicker = ref(false)
const collections = ref([])

const book = computed(() => route.params.book || '')
const chapter = computed(() => route.params.chapter || '')
const verse = computed(() => route.params.verse || '')

async function loadVerse() {
  if (!book.value || !chapter.value) return

  verseData.value = null
  contextData.value = null
  crossRefs.value = []
  selectedWord.value = null
  shareUrl.value = null
  contextStore.clear()

  if (verse.value) {
    const result = await api.getVerse(book.value, chapter.value, verse.value, props.depth)
    if (result) verseData.value = result
  } else {
    const result = await api.getChapter(book.value, chapter.value, props.depth)
    if (result) verseData.value = result
  }

  if (verse.value && props.depth !== 'basic') {
    const [ctx, refs] = await Promise.all([
      api.getContext(book.value, chapter.value, verse.value),
      api.getCrossReferences(book.value, chapter.value, verse.value),
    ])
    if (ctx) contextData.value = ctx
    if (refs) crossRefs.value = refs.references || refs || []

    // Push context data to the shared store for the sidebar
    contextStore.setContext({
      cultural: contextData.value?.cultural || [],
      crossRefs: crossRefs.value,
      literary: verseData.value?.literary_structures || [],
      reference: `${book.value} ${chapter.value}:${verse.value}`,
    })
  }
}

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

async function doExport(format) {
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

async function doShare() {
  if (!verse.value) return
  const ref = `${book.value} ${chapter.value}:${verse.value}`
  const data = await api.createShare('verse', ref, { book: book.value, chapter: chapter.value, verse: verse.value })
  if (data && data.token) {
    shareUrl.value = `${window.location.origin}/share/${data.token}`
  }
}

async function addToCollection() {
  const data = await api.getCollections()
  collections.value = data?.collections || data || []
  showCollectionPicker.value = true
}

async function saveToCollection(collectionId) {
  await api.addToCollection(collectionId, book.value, chapter.value, verse.value)
  showCollectionPicker.value = false
}

onMounted(loadVerse)

watch(
  () => [route.params.book, route.params.chapter, route.params.verse, props.depth],
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
