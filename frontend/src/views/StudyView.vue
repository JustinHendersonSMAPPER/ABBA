<template>
  <div class="study-view">
    <LoadingState v-if="api.loading.value" label="Loading verse…" />
    <div v-else-if="api.error.value" class="error">{{ api.error.value }}</div>

    <template v-else-if="verseData">
      <header class="study-header">
        <h1 class="study-ref">
          {{ verseData.reference || (verseData.book_name + ' ' + verseData.chapter + ':' + verseData.verse) }}
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
          :rich-flags="(verseData.richness_flags || []).map(f => ({ richness: 0, explanation: typeof f === 'string' ? f : f.explanation }))"
          @word-click="onWordClick"
        />
        <p v-else class="verse-text">{{ verseData.text }}</p>
      </section>

      <!-- Original Language panel: shown whenever the API returned words -->
      <section
        v-if="verseData.words && verseData.words.length"
        class="study-section orig-lang-section"
        aria-label="Original Language"
      >
        <h2 class="section-heading orig-lang-heading">
          Original Language{{ origLangLabel ? ' (' + origLangLabel + ')' : '' }}
        </h2>
        <div class="orig-lang-chips" role="list">
          <div
            v-for="(w, i) in verseData.words"
            :key="i"
            class="orig-chip"
            role="listitem"
          >
            <span
              class="orig-chip__text"
              :dir="(w.language === 'hebrew' || w.language === 'aramaic') ? 'rtl' : 'ltr'"
            >{{ w.original_text || '' }}</span>
            <span v-if="w.transliteration" class="orig-chip__translit">{{ w.transliteration }}</span>
            <span v-if="w.english_gloss" class="orig-chip__gloss">{{ w.english_gloss }}</span>
            <router-link
              v-if="w.strongs_number"
              :to="`/lexicon/${w.strongs_number}`"
              class="orig-chip__strongs"
              :title="`Open lexicon entry for ${w.strongs_number}`"
            >{{ w.strongs_number }}</router-link>
          </div>
        </div>
      </section>

      <WordJourneyCard
        v-if="selectedWord"
        :detail="selectedWord"
        class="word-card-inline"
        @close="selectedWord = null"
      />

      <section v-if="depth !== 'basic' && verseData.parallel_passages && verseData.parallel_passages.length" class="study-section">
        <h2 class="section-heading">Parallel Passages</h2>
        <div v-for="(p, i) in verseData.parallel_passages" :key="i" class="translation-row">
          <span class="trans-name">{{ (p as Record<string, unknown>).reference || '' }}</span>
          <span class="trans-text">{{ (p as Record<string, unknown>).text || '' }}</span>
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

      <section v-if="verseData.cultural_context && verseData.cultural_context.length" class="study-section">
        <h2 class="section-heading">Context</h2>
        <div>
          <h3 class="sub-heading">Cultural Background</h3>
          <p v-for="(item, i) in verseData.cultural_context" :key="'c-' + i" class="context-text">
            {{ item.text }}
          </p>
        </div>
      </section>

      <section
        v-if="depth !== 'basic' && verseData.cross_references && verseData.cross_references.length"
        class="study-section xref-section"
      >
        <h2 class="section-heading">Cross-References</h2>
        <p class="xref-intro">Related passages — and why they connect.</p>
        <ul class="xref-cards">
          <li v-for="(ref, i) in verseData.cross_references" :key="i" class="xref-card">
            <div class="xref-card__head">
              <router-link
                v-if="ref.book_id && ref.chapter && ref.verse"
                :to="`/study/${ref.book_id}/${ref.chapter}/${ref.verse}`"
                class="xref-ref"
              >
                {{ ref.label || `${ref.book_name || ref.book_id} ${ref.chapter}:${ref.verse}` }}
                <span class="xref-arrow" aria-hidden="true">→</span>
              </router-link>
              <span v-else class="xref-ref">{{ ref.label }}</span>
              <ProvenanceChip
                v-if="ref.id != null"
                entity-type="cross_reference"
                :entity-id="ref.id"
              />
            </div>
            <p v-if="ref.text" class="xref-target-text">{{ ref.text }}</p>
            <p v-if="ref.note" class="xref-why">
              <span class="xref-why__label">Why linked</span>{{ ref.note }}
            </p>
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
import { useRoute, useRouter } from 'vue-router'
import { useApi } from '../composables/useApi'
import { useContextStore } from '../stores/context'
import { useTranslationStore } from '../stores/translation'
import type { VerseResponse, AudioResource, CollectionInfo, CulturalNote } from '../types/api'
import TranslationLens from '../components/TranslationLens.vue'
import WordJourneyCard from '../components/WordJourneyCard.vue'
import LiteraryModeIndicator from '../components/LiteraryModeIndicator.vue'
import SyntaxTreeView from '../components/SyntaxTreeView.vue'
import ManuscriptVariants from '../components/ManuscriptVariants.vue'
import DiscourseView from '../components/DiscourseView.vue'
import SemanticDomainBadge from '../components/SemanticDomainBadge.vue'
import NotesPanel from '../components/NotesPanel.vue'
import AudioPlayer from '../components/AudioPlayer.vue'
import LoadingState from '../components/LoadingState.vue'
import ProvenanceChip from '../components/ProvenanceChip.vue'

interface WordDetail {
  original_text?: string
  transliteration?: string
  english_gloss?: string
  strongs_number?: string
  morphology_code?: string
  morphology_description?: string
  part_of_speech?: string
  language?: string
}

const props = defineProps<{
  depth?: string
}>()

const route = useRoute()
const router = useRouter()
const api = useApi()
const contextStore = useContextStore()
const translationStore = useTranslationStore()

const verseData = ref<VerseResponse | null>(null)
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

/** Derive a human-readable language label from the first word that has one */
const origLangLabel = computed<string>(() => {
  const lang = verseData.value?.words?.find(w => w.language)?.language
  if (!lang) return ''
  return lang.charAt(0).toUpperCase() + lang.slice(1)
})

async function loadVerse(): Promise<void> {
  if (!book.value || !chapter.value) return

  // Redirect /study/:book/:chapter → /study/:book/:chapter/1 so we always have a verse
  if (!verse.value) {
    router.replace(`/study/${book.value}/${chapter.value}/1`)
    return
  }

  verseData.value = null
  selectedWord.value = null
  shareUrl.value = null
  audioData.value = null
  feedbackGiven.value = null
  contextStore.clear()

  const bookIdNum = Number(book.value)
  // Always fetch at least 'standard' depth so original-language words are always included.
  // If the user has chosen a richer depth (deep/scholarly), honour that instead.
  const fetchDepth = depth.value === 'basic' ? 'standard' : depth.value
  const result = await api.getVerse(translationStore.current, bookIdNum, chapter.value, verse.value, fetchDepth)
  if (result) verseData.value = result

  if (verse.value && depth.value !== 'basic') {
    const bookIdStr = book.value
    const isScholarly = depth.value === 'deep' || depth.value === 'scholarly'

    if (isScholarly) {
      const [syntaxResult, discourseResult, variantsResult] = await Promise.all([
        api.getSyntaxTree(bookIdStr, chapter.value, verse.value),
        api.getDiscourseUnits(bookIdStr, chapter.value, verse.value),
        api.getManuscriptVariants(bookIdStr, chapter.value, verse.value),
      ])

      if (verseData.value) {
        if (syntaxResult && !verseData.value.syntax_tree) {
          verseData.value.syntax_tree = syntaxResult as VerseResponse['syntax_tree']
        }
        if (discourseResult && !verseData.value.discourse_units) {
          verseData.value.discourse_units = ((discourseResult as Record<string, unknown>).units || discourseResult || []) as VerseResponse['discourse_units']
        }
        if (variantsResult && !verseData.value.manuscript_variants) {
          verseData.value.manuscript_variants = ((variantsResult as Record<string, unknown>).variants || variantsResult || []) as VerseResponse['manuscript_variants']
        }
      }
    }

    // Push embedded context data to the shared store for the sidebar
    contextStore.setContext({
      cultural: (verseData.value?.cultural_context || []).map((c: CulturalNote) =>
        (c as unknown as Record<string, unknown>)
      ),
      crossRefs: (verseData.value?.cross_references || []).map(r => r as Record<string, unknown>),
      literary: verseData.value?.literary_structures || [],
      reference: `${book.value} ${chapter.value}:${verse.value}`,
    })
  }
}

function onWordClick(word: string | Record<string, unknown>, _flags: Record<string, unknown>): void {
  const w = typeof word === 'string' ? {} : word as Record<string, unknown>
  const strongs = (w.strongs_number as string | undefined)
  if (strongs || w.original_text) {
    selectedWord.value = {
      original_text: (w.original_text as string | undefined),
      transliteration: (w.transliteration as string | undefined),
      english_gloss: (w.english_gloss as string | undefined),
      strongs_number: strongs,
      morphology_code: (w.morphology_code as string | undefined),
      morphology_description: (w.morphology_description as string | undefined),
      part_of_speech: (w.part_of_speech as string | undefined),
      language: (w.language as string | undefined),
    }
  }
}

async function doExport(format: string): Promise<void> {
  if (!verse.value) return
  const data = await api.exportVerse(api.DEFAULT_TRANSLATION, book.value, chapter.value, verse.value, format)
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
  const firstConcept = verseData.value?.concepts?.[0]
  const conceptName = (firstConcept && typeof firstConcept === 'object' ? (firstConcept as Record<string, unknown>).name as string : null) || 'general'
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

/* ── Cross-reference cards ───────────────────────────────── */
.xref-intro {
  font-family: var(--font-ui);
  font-size: 0.82rem;
  opacity: 0.6;
  margin: -0.4rem 0 0.85rem;
}

.xref-cards {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}

.xref-card {
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 0.7rem 0.85rem;
  background: var(--color-surface);
  transition: border-color 0.15s ease;
}

.xref-card:hover {
  border-color: var(--color-accent);
}

.xref-card__head {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.35rem;
}

.xref-ref {
  color: var(--color-accent);
  text-decoration: none;
  font-family: var(--font-ui);
  font-weight: 600;
  font-size: 0.92rem;
}

.xref-ref:hover {
  text-decoration: underline;
}

.xref-arrow {
  opacity: 0.5;
}

.xref-target-text {
  font-family: var(--font-reading, Georgia, serif);
  font-size: 0.95rem;
  line-height: 1.55;
  margin: 0.45rem 0 0;
  padding-left: 0.7rem;
  border-left: 2px solid var(--color-border);
  opacity: 0.9;
}

.xref-why {
  font-size: 0.85rem;
  line-height: 1.5;
  margin: 0.5rem 0 0;
  padding-top: 0.45rem;
  border-top: 1px dashed var(--color-border);
}

.xref-why__label {
  font-family: var(--font-ui);
  font-weight: 600;
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  opacity: 0.55;
  margin-right: 0.45rem;
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

/* ── Original Language panel ─────────────────────────────── */
.orig-lang-section {
  /* inherits .study-section padding/border */
}

.orig-lang-heading {
  /* inherits .section-heading colour */
}

.orig-lang-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.orig-chip {
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  gap: 0.15rem;
  padding: 0.45rem 0.65rem;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  background: var(--color-surface);
  min-width: 3.5rem;
  text-align: center;
  transition: border-color 0.15s ease;
}

.orig-chip:hover {
  border-color: var(--color-accent);
}

/* Primary original-language word — slightly larger, serif */
.orig-chip__text {
  font-family: var(--font-reading, Georgia, serif);
  font-size: 1.15rem;
  line-height: 1.3;
  color: var(--color-text);
  font-weight: 500;
}

/* Transliteration — muted, small */
.orig-chip__translit {
  font-family: var(--font-ui);
  font-size: 0.7rem;
  opacity: 0.55;
  letter-spacing: 0.02em;
}

/* English gloss — small */
.orig-chip__gloss {
  font-family: var(--font-ui);
  font-size: 0.72rem;
  opacity: 0.75;
  font-style: italic;
}

/* Strong's number — accent link */
.orig-chip__strongs {
  font-family: var(--font-ui);
  font-size: 0.68rem;
  color: var(--color-accent);
  text-decoration: none;
  margin-top: 0.1rem;
}

.orig-chip__strongs:hover {
  text-decoration: underline;
}
</style>
