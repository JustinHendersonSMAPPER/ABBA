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
    </template>

    <p v-else class="empty-state">No verse selected.</p>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { useApi } from '../composables/useApi.js'
import TranslationLens from '../components/TranslationLens.vue'
import WordJourneyCard from '../components/WordJourneyCard.vue'
import LiteraryModeIndicator from '../components/LiteraryModeIndicator.vue'
import SyntaxTreeView from '../components/SyntaxTreeView.vue'
import ManuscriptVariants from '../components/ManuscriptVariants.vue'
import DiscourseView from '../components/DiscourseView.vue'
import SemanticDomainBadge from '../components/SemanticDomainBadge.vue'

const props = defineProps({
  depth: { type: String, default: 'basic' },
})

const route = useRoute()
const api = useApi()

const verseData = ref(null)
const contextData = ref(null)
const crossRefs = ref([])
const selectedWord = ref(null)

const book = computed(() => route.params.book || '')
const chapter = computed(() => route.params.chapter || '')
const verse = computed(() => route.params.verse || '')

async function loadVerse() {
  if (!book.value || !chapter.value) return

  verseData.value = null
  contextData.value = null
  crossRefs.value = []
  selectedWord.value = null

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
  margin-bottom: 1.25rem;
  flex-wrap: wrap;
}

.study-ref {
  font-family: var(--font-ui);
  font-size: 1.5rem;
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
</style>
