<template>
  <div class="lexicon-view">
    <header class="lexicon-header">
      <h1 class="lexicon-title">Word Study</h1>
      <div class="lookup-bar">
        <input
          v-model="strongsInput"
          type="text"
          placeholder="Enter Strong's number (e.g., H2617 or G26)"
          class="lookup-input"
          @keyup.enter="loadWord"
        />
        <button class="lookup-btn" @click="loadWord" :disabled="!strongsInput.trim()">Look Up</button>
      </div>
    </header>

    <div v-if="api.loading.value" class="status-msg">Loading...</div>
    <div v-else-if="api.error.value" class="status-msg error">{{ api.error.value }}</div>

    <div v-if="entry" class="word-entry">
      <div class="word-main">
        <span class="word-original">{{ entry.word || entry.original || '' }}</span>
        <span v-if="entry.transliteration" class="word-translit">{{ entry.transliteration }}</span>
        <span class="word-strongs">{{ entry.strongs_number || strongsNumber }}</span>
      </div>

      <div v-if="entry.pronunciation" class="word-pronunciation">
        {{ entry.pronunciation }}
      </div>

      <section class="entry-section" v-if="entry.short_definition || entry.gloss">
        <h2 class="section-label">Short Definition</h2>
        <p class="definition-text">{{ entry.short_definition || entry.gloss }}</p>
      </section>

      <section class="entry-section" v-if="entry.full_definition || entry.definition">
        <h2 class="section-label">Full Definition</h2>
        <p class="definition-text">{{ entry.full_definition || entry.definition }}</p>
      </section>

      <section class="entry-section" v-if="explanation">
        <h2 class="section-label">Why This Matters</h2>
        <p class="explanation-text">{{ explanation.explanation || explanation }}</p>
      </section>

      <section class="entry-section" v-if="entry.etymology">
        <h2 class="section-label">Etymology</h2>
        <p class="definition-text">{{ entry.etymology }}</p>
      </section>

      <section class="entry-section" v-if="domains && domains.length">
        <h2 class="section-label">Semantic Domains</h2>
        <div class="domain-list">
          <router-link v-for="d in domains" :key="d.domain_code" :to="`/domains`" class="domain-badge domain-link">
            {{ d.domain_name }}
          </router-link>
        </div>
      </section>

      <section class="entry-section" v-if="entry.occurrences || entry.frequency">
        <h2 class="section-label">Usage</h2>
        <p class="definition-text">
          Appears {{ entry.occurrences || entry.frequency }} times in the Bible.
        </p>
      </section>

      <div class="word-actions">
        <router-link
          :to="{ name: 'search', query: { q: strongsNumber, mode: 'strongs' } }"
          class="action-link"
        >Find all occurrences</router-link>
      </div>
    </div>

    <p v-else-if="!api.loading.value && strongsNumber" class="status-msg">
      No lexicon entry found for {{ strongsNumber }}.
    </p>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'
import { useRoute } from 'vue-router'
import { useApi } from '../composables/useApi'
import type { LexiconEntry, WordExplanation, SemanticDomain } from '../types/api'

const route = useRoute()
const api = useApi()

const strongsInput = ref('')
const strongsNumber = ref('')
const entry = ref<LexiconEntry | null>(null)
const explanation = ref<WordExplanation | null>(null)
const domains = ref<SemanticDomain[]>([])

onMounted(() => {
  if (route.params.strongs) {
    strongsInput.value = route.params.strongs as string
    strongsNumber.value = route.params.strongs as string
    loadWord()
  }
})

watch(() => route.params.strongs, (val) => {
  if (val && val !== strongsNumber.value) {
    strongsInput.value = val as string
    strongsNumber.value = val as string
    loadWord()
  }
})

async function loadWord() {
  if (!strongsInput.value.trim()) return
  strongsNumber.value = strongsInput.value.trim().toUpperCase()
  entry.value = null
  explanation.value = null
  domains.value = []

  const [lexData, explainData, domainData] = await Promise.all([
    api.getWordDetail(strongsNumber.value),
    api.getWordExplanation(strongsNumber.value),
    api.getWordDomains(strongsNumber.value),
  ])

  if (lexData) entry.value = lexData
  if (explainData) explanation.value = explainData
  if (domainData) {
    const domainResult = domainData as unknown as Record<string, unknown>
    domains.value = (domainResult.domains as SemanticDomain[]) || []
  }
}
</script>

<style scoped>
.lexicon-header { margin-bottom: 1.5rem; }
.lexicon-title { font-family: var(--font-ui); font-size: 1.5rem; margin-bottom: 0.75rem; }

.lookup-bar { display: flex; gap: 0.5rem; }
.lookup-input {
  flex: 1;
  padding: 0.5rem 0.75rem;
  border: 2px solid var(--color-border);
  border-radius: 6px;
  font-size: 1rem;
  font-family: var(--font-ui);
  background: var(--color-surface);
  color: var(--color-text);
}
.lookup-input:focus { outline: none; border-color: var(--color-accent); }
.lookup-btn {
  padding: 0.5rem 1.25rem;
  background: var(--color-accent);
  color: white;
  border: none;
  border-radius: 6px;
  font-family: var(--font-ui);
  font-weight: 600;
  cursor: pointer;
}
.lookup-btn:disabled { opacity: 0.5; cursor: not-allowed; }

.word-entry { margin-top: 1rem; }
.word-main { margin-bottom: 0.75rem; }
.word-original { font-size: 2rem; font-weight: 700; display: block; line-height: 1.3; }
.word-translit { font-size: 1rem; font-style: italic; opacity: 0.7; display: block; }
.word-strongs { font-family: var(--font-ui); font-size: 0.85rem; opacity: 0.5; }
.word-pronunciation { font-family: var(--font-ui); font-size: 0.9rem; opacity: 0.6; margin-bottom: 1rem; }

.entry-section { margin-bottom: 1.25rem; padding-top: 0.75rem; border-top: 1px solid var(--color-border); }
.section-label { font-family: var(--font-ui); font-size: 0.85rem; color: var(--color-accent); margin-bottom: 0.4rem; }
.definition-text { font-size: 0.95rem; line-height: 1.6; }
.explanation-text { font-size: 0.95rem; line-height: 1.6; font-style: italic; background: rgba(74, 111, 165, 0.05); padding: 0.75rem; border-radius: 6px; }

.domain-list { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.domain-badge {
  padding: 0.2rem 0.6rem;
  background: rgba(74, 111, 165, 0.1);
  border-radius: 12px;
  font-size: 0.8rem;
  font-family: var(--font-ui);
}

.word-actions { margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid var(--color-border); }
.action-link {
  font-family: var(--font-ui);
  color: var(--color-accent);
  text-decoration: none;
  font-weight: 600;
}
.action-link:hover { text-decoration: underline; }

.status-msg { font-family: var(--font-ui); font-size: 0.9rem; opacity: 0.6; padding: 1rem 0; }
.error { color: #c0392b; opacity: 1; }
.domain-link { text-decoration: none; color: inherit; }
.domain-link:hover { background: rgba(74, 111, 165, 0.2); }
</style>
