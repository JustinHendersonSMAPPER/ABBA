<template>
  <div class="search-results">
    <header class="search-header">
      <h1 class="search-title">Search</h1>
      <div class="search-controls">
        <select v-model="searchMode" class="mode-select">
          <option value="text">Text Search</option>
          <option value="semantic">Semantic Search</option>
          <option value="strongs">Strong's Number</option>
          <option value="multilingual">Multilingual</option>
        </select>
        <input
          v-model="query"
          type="text"
          :placeholder="placeholders[searchMode]"
          class="search-input"
          @keyup.enter="doSearch"
        />
        <button class="search-btn" @click="doSearch" :disabled="!query.trim()">Search</button>
      </div>
    </header>

    <div v-if="searchMode === 'multilingual'" class="multilingual-options">
      <select v-model="sourceLang" class="mode-select">
        <option value="en">English</option>
        <option value="he">Hebrew</option>
        <option value="el">Greek</option>
      </select>
      <input v-model="targetTranslations" type="text" placeholder="Target translations (comma-separated)" class="search-input" />
    </div>

    <LoadingState v-if="api.loading.value" label="Searching…" />
    <div v-else-if="api.error.value" class="status-msg error">{{ api.error.value }}</div>

    <div v-if="results && results.length" class="results-list">
      <p class="results-count">{{ results.length }} result{{ results.length !== 1 ? 's' : '' }}</p>
      <div v-for="(r, i) in results" :key="i" class="result-card">
        <router-link
          :to="resultLink(r)"
          class="result-ref"
        >{{ r.reference || r.book_name + ' ' + r.chapter + ':' + r.verse }}</router-link>
        <p class="result-text">{{ r.text || r.snippet || '' }}</p>
        <div v-if="r.score != null" class="relevance-row">
          <div class="relevance-bar-track">
            <div class="relevance-bar-fill" :style="{ width: Math.round(r.score * 100) + '%' }"></div>
          </div>
          <span class="relevance-label">{{ Math.round(r.score * 100) }}% match</span>
        </div>
        <span v-if="r.match_type" class="match-badge">{{ matchTypeLabel(r.match_type) }}</span>
        <span v-if="r.explanation" class="match-explain">{{ r.explanation }}</span>
      </div>
    </div>

    <p v-else-if="searched && !api.loading.value && searchMode !== 'semantic'" class="status-msg">
      No results found. Try different search terms.
    </p>
    <div v-else-if="searched && !api.loading.value && searchMode === 'semantic'" class="status-msg semantic-empty">
      <p>No conceptually related passages found.</p>
      <p class="semantic-hint">Semantic search works best with a natural-language theme — try <em>"comfort in suffering"</em> rather than exact words.</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useApi } from '../composables/useApi'
import type { SearchResult } from '../types/api'
import LoadingState from '../components/LoadingState.vue'

const route = useRoute()
const router = useRouter()
const api = useApi()

const query = ref<string>('')
const searchMode = ref<string>('text')
const results = ref<SearchResult[] | null>(null)
const searched = ref<boolean>(false)
const sourceLang = ref('en')
const targetTranslations = ref('')

const placeholders: Record<string, string> = {
  text: 'Search for words or phrases...',
  semantic: 'Describe what you are looking for...',
  strongs: 'Enter a Strong\'s number (e.g., H2617, G26)...',
  multilingual: 'Search across languages...',
}

onMounted(() => {
  if (route.query.q) query.value = route.query.q as string
  if (route.query.mode) searchMode.value = route.query.mode as string
  if (query.value) doSearch()
})

async function doSearch() {
  if (!query.value.trim()) return
  searched.value = true
  results.value = null

  router.replace({ query: { q: query.value, mode: searchMode.value } })

  let data = null
  if (searchMode.value === 'text') {
    data = await api.searchText(query.value)
  } else if (searchMode.value === 'semantic') {
    data = await api.semanticSearch(query.value)
  } else if (searchMode.value === 'strongs') {
    data = await api.searchStrongs(query.value.trim())
  } else if (searchMode.value === 'multilingual') {
    data = await api.multilingualSearch(query.value, sourceLang.value, targetTranslations.value || null)
  }

  if (data) {
    if (Array.isArray(data)) {
      results.value = data as SearchResult[]
    } else {
      results.value = (data as Record<string, unknown>).results as SearchResult[] || []
    }
  }
}

function resultLink(r: SearchResult) {
  const book = (r as Record<string, unknown>).book_id || (r as Record<string, unknown>).book || ''
  const ch = (r as Record<string, unknown>).chapter || 1
  const v = (r as Record<string, unknown>).verse || ''
  return v ? `/study/${book}/${ch}/${v}` : `/study/${book}/${ch}`
}

function matchTypeLabel(raw: string): string {
  switch (raw) {
    case 'exact': return 'Exact text'
    case 'semantic': return 'Semantic'
    case 'both': return 'Text + meaning'
    default: return raw
  }
}
</script>

<style scoped>
.search-header { margin-bottom: 1.5rem; }
.search-title { font-family: var(--font-ui); font-size: 1.5rem; margin-bottom: 0.75rem; }

.search-controls {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}
.mode-select {
  padding: 0.5rem;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  font-family: var(--font-ui);
  font-size: 0.9rem;
  background: var(--color-surface);
  color: var(--color-text);
}
.search-input {
  flex: 1;
  min-width: 200px;
  padding: 0.5rem 0.75rem;
  border: 2px solid var(--color-border);
  border-radius: 6px;
  font-size: 1rem;
  font-family: var(--font-ui);
  background: var(--color-surface);
  color: var(--color-text);
}
.search-input:focus { outline: none; border-color: var(--color-accent); }
.search-btn {
  padding: 0.5rem 1.25rem;
  background: var(--color-accent);
  color: white;
  border: none;
  border-radius: 6px;
  font-family: var(--font-ui);
  font-weight: 600;
  cursor: pointer;
}
.search-btn:disabled { opacity: 0.5; cursor: not-allowed; }

.results-count {
  font-family: var(--font-ui);
  font-size: 0.85rem;
  opacity: 0.6;
  margin-bottom: 0.75rem;
}
.result-card {
  padding: 0.75rem;
  margin-bottom: 0.5rem;
  border-radius: 6px;
  background: rgba(0, 0, 0, 0.02);
  border: 1px solid var(--color-border);
}
.result-ref {
  font-family: var(--font-ui);
  font-weight: 600;
  color: var(--color-accent);
  text-decoration: none;
}
.result-ref:hover { text-decoration: underline; }
.result-text {
  font-size: 0.9rem;
  margin-top: 0.25rem;
  line-height: 1.5;
}
.match-badge {
  display: inline-block;
  font-size: 0.7rem;
  background: var(--color-accent);
  color: white;
  padding: 0.1rem 0.4rem;
  border-radius: 10px;
  margin-top: 0.25rem;
}
.match-explain {
  display: block;
  font-size: 0.8rem;
  opacity: 0.6;
  margin-top: 0.2rem;
  font-style: italic;
}
.status-msg {
  font-family: var(--font-ui);
  font-size: 0.9rem;
  opacity: 0.6;
  padding: 1rem 0;
}
.error { color: #c0392b; opacity: 1; }
.multilingual-options { display: flex; gap: 0.5rem; margin-bottom: 1rem; flex-wrap: wrap; }

.relevance-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 0.35rem;
  margin-bottom: 0.15rem;
}

.relevance-bar-track {
  width: 80px;
  height: 5px;
  background: var(--color-border, #ddd);
  border-radius: 3px;
  overflow: hidden;
  flex-shrink: 0;
}

.relevance-bar-fill {
  height: 100%;
  background: var(--color-accent, #4a6fa5);
  border-radius: 3px;
  transition: width 0.2s;
}

.relevance-label {
  font-size: 0.72rem;
  font-family: var(--font-ui);
  opacity: 0.65;
  white-space: nowrap;
}

.semantic-empty {
  opacity: 1;
}

.semantic-hint {
  margin-top: 0.35rem;
  font-size: 0.85rem;
  opacity: 0.7;
  line-height: 1.5;
}
</style>
