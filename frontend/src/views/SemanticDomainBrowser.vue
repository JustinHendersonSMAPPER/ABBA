<template>
  <div class="domain-browser">
    <header class="browser-header">
      <h1 class="browser-title">Semantic Domains</h1>
      <p class="browser-subtitle">Explore how biblical words are organized by meaning (Louw-Nida)</p>
    </header>

    <nav v-if="breadcrumbs.length" class="breadcrumbs">
      <button class="crumb" @click="loadDomains(null); breadcrumbs = []">All Domains</button>
      <span v-for="(crumb, i) in breadcrumbs" :key="i">
        <span class="crumb-sep">/</span>
        <button class="crumb" @click="navigateTo(i)">{{ crumb.name }}</button>
      </span>
    </nav>

    <div v-if="api.loading.value" class="status-msg">Loading...</div>
    <div v-else-if="api.error.value" class="status-msg error">{{ api.error.value }}</div>

    <div v-if="domains.length" class="domain-grid">
      <div
        v-for="domain in domains"
        :key="domain.domain_code"
        class="domain-card"
        @click="selectDomain(domain)"
      >
        <span class="domain-code">{{ domain.domain_code }}</span>
        <h3 class="domain-name">{{ domain.domain_name || domain.name }}</h3>
        <p v-if="domain.description" class="domain-desc">{{ domain.description }}</p>
        <span v-if="domain.child_count" class="domain-children">{{ domain.child_count }} sub-domains</span>
      </div>
    </div>

    <div v-if="words.length" class="words-section">
      <h2 class="section-heading">Words in this domain</h2>
      <div class="word-grid">
        <router-link
          v-for="word in words"
          :key="word.strongs_number"
          :to="`/lexicon/${word.strongs_number}`"
          class="word-card"
        >
          <span class="word-original">{{ word.word || word.original || word.strongs_number }}</span>
          <span v-if="word.transliteration" class="word-translit">{{ word.transliteration }}</span>
          <span class="word-gloss">{{ word.gloss || word.short_definition || '' }}</span>
          <span class="word-strongs">{{ word.strongs_number }}</span>
        </router-link>
      </div>
    </div>

    <p v-if="!domains.length && !words.length && !api.loading.value" class="status-msg">
      No domains found.
    </p>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useApi } from '../composables/useApi'
import type { SemanticDomain, WordDomainResult } from '../types/api'

interface Breadcrumb {
  code: string
  name: string
}

const api = useApi()
const domains = ref<SemanticDomain[]>([])
const words = ref<WordDomainResult[]>([])
const breadcrumbs = ref<Breadcrumb[]>([])

onMounted(() => loadDomains(null))

async function loadDomains(parentCode: string | null): Promise<void> {
  words.value = []
  const data = await api.getSemanticDomains(parentCode)
  if (data) {
    domains.value = (data as unknown as { domains?: SemanticDomain[] }).domains || data || []
  } else {
    domains.value = []
  }
}

async function selectDomain(domain: SemanticDomain): Promise<void> {
  breadcrumbs.value.push({ code: domain.domain_code, name: domain.domain_name || domain.name || domain.domain_code })

  // Load sub-domains
  const subData = await api.getSemanticDomains(domain.domain_code)
  const subs: SemanticDomain[] = (subData as unknown as { domains?: SemanticDomain[] })?.domains || subData || []

  if (subs.length) {
    domains.value = subs
    words.value = []
  } else {
    // Leaf domain — load words using the dedicated endpoint
    domains.value = []
    const wordData = await api.getDomainWords(domain.domain_code)
    if (wordData) {
      words.value = (wordData as unknown as { words?: WordDomainResult[] }).words || wordData || []
    } else {
      words.value = []
    }
  }
}

function navigateTo(index: number): void {
  const crumb = breadcrumbs.value[index]
  breadcrumbs.value = breadcrumbs.value.slice(0, index + 1)
  loadDomains(crumb.code)
}
</script>

<style scoped>
.browser-header { margin-bottom: 1.5rem; }
.browser-title { font-family: var(--font-ui); font-size: 1.5rem; margin-bottom: 0.25rem; }
.browser-subtitle { font-size: 0.9rem; opacity: 0.6; }

.breadcrumbs {
  display: flex;
  align-items: center;
  gap: 0.15rem;
  margin-bottom: 1rem;
  flex-wrap: wrap;
}
.crumb {
  background: none;
  border: none;
  font-family: var(--font-ui);
  font-size: 0.85rem;
  color: var(--color-accent);
  cursor: pointer;
  padding: 0.15rem 0.3rem;
  border-radius: 4px;
}
.crumb:hover { background: rgba(74, 111, 165, 0.08); }
.crumb-sep { opacity: 0.3; font-size: 0.8rem; }

.domain-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 0.75rem;
  margin-bottom: 1.5rem;
}
.domain-card {
  padding: 1rem;
  border: 1px solid var(--color-border);
  border-radius: 8px;
  cursor: pointer;
  transition: border-color 0.15s;
}
.domain-card:hover { border-color: var(--color-accent); }
.domain-code { font-family: var(--font-ui); font-size: 0.7rem; opacity: 0.4; }
.domain-name { font-family: var(--font-ui); font-size: 0.95rem; margin: 0.15rem 0 0.25rem; }
.domain-desc { font-size: 0.8rem; opacity: 0.6; line-height: 1.4; }
.domain-children { font-size: 0.75rem; opacity: 0.4; font-family: var(--font-ui); }

.section-heading { font-family: var(--font-ui); font-size: 1rem; color: var(--color-accent); margin-bottom: 0.75rem; }
.word-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 0.5rem;
}
.word-card {
  display: flex;
  flex-direction: column;
  padding: 0.75rem;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  text-decoration: none;
  color: inherit;
  transition: border-color 0.15s;
}
.word-card:hover { border-color: var(--color-accent); }
.word-original { font-size: 1.2rem; font-weight: 700; }
.word-translit { font-size: 0.8rem; font-style: italic; opacity: 0.6; }
.word-gloss { font-size: 0.85rem; margin-top: 0.25rem; }
.word-strongs { font-family: var(--font-ui); font-size: 0.7rem; opacity: 0.4; margin-top: 0.25rem; }

.status-msg { font-family: var(--font-ui); font-size: 0.9rem; opacity: 0.6; padding: 1rem 0; }
.error { color: #c0392b; opacity: 1; }
</style>
