<template>
  <div class="concept-explorer">
    <header class="explorer-header">
      <h1 class="explorer-title">Concept Explorer</h1>
      <p class="explorer-subtitle">Discover biblical concepts from everyday questions</p>
    </header>

    <div class="search-box">
      <input
        v-model="query"
        type="text"
        placeholder="What does the Bible say about... (e.g., anxiety, forgiveness, money)"
        class="search-input"
        @keyup.enter="search"
      />
      <button class="search-btn" @click="search" :disabled="!query.trim()">Discover</button>
    </div>

    <LoadingState v-if="api.loading.value" label="Searching…" />
    <div v-else-if="api.error.value" class="error">{{ api.error.value }}</div>

    <template v-if="result">
      <section v-if="result.matched_concepts && result.matched_concepts.length" class="result-section">
        <h2 class="section-heading">Biblical Concepts</h2>
        <div v-for="concept in result.matched_concepts" :key="concept.name" class="concept-card">
          <div class="concept-link" @click="selectConcept(concept.name)">
            <span class="concept-name">{{ concept.name }}</span>
            <span v-if="concept.verse_count" class="verse-badge">{{ concept.verse_count }} verses</span>
          </div>
          <p v-if="concept.description" class="concept-desc">{{ concept.description }}</p>
          <div class="concept-feedback">
            <button class="fb-btn" @click="giveFeedback(concept.name, 'relevant')">Relevant</button>
            <button class="fb-btn" @click="giveFeedback(concept.name, 'irrelevant')">Not relevant</button>
          </div>
        </div>
      </section>

      <ConceptGraphView v-if="graphData && graphData.nodes" :graph="graphData" />

      <section v-if="result.matched_life_topics && result.matched_life_topics.length" class="result-section">
        <h2 class="section-heading">Life Topics</h2>
        <div v-for="topic in result.matched_life_topics" :key="topic.slug || topic.id" class="topic-card">
          <router-link :to="`/topics/${encodeURIComponent(topic.slug || topic.id)}`" class="topic-link">
            <span v-if="topic.icon" class="topic-icon">{{ topic.icon }}</span>
            <span class="topic-name">{{ topic.name }}</span>
            <span v-if="topic.category" class="topic-category">{{ topic.category }}</span>
          </router-link>
          <p v-if="topic.description" class="topic-desc">{{ topic.description }}</p>
        </div>
      </section>

      <section v-if="result.suggested_searches && result.suggested_searches.length" class="result-section">
        <h2 class="section-heading">Related Searches</h2>
        <div class="suggestion-chips">
          <button
            v-for="(s, i) in result.suggested_searches"
            :key="i"
            class="suggestion-chip"
            @click="query = s.replace('topic: ', '').replace('words in domain: ', ''); search()"
          >
            {{ s }}
          </button>
        </div>
      </section>

      <p v-if="!result.matched_concepts?.length && !result.matched_life_topics?.length" class="no-results">
        No matching concepts found. Try different words or a broader question.
      </p>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useApi } from '../composables/useApi'
import type { ConceptDiscoveryResult, ConceptGraph } from '../types/api'
import ConceptGraphView from '../components/ConceptGraphView.vue'
import LoadingState from '../components/LoadingState.vue'

const api = useApi()
const query = ref<string>('')
const result = ref<ConceptDiscoveryResult | null>(null)
const graphData = ref<ConceptGraph | null>(null)

async function search() {
  if (!query.value.trim()) return
  graphData.value = null
  result.value = await api.discoverConcepts(query.value)
  if (result.value?.matched_concepts?.length && result.value.matched_concepts[0]?.name) {
    selectConcept(result.value.matched_concepts[0].name)
  }
}

async function selectConcept(name: string | undefined) {
  if (!name) return
  const graph = await api.getConceptGraph(name, 1)
  if (graph) {
    graphData.value = {
      ...graph,
      relationships: (graph as unknown as Record<string, unknown>).relationships as Array<{source_concept: string; target_concept: string; relationship_type: string; weight?: number}> || []
    }
  }
}

async function giveFeedback(conceptName: string | undefined, feedbackType: string): Promise<void> {
  if (!conceptName) return
  await api.submitConceptFeedback(conceptName, 'discovery', feedbackType)
}
</script>

<style scoped>
.explorer-header { margin-bottom: 1.5rem; }
.explorer-title { font-family: var(--font-ui); font-size: 1.5rem; margin-bottom: 0.25rem; }
.explorer-subtitle { font-size: 0.9rem; opacity: 0.6; }

.search-box {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1.5rem;
}
.search-input {
  flex: 1;
  padding: 0.75rem 1rem;
  font-size: 1rem;
  border: 2px solid var(--color-border);
  border-radius: 8px;
  font-family: var(--font-ui);
}
.search-input:focus { outline: none; border-color: var(--color-accent); }
.search-btn {
  padding: 0.75rem 1.5rem;
  background: var(--color-accent);
  color: white;
  border: none;
  border-radius: 8px;
  font-family: var(--font-ui);
  font-weight: 600;
  cursor: pointer;
}
.search-btn:disabled { opacity: 0.5; cursor: not-allowed; }

.result-section {
  margin-bottom: 1.5rem;
  padding-top: 1rem;
  border-top: 1px solid var(--color-border);
}
.section-heading {
  font-family: var(--font-ui);
  font-size: 1rem;
  color: var(--color-accent);
  margin-bottom: 0.75rem;
}

.concept-card, .topic-card {
  padding: 0.75rem;
  margin-bottom: 0.5rem;
  border-radius: 6px;
  background: rgba(0,0,0,0.02);
}
.concept-link {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
  color: inherit;
}
.concept-link:hover .concept-name { color: var(--color-accent); }
.topic-link {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  text-decoration: none;
  color: inherit;
}
.concept-name, .topic-name { font-weight: 600; }
.verse-badge {
  font-size: 0.75rem;
  background: var(--color-accent);
  color: white;
  padding: 0.15rem 0.5rem;
  border-radius: 12px;
}
.topic-category { font-size: 0.8rem; opacity: 0.5; }
.concept-desc, .topic-desc { font-size: 0.85rem; opacity: 0.7; margin-top: 0.25rem; }

.suggestion-chips { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.suggestion-chip {
  padding: 0.4rem 0.75rem;
  background: rgba(0,0,0,0.06);
  border: 1px solid var(--color-border);
  border-radius: 16px;
  font-size: 0.85rem;
  cursor: pointer;
  font-family: var(--font-ui);
}
.suggestion-chip:hover { background: rgba(0,0,0,0.1); }

.no-results, .loading, .error { font-family: var(--font-ui); font-size: 0.9rem; opacity: 0.6; padding: 1rem 0; }
.error { color: #c0392b; opacity: 1; }
.concept-feedback { display: flex; gap: 0.3rem; margin-top: 0.35rem; }
.fb-btn { padding: 0.15rem 0.5rem; border: 1px solid var(--color-border); border-radius: 12px; background: none; font-family: var(--font-ui); font-size: 0.7rem; cursor: pointer; color: var(--color-text); opacity: 0.6; }
.fb-btn:hover { opacity: 1; border-color: var(--color-accent); }
</style>
