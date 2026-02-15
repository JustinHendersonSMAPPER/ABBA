<template>
  <div class="topic-navigator">
    <h1 class="page-title">Life Topics</h1>
    <p class="page-subtitle">Start with what matters to you, and discover what Scripture says.</p>

    <div class="search-bar">
      <input
        v-model="searchQuery"
        type="text"
        class="search-input"
        placeholder="Search topics (e.g., anxiety, forgiveness, purpose)..."
      />
    </div>

    <div v-if="api.loading.value" class="loading">Loading topics...</div>
    <div v-else-if="api.error.value" class="error">{{ api.error.value }}</div>

    <div v-else-if="filteredTopics.length" class="topic-grid">
      <div
        v-for="topic in filteredTopics"
        :key="topic.id"
        class="topic-card"
        @click="selectTopic(topic)"
      >
        <h3 class="topic-name">{{ topic.name }}</h3>
        <p v-if="topic.description" class="topic-desc">{{ topic.description }}</p>
        <span v-if="topic.verse_count" class="topic-count">
          {{ topic.verse_count }} passages
        </span>
      </div>
    </div>

    <p v-else-if="searchQuery" class="empty-state">
      No topics match "{{ searchQuery }}". Try different words.
    </p>

    <div v-if="selectedTopic" class="topic-detail">
      <h2>{{ selectedTopic.name }}</h2>
      <p v-if="selectedTopic.description">{{ selectedTopic.description }}</p>

      <div v-if="topicDetail" class="study-steps">
        <div v-for="(step, i) in topicDetail.steps || []" :key="i" class="study-step">
          <span class="step-num">{{ i + 1 }}</span>
          <div class="step-content">
            <h4>{{ step.title }}</h4>
            <p>{{ step.summary }}</p>
            <router-link
              v-if="step.book && step.chapter && step.verse"
              :to="`/study/${step.book}/${step.chapter}/${step.verse}`"
              class="step-link"
            >
              Read {{ step.reference || `${step.book} ${step.chapter}:${step.verse}` }} &rarr;
            </router-link>
          </div>
        </div>
      </div>

      <button class="back-btn" @click="selectedTopic = null; topicDetail = null">
        &larr; Back to topics
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useApi } from '../composables/useApi'
import type { TopicSummary, TopicDetail } from '../types/api'

const api = useApi()

const topics = ref<TopicSummary[]>([])
const searchQuery = ref<string>('')
const selectedTopic = ref<TopicSummary | null>(null)
const topicDetail = ref<TopicDetail | null>(null)
const searchTimeout = ref<ReturnType<typeof setTimeout> | null>(null)
const apiSearchResults = ref<TopicSummary[] | null>(null)

onMounted(async () => {
  const result = await api.getTopics()
  if (result) {
    if (Array.isArray(result)) {
      topics.value = result as TopicSummary[]
    } else {
      topics.value = (result as Record<string, unknown>).topics as TopicSummary[] || []
    }
  }
})

watch(() => searchQuery.value, (q) => {
  if (searchTimeout.value) clearTimeout(searchTimeout.value)
  apiSearchResults.value = null
  if (q.trim().length >= 2) {
    searchTimeout.value = setTimeout(async () => {
      const results = await api.searchTopics(q)
      if (results) apiSearchResults.value = results as TopicSummary[]
    }, 300)
  }
})

const filteredTopics = computed(() => {
  if (selectedTopic.value) return []
  if (apiSearchResults.value) return apiSearchResults.value
  if (!searchQuery.value) return topics.value
  const q = searchQuery.value.toLowerCase()
  return topics.value.filter(
    (t) =>
      t.name.toLowerCase().includes(q) ||
      (t.description && t.description.toLowerCase().includes(q))
  )
})

async function selectTopic(topic: TopicSummary) {
  selectedTopic.value = topic
  const result = await api.getTopic(topic.id)
  if (result) {
    topicDetail.value = result
  }
}
</script>

<style scoped>
.page-title {
  font-family: var(--font-ui);
  font-size: 1.5rem;
  margin-bottom: 0.25rem;
}

.page-subtitle {
  font-size: 0.95rem;
  opacity: 0.6;
  margin-bottom: 1.5rem;
}

.search-bar {
  margin-bottom: 1.5rem;
}

.search-input {
  width: 100%;
  padding: 0.65rem 1rem;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  font-size: 0.95rem;
  font-family: var(--font-ui);
  background: white;
  color: var(--color-text);
}

.search-input:focus {
  outline: none;
  border-color: var(--color-accent);
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

.topic-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 1rem;
}

.topic-card {
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 1rem;
  cursor: pointer;
  transition: border-color 0.15s, box-shadow 0.15s;
  background: white;
}

.topic-card:hover {
  border-color: var(--color-accent);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

.topic-name {
  font-family: var(--font-ui);
  font-size: 1rem;
  margin-bottom: 0.25rem;
}

.topic-desc {
  font-size: 0.85rem;
  opacity: 0.7;
  line-height: 1.4;
  margin-bottom: 0.5rem;
}

.topic-count {
  font-size: 0.75rem;
  font-family: var(--font-ui);
  opacity: 0.5;
}

.topic-detail {
  margin-top: 1.5rem;
}

.topic-detail h2 {
  font-family: var(--font-ui);
  font-size: 1.25rem;
  margin-bottom: 0.5rem;
}

.study-steps {
  margin-top: 1rem;
}

.study-step {
  display: flex;
  gap: 0.75rem;
  padding: 0.75rem 0;
  border-bottom: 1px solid var(--color-border);
}

.step-num {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  min-width: 28px;
  border-radius: 50%;
  background: var(--color-accent);
  color: white;
  font-size: 0.8rem;
  font-weight: 700;
  font-family: var(--font-ui);
}

.step-content h4 {
  font-size: 0.95rem;
  margin-bottom: 0.15rem;
}

.step-content p {
  font-size: 0.85rem;
  opacity: 0.7;
  margin-bottom: 0.3rem;
}

.step-link {
  font-size: 0.85rem;
  color: var(--color-accent);
  text-decoration: none;
}

.step-link:hover {
  text-decoration: underline;
}

.back-btn {
  margin-top: 1rem;
  padding: 0.4rem 0.8rem;
  background: none;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.85rem;
  font-family: var(--font-ui);
  color: var(--color-text);
}

.back-btn:hover {
  background: rgba(0, 0, 0, 0.03);
}
</style>
