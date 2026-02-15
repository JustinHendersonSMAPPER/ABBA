<template>
  <div class="shared-view">
    <div v-if="api.loading.value" class="loading">Loading shared content...</div>
    <div v-else-if="api.error.value" class="error">{{ api.error.value }}</div>

    <template v-else-if="shared">
      <header class="shared-header">
        <span class="shared-badge">Shared</span>
        <h1 class="shared-title">{{ shared.title }}</h1>
      </header>

      <div v-if="shared.share_type === 'verse' && shared.content" class="shared-verse">
        <p class="verse-text">{{ shared.content.text || '' }}</p>
        <div v-if="shared.content.book" class="verse-ref">
          {{ shared.content.book }} {{ shared.content.chapter }}:{{ shared.content.verse }}
        </div>
      </div>

      <div v-else class="shared-content">
        <pre v-if="typeof shared.content === 'object'">{{ JSON.stringify(shared.content, null, 2) }}</pre>
        <p v-else>{{ shared.content }}</p>
      </div>

      <div class="shared-actions">
        <router-link
          v-if="shared.content && shared.content.book"
          :to="`/study/${shared.content.book}/${shared.content.chapter}/${shared.content.verse}`"
          class="open-link"
        >Open in Study View</router-link>
      </div>

      <p class="shared-meta">
        Shared {{ shared.created_at ? new Date(shared.created_at).toLocaleDateString() : '' }}
      </p>
    </template>

    <p v-else class="empty-state">Shared content not found.</p>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { useApi } from '../composables/useApi'
import type { ShareData } from '../types/api'

const route = useRoute()
const api = useApi()
const shared = ref<ShareData | null>(null)

onMounted(async () => {
  const token = route.params.token as string
  if (token) {
    const data = await api.getShare(token)
    if (data) shared.value = data
  }
})
</script>

<style scoped>
.shared-header {
  margin-bottom: 1.5rem;
}
.shared-badge {
  display: inline-block;
  padding: 0.15rem 0.5rem;
  background: var(--color-accent);
  color: white;
  border-radius: 12px;
  font-family: var(--font-ui);
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  margin-bottom: 0.5rem;
}
.shared-title {
  font-family: var(--font-ui);
  font-size: 1.5rem;
}

.shared-verse {
  padding: 1.5rem;
  background: rgba(74, 111, 165, 0.05);
  border-radius: 8px;
  border-left: 4px solid var(--color-accent);
  margin-bottom: 1.5rem;
}
.verse-text {
  font-family: var(--font-reading);
  font-size: 1.15rem;
  line-height: 1.8;
  margin-bottom: 0.75rem;
}
.verse-ref {
  font-family: var(--font-ui);
  font-size: 0.85rem;
  opacity: 0.6;
  font-weight: 600;
}

.shared-content {
  margin-bottom: 1.5rem;
}
.shared-content pre {
  font-size: 0.85rem;
  overflow-x: auto;
  background: rgba(0, 0, 0, 0.03);
  padding: 1rem;
  border-radius: 6px;
}

.shared-actions {
  margin-bottom: 1rem;
}
.open-link {
  display: inline-block;
  padding: 0.5rem 1rem;
  background: var(--color-accent);
  color: white;
  text-decoration: none;
  border-radius: 6px;
  font-family: var(--font-ui);
  font-weight: 600;
  font-size: 0.9rem;
}
.open-link:hover {
  opacity: 0.9;
}

.shared-meta {
  font-family: var(--font-ui);
  font-size: 0.8rem;
  opacity: 0.4;
}

.loading, .error, .empty-state {
  font-family: var(--font-ui);
  font-size: 0.9rem;
  opacity: 0.6;
  padding: 2rem 0;
}
.error { color: #c0392b; opacity: 1; }
</style>
