<template>
  <div class="collections-view">
    <header class="collections-header">
      <h1 class="collections-title">My Collections</h1>
      <p class="collections-subtitle">Save and organise verses for study</p>
    </header>

    <div class="create-form" v-if="showCreate">
      <input v-model="newName" type="text" placeholder="Collection name" class="form-input" />
      <input v-model="newDesc" type="text" placeholder="Description (optional)" class="form-input" />
      <div class="form-actions">
        <button class="btn-primary" @click="createCol" :disabled="!newName.trim()">Create</button>
        <button class="btn-secondary" @click="showCreate = false">Cancel</button>
      </div>
    </div>
    <button v-else class="btn-primary" @click="showCreate = true">New Collection</button>

    <div v-if="api.loading.value" class="status-msg">Loading...</div>
    <div v-else-if="api.error.value" class="status-msg error">{{ api.error.value }}</div>

    <div v-if="!activeCollection" class="collections-grid">
      <div
        v-for="col in collections"
        :key="col.id"
        class="collection-card"
        @click="viewCollection(col)"
      >
        <h3 class="col-name">{{ col.name }}</h3>
        <p v-if="col.description" class="col-desc">{{ col.description }}</p>
        <span class="col-count">{{ col.item_count || 0 }} verses</span>
        <button class="col-delete" @click.stop="removeCollection(col.id)" title="Delete collection">&times;</button>
      </div>
      <p v-if="collections.length === 0 && !api.loading.value" class="status-msg">
        No collections yet. Create one to start saving verses.
      </p>
    </div>

    <div v-if="activeCollection" class="collection-detail">
      <button class="back-btn" @click="activeCollection = null; items = []">&larr; Back to collections</button>
      <h2 class="detail-name">{{ activeCollection.name }}</h2>
      <p v-if="activeCollection.description" class="detail-desc">{{ activeCollection.description }}</p>

      <div v-if="items.length" class="items-list">
        <div v-for="(item, i) in items" :key="i" class="item-card">
          <router-link
            :to="`/study/${item.book_id}/${item.chapter}/${item.verse}`"
            class="item-ref"
          >{{ item.book_name || item.book_id }} {{ item.chapter }}:{{ item.verse }}</router-link>
          <p v-if="item.text" class="item-text">{{ item.text }}</p>
          <p v-if="item.note" class="item-note">{{ item.note }}</p>
        </div>
      </div>
      <p v-else class="status-msg">No verses in this collection yet.</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useApi } from '../composables/useApi'
import type { CollectionInfo } from '../types/api'

interface CollectionItem {
  book_id: string
  book_name?: string
  chapter: string | number
  verse: string | number
  text?: string
  note?: string
}

const api = useApi()

const collections = ref<CollectionInfo[]>([])
const activeCollection = ref<CollectionInfo | null>(null)
const items = ref<CollectionItem[]>([])
const showCreate = ref(false)
const newName = ref('')
const newDesc = ref('')

onMounted(loadCollections)

async function loadCollections(): Promise<void> {
  const data = await api.getCollections() as Record<string, unknown> | null
  if (data) collections.value = ((data as Record<string, unknown>).collections || data || []) as CollectionInfo[]
}

async function createCol(): Promise<void> {
  if (!newName.value.trim()) return
  await api.createCollection(newName.value.trim(), newDesc.value.trim())
  newName.value = ''
  newDesc.value = ''
  showCreate.value = false
  await loadCollections()
}

async function viewCollection(col: CollectionInfo): Promise<void> {
  activeCollection.value = col
  const data = await api.getCollectionItems(col.id) as Record<string, unknown> | null
  if (data) items.value = ((data as Record<string, unknown>).items || data || []) as CollectionItem[]
}

async function removeCollection(id: string): Promise<void> {
  await api.deleteCollection(id)
  await loadCollections()
}
</script>

<style scoped>
.collections-header { margin-bottom: 1.25rem; }
.collections-title { font-family: var(--font-ui); font-size: 1.5rem; margin-bottom: 0.25rem; }
.collections-subtitle { font-size: 0.9rem; opacity: 0.6; }

.create-form {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin-bottom: 1.25rem;
  padding: 1rem;
  border: 1px solid var(--color-border);
  border-radius: 8px;
  background: rgba(0, 0, 0, 0.02);
}
.form-input {
  padding: 0.5rem 0.75rem;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  font-family: var(--font-ui);
  font-size: 0.9rem;
  background: var(--color-surface);
  color: var(--color-text);
}
.form-actions { display: flex; gap: 0.5rem; }
.btn-primary {
  padding: 0.4rem 1rem;
  background: var(--color-accent);
  color: white;
  border: none;
  border-radius: 4px;
  font-family: var(--font-ui);
  font-weight: 600;
  cursor: pointer;
  margin-bottom: 1rem;
}
.btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-secondary {
  padding: 0.4rem 1rem;
  background: none;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  font-family: var(--font-ui);
  cursor: pointer;
  color: var(--color-text);
}

.collections-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 0.75rem;
}
.collection-card {
  position: relative;
  padding: 1rem;
  border: 1px solid var(--color-border);
  border-radius: 8px;
  cursor: pointer;
  transition: border-color 0.15s;
}
.collection-card:hover { border-color: var(--color-accent); }
.col-name { font-family: var(--font-ui); font-size: 1rem; margin-bottom: 0.25rem; }
.col-desc { font-size: 0.85rem; opacity: 0.6; margin-bottom: 0.25rem; }
.col-count { font-size: 0.75rem; opacity: 0.5; font-family: var(--font-ui); }
.col-delete {
  position: absolute;
  top: 0.5rem;
  right: 0.5rem;
  background: none;
  border: none;
  font-size: 1.1rem;
  cursor: pointer;
  opacity: 0.3;
  color: var(--color-text);
}
.col-delete:hover { opacity: 1; color: #c0392b; }

.back-btn {
  background: none;
  border: none;
  font-family: var(--font-ui);
  color: var(--color-accent);
  cursor: pointer;
  padding: 0;
  margin-bottom: 0.75rem;
  font-size: 0.9rem;
}
.detail-name { font-family: var(--font-ui); font-size: 1.25rem; margin-bottom: 0.25rem; }
.detail-desc { font-size: 0.9rem; opacity: 0.6; margin-bottom: 1rem; }

.item-card {
  padding: 0.75rem;
  margin-bottom: 0.5rem;
  border: 1px solid var(--color-border);
  border-radius: 6px;
}
.item-ref { font-family: var(--font-ui); font-weight: 600; color: var(--color-accent); text-decoration: none; }
.item-ref:hover { text-decoration: underline; }
.item-text { font-size: 0.9rem; margin-top: 0.25rem; line-height: 1.5; }
.item-note { font-size: 0.8rem; opacity: 0.6; font-style: italic; margin-top: 0.15rem; }

.status-msg { font-family: var(--font-ui); font-size: 0.9rem; opacity: 0.6; padding: 1rem 0; }
.error { color: #c0392b; opacity: 1; }
</style>
