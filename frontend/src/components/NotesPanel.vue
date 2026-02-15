<template>
  <section class="notes-panel" v-if="bookId && chapter && verse">
    <h2 class="panel-heading">Notes</h2>

    <div class="note-form">
      <textarea
        v-model="newNote"
        placeholder="Add a note about this verse..."
        class="note-input"
        rows="3"
      ></textarea>
      <div class="note-form-row">
        <select v-model="noteType" class="type-select">
          <option value="personal">Personal</option>
          <option value="study">Study</option>
        </select>
        <button class="save-btn" @click="saveNote" :disabled="!newNote.trim() || saving">
          {{ saving ? 'Saving...' : 'Save Note' }}
        </button>
      </div>
    </div>

    <div v-if="loading" class="status-msg">Loading notes...</div>

    <div v-if="notes.length" class="notes-list">
      <div v-for="note in notes" :key="note.id" class="note-card">
        <div class="note-meta">
          <span class="note-type">{{ note.note_type || 'personal' }}</span>
          <span v-if="note.created_at" class="note-date">{{ formatDate(note.created_at) }}</span>
          <button class="note-delete" @click="removeNote(note.id)" title="Delete note">&times;</button>
        </div>
        <p class="note-content">{{ note.content }}</p>
      </div>
    </div>

    <p v-else-if="!loading" class="status-msg">No notes yet for this verse.</p>
  </section>
</template>

<script setup>
import { ref, watch, onMounted } from 'vue'
import { useApi } from '../composables/useApi.js'

const props = defineProps({
  bookId: { type: [String, Number], default: '' },
  chapter: { type: [String, Number], default: '' },
  verse: { type: [String, Number], default: '' },
})

const api = useApi()
const notes = ref([])
const newNote = ref('')
const noteType = ref('personal')
const loading = ref(false)
const saving = ref(false)

onMounted(loadNotes)

watch(() => [props.bookId, props.chapter, props.verse], loadNotes)

async function loadNotes() {
  if (!props.bookId || !props.chapter || !props.verse) return
  loading.value = true
  const data = await api.getNotes(props.bookId, props.chapter, props.verse)
  if (data) notes.value = data.notes || data || []
  loading.value = false
}

async function saveNote() {
  if (!newNote.value.trim()) return
  saving.value = true
  await api.createNote(props.bookId, props.chapter, props.verse, newNote.value.trim(), noteType.value)
  newNote.value = ''
  saving.value = false
  await loadNotes()
}

async function removeNote(noteId) {
  await api.deleteNote(noteId)
  await loadNotes()
}

function formatDate(dateStr) {
  try {
    return new Date(dateStr).toLocaleDateString()
  } catch {
    return dateStr
  }
}
</script>

<style scoped>
.notes-panel {
  margin-top: 1.5rem;
  padding-top: 1rem;
  border-top: 1px solid var(--color-border);
}
.panel-heading {
  font-family: var(--font-ui);
  font-size: 1rem;
  color: var(--color-accent);
  margin-bottom: 0.75rem;
}

.note-form { margin-bottom: 1rem; }
.note-input {
  width: 100%;
  padding: 0.5rem 0.75rem;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  font-family: var(--font-ui);
  font-size: 0.9rem;
  resize: vertical;
  background: var(--color-surface);
  color: var(--color-text);
}
.note-input:focus { outline: none; border-color: var(--color-accent); }
.note-form-row {
  display: flex;
  gap: 0.5rem;
  margin-top: 0.4rem;
  align-items: center;
}
.type-select {
  padding: 0.3rem 0.5rem;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  font-family: var(--font-ui);
  font-size: 0.8rem;
  background: var(--color-surface);
  color: var(--color-text);
}
.save-btn {
  padding: 0.3rem 0.75rem;
  background: var(--color-accent);
  color: white;
  border: none;
  border-radius: 4px;
  font-family: var(--font-ui);
  font-size: 0.85rem;
  font-weight: 600;
  cursor: pointer;
}
.save-btn:disabled { opacity: 0.5; cursor: not-allowed; }

.note-card {
  padding: 0.6rem;
  margin-bottom: 0.4rem;
  border: 1px solid var(--color-border);
  border-radius: 6px;
}
.note-meta {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.25rem;
}
.note-type {
  font-family: var(--font-ui);
  font-size: 0.7rem;
  text-transform: uppercase;
  background: rgba(74, 111, 165, 0.1);
  padding: 0.1rem 0.4rem;
  border-radius: 3px;
}
.note-date { font-size: 0.75rem; opacity: 0.5; font-family: var(--font-ui); }
.note-delete {
  margin-left: auto;
  background: none;
  border: none;
  font-size: 1rem;
  cursor: pointer;
  opacity: 0.3;
  color: var(--color-text);
}
.note-delete:hover { opacity: 1; color: #c0392b; }
.note-content { font-size: 0.9rem; line-height: 1.5; }

.status-msg { font-family: var(--font-ui); font-size: 0.85rem; opacity: 0.5; font-style: italic; }
</style>
