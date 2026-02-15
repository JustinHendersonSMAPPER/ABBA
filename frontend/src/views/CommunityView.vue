<template>
  <div class="community-view">
    <header class="community-header">
      <h1 class="community-title">Help Improve ABBA</h1>
      <p class="community-subtitle">Submit corrections, suggest concepts, and review contributions</p>
    </header>

    <div class="tab-bar">
      <button class="tab-btn" :class="{ active: tab === 'contributions' }" @click="tab = 'contributions'">Contributions</button>
      <button class="tab-btn" :class="{ active: tab === 'proposals' }" @click="tab = 'proposals'">Concept Proposals</button>
    </div>

    <!-- Contributions Tab -->
    <div v-if="tab === 'contributions'" class="tab-content">
      <div class="filters">
        <select v-model="contribFilter" class="filter-select" @change="loadContributions">
          <option value="">All</option>
          <option value="pending">Pending</option>
          <option value="approved">Approved</option>
          <option value="rejected">Rejected</option>
        </select>
        <button class="btn-primary" @click="showContribForm = !showContribForm">
          {{ showContribForm ? 'Cancel' : 'Submit Contribution' }}
        </button>
      </div>

      <div v-if="showContribForm" class="submit-form">
        <select v-model="newContrib.contribution_type" class="form-input">
          <option value="correction">Text Correction</option>
          <option value="note">Study Note</option>
          <option value="cross_reference">Cross-Reference</option>
          <option value="cultural_context">Cultural Context</option>
        </select>
        <input v-model="newContrib.book_id" type="text" placeholder="Book ID (e.g., GEN)" class="form-input" />
        <input v-model="newContrib.chapter" type="number" placeholder="Chapter" class="form-input" />
        <input v-model="newContrib.verse" type="number" placeholder="Verse" class="form-input" />
        <textarea v-model="newContrib.content" placeholder="Your contribution..." class="form-textarea" rows="4"></textarea>
        <input v-model="newContrib.source" type="text" placeholder="Source or reference (optional)" class="form-input" />
        <button class="btn-primary" @click="submitContribution" :disabled="!newContrib.content">Submit</button>
      </div>

      <div v-if="api.loading.value" class="status-msg">Loading...</div>

      <div v-if="contributions.length" class="item-list">
        <div v-for="c in contributions" :key="c.id" class="item-card">
          <div class="item-header">
            <span class="item-type">{{ c.contribution_type }}</span>
            <span class="item-status" :class="'status-' + c.status">{{ c.status }}</span>
            <span v-if="c.book_id" class="item-ref">{{ c.book_id }} {{ c.chapter }}:{{ c.verse }}</span>
          </div>
          <p class="item-content">{{ c.content }}</p>
          <p v-if="c.source" class="item-source">Source: {{ c.source }}</p>

          <div v-if="c.status === 'pending'" class="review-actions">
            <button class="review-btn approve" @click="reviewItem(c.id, 'approved')">Approve</button>
            <button class="review-btn reject" @click="reviewItem(c.id, 'rejected')">Reject</button>
          </div>
        </div>
      </div>
      <p v-else-if="!api.loading.value" class="status-msg">No contributions yet.</p>
    </div>

    <!-- Concept Proposals Tab -->
    <div v-if="tab === 'proposals'" class="tab-content">
      <div class="filters">
        <select v-model="proposalFilter" class="filter-select" @change="loadProposals">
          <option value="">All</option>
          <option value="pending">Pending</option>
          <option value="approved">Approved</option>
          <option value="rejected">Rejected</option>
        </select>
        <button class="btn-primary" @click="showProposalForm = !showProposalForm">
          {{ showProposalForm ? 'Cancel' : 'Propose Concept' }}
        </button>
      </div>

      <div v-if="showProposalForm" class="submit-form">
        <input v-model="newProposal.concept_name" type="text" placeholder="Concept name (e.g., Divine Mercy)" class="form-input" />
        <textarea v-model="newProposal.description" placeholder="Describe the concept..." class="form-textarea" rows="3"></textarea>
        <input v-model="newProposal.strongs_numbers" type="text" placeholder="Related Strong's numbers (comma-separated)" class="form-input" />
        <input v-model="newProposal.verse_references" type="text" placeholder="Key verses (comma-separated, e.g., GEN.1.1, JHN.3.16)" class="form-input" />
        <button class="btn-primary" @click="submitProposal" :disabled="!newProposal.concept_name">Submit</button>
      </div>

      <div v-if="proposals.length" class="item-list">
        <div v-for="p in proposals" :key="p.id" class="item-card">
          <div class="item-header">
            <span class="item-type concept">{{ p.concept_name }}</span>
            <span class="item-status" :class="'status-' + p.status">{{ p.status }}</span>
          </div>
          <p class="item-content">{{ p.description }}</p>
          <p v-if="p.strongs_numbers" class="item-source">Strong's: {{ p.strongs_numbers }}</p>
        </div>
      </div>
      <p v-else-if="!api.loading.value" class="status-msg">No concept proposals yet.</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useApi } from '../composables/useApi'
import type { Contribution, ConceptProposal } from '../types/api'

interface ContribForm {
  contribution_type: string
  book_id: string
  chapter: number | null
  verse: number | null
  content: string
  source: string
}

interface ProposalForm {
  concept_name: string
  description: string
  strongs_numbers: string
  verse_references: string
}

const api = useApi()

const tab = ref<string>('contributions')
const contribFilter = ref<string>('')
const proposalFilter = ref<string>('')
const showContribForm = ref<boolean>(false)
const showProposalForm = ref<boolean>(false)

const contributions = ref<Contribution[]>([])
const proposals = ref<ConceptProposal[]>([])

const newContrib = ref<ContribForm>({
  contribution_type: 'correction',
  book_id: '',
  chapter: null,
  verse: null,
  content: '',
  source: '',
})

const newProposal = ref<ProposalForm>({
  concept_name: '',
  description: '',
  strongs_numbers: '',
  verse_references: '',
})

onMounted(() => {
  loadContributions()
  loadProposals()
})

async function loadContributions(): Promise<void> {
  const data = await api.listContributions(contribFilter.value || undefined)
  if (data) contributions.value = (data as unknown as { contributions?: Contribution[] }).contributions || data || []
}

async function loadProposals(): Promise<void> {
  const data = await api.listConceptProposals(proposalFilter.value || undefined)
  if (data) proposals.value = (data as unknown as { proposals?: ConceptProposal[] }).proposals || data || []
}

async function submitContribution(): Promise<void> {
  const payload: Record<string, unknown> = { ...newContrib.value }
  if (payload.chapter) payload.chapter = Number(payload.chapter)
  if (payload.verse) payload.verse = Number(payload.verse)
  await api.createContribution(payload)
  showContribForm.value = false
  newContrib.value = { contribution_type: 'correction', book_id: '', chapter: null, verse: null, content: '', source: '' }
  await loadContributions()
}

async function submitProposal(): Promise<void> {
  const payload: Record<string, unknown> = {
    concept_name: newProposal.value.concept_name,
    description: newProposal.value.description,
    strongs_numbers: newProposal.value.strongs_numbers ? newProposal.value.strongs_numbers.split(',').map((s: string) => s.trim()) : [],
    verse_references: newProposal.value.verse_references ? newProposal.value.verse_references.split(',').map((s: string) => s.trim()) : [],
  }
  await api.createConceptProposal(payload)
  showProposalForm.value = false
  newProposal.value = { concept_name: '', description: '', strongs_numbers: '', verse_references: '' }
  await loadProposals()
}

async function reviewItem(id: string, decision: string): Promise<void> {
  await api.reviewContribution(id, decision, '')
  await loadContributions()
}
</script>

<style scoped>
.community-header { margin-bottom: 1.25rem; }
.community-title { font-family: var(--font-ui); font-size: 1.5rem; margin-bottom: 0.25rem; }
.community-subtitle { font-size: 0.9rem; opacity: 0.6; }

.tab-bar { display: flex; gap: 0; margin-bottom: 1.5rem; border-bottom: 2px solid var(--color-border); }
.tab-btn {
  padding: 0.5rem 1rem;
  background: none;
  border: none;
  border-bottom: 2px solid transparent;
  margin-bottom: -2px;
  font-family: var(--font-ui);
  font-size: 0.9rem;
  cursor: pointer;
  color: var(--color-text);
  opacity: 0.6;
}
.tab-btn.active { border-bottom-color: var(--color-accent); opacity: 1; font-weight: 600; }

.filters { display: flex; gap: 0.5rem; margin-bottom: 1rem; align-items: center; }
.filter-select {
  padding: 0.4rem 0.6rem;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  font-family: var(--font-ui);
  font-size: 0.85rem;
  background: var(--color-surface);
  color: var(--color-text);
}

.btn-primary {
  padding: 0.4rem 1rem;
  background: var(--color-accent);
  color: white;
  border: none;
  border-radius: 4px;
  font-family: var(--font-ui);
  font-weight: 600;
  cursor: pointer;
  font-size: 0.85rem;
}
.btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }

.submit-form {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin-bottom: 1.5rem;
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
.form-textarea {
  padding: 0.5rem 0.75rem;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  font-family: var(--font-ui);
  font-size: 0.9rem;
  resize: vertical;
  background: var(--color-surface);
  color: var(--color-text);
}

.item-list { display: flex; flex-direction: column; gap: 0.5rem; }
.item-card {
  padding: 0.75rem;
  border: 1px solid var(--color-border);
  border-radius: 6px;
}
.item-header { display: flex; gap: 0.5rem; align-items: center; margin-bottom: 0.35rem; flex-wrap: wrap; }
.item-type {
  font-family: var(--font-ui);
  font-size: 0.75rem;
  font-weight: 700;
  text-transform: uppercase;
  padding: 0.1rem 0.4rem;
  background: rgba(0, 0, 0, 0.06);
  border-radius: 4px;
}
.item-type.concept { background: rgba(74, 111, 165, 0.1); color: var(--color-accent); }
.item-status { font-family: var(--font-ui); font-size: 0.7rem; font-weight: 700; text-transform: uppercase; padding: 0.1rem 0.4rem; border-radius: 8px; }
.status-pending { background: #fef9e7; color: #d68910; }
.status-approved { background: #eafaf1; color: #27ae60; }
.status-rejected { background: #fde8e8; color: #c0392b; }
.item-ref { font-family: var(--font-ui); font-size: 0.8rem; opacity: 0.5; margin-left: auto; }
.item-content { font-size: 0.9rem; line-height: 1.5; }
.item-source { font-size: 0.8rem; opacity: 0.5; margin-top: 0.25rem; }

.review-actions { display: flex; gap: 0.4rem; margin-top: 0.5rem; }
.review-btn {
  padding: 0.25rem 0.6rem;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  font-family: var(--font-ui);
  font-size: 0.8rem;
  cursor: pointer;
  background: none;
  color: var(--color-text);
}
.review-btn.approve:hover { background: #eafaf1; color: #27ae60; border-color: #27ae60; }
.review-btn.reject:hover { background: #fde8e8; color: #c0392b; border-color: #c0392b; }

.status-msg { font-family: var(--font-ui); font-size: 0.9rem; opacity: 0.6; padding: 1rem 0; }
</style>
