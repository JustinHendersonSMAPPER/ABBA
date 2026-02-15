<template>
  <div class="analysis-view">
    <header class="analysis-header">
      <h1 class="analysis-title">Word Analysis</h1>
      <p class="analysis-subtitle">Explore word frequency and morphological patterns</p>
    </header>

    <div class="tab-bar">
      <button class="tab-btn" :class="{ active: tab === 'frequency' }" @click="tab = 'frequency'">Word Frequency</button>
      <button class="tab-btn" :class="{ active: tab === 'morphology' }" @click="tab = 'morphology'">Morphology Patterns</button>
    </div>

    <!-- Frequency Tab -->
    <div v-if="tab === 'frequency'" class="tab-content">
      <div class="controls">
        <input v-model="freqPattern" type="text" placeholder="Strong's pattern (e.g., H26*)" class="control-input" />
        <input v-model.number="minFreq" type="number" min="1" placeholder="Min frequency" class="control-input narrow" />
        <button class="search-btn" @click="loadFrequency">Analyze</button>
      </div>

      <div v-if="freqLoading" class="status-msg">Analyzing...</div>

      <table v-if="freqResults.length" class="results-table">
        <thead>
          <tr>
            <th>Strong's</th>
            <th>Word</th>
            <th>Gloss</th>
            <th>Frequency</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(r, i) in freqResults" :key="i">
            <td>
              <router-link :to="`/lexicon/${r.strongs_number || r.strongs}`" class="strongs-link">
                {{ r.strongs_number || r.strongs }}
              </router-link>
            </td>
            <td class="word-col">{{ r.word || r.original || '' }}</td>
            <td>{{ r.gloss || r.short_definition || '' }}</td>
            <td class="freq-col">{{ r.frequency || r.count }}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Morphology Tab -->
    <div v-if="tab === 'morphology'" class="tab-content">
      <div class="controls">
        <select v-model="morphLang" class="control-select">
          <option value="hebrew">Hebrew</option>
          <option value="greek">Greek</option>
        </select>
        <input v-model="morphPattern" type="text" placeholder="Pattern filter (optional)" class="control-input" />
        <button class="search-btn" @click="loadMorphology">Analyze</button>
      </div>

      <div v-if="morphLoading" class="status-msg">Analyzing...</div>

      <div v-if="morphResults.length" class="morph-grid">
        <div v-for="(r, i) in morphResults" :key="i" class="morph-card">
          <div class="morph-header">
            <span class="morph-code">{{ r.pattern || r.morphology_code || r.code }}</span>
            <span v-if="r.count" class="morph-count">{{ r.count }} occurrences</span>
          </div>
          <p v-if="r.description || r.label" class="morph-desc">{{ r.description || r.label }}</p>
          <p v-if="r.example" class="morph-example">e.g., {{ r.example }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useApi } from '../composables/useApi'
import type { FrequencyResult, MorphologyResult } from '../types/api'

const api = useApi()
const tab = ref<string>('frequency')

// Frequency state
const freqPattern = ref<string>('')
const minFreq = ref<number>(1)
const freqResults = ref<FrequencyResult[]>([])
const freqLoading = ref<boolean>(false)

// Morphology state
const morphLang = ref<string>('hebrew')
const morphPattern = ref<string>('')
const morphResults = ref<MorphologyResult[]>([])
const morphLoading = ref<boolean>(false)

async function loadFrequency(): Promise<void> {
  freqLoading.value = true
  freqResults.value = []
  try {
    const options: Record<string, string> = {}
    if (freqPattern.value) options.strongs_pattern = freqPattern.value
    if (minFreq.value > 1) options.min_frequency = String(minFreq.value)
    const data = await api.getAnalysisFrequency(options)
    if (data) {
      freqResults.value = (data as unknown as { results?: FrequencyResult[] }).results || data || []
    }
  } finally {
    freqLoading.value = false
  }
}

async function loadMorphology(): Promise<void> {
  morphLoading.value = true
  morphResults.value = []
  try {
    const options: Record<string, string> = { language: morphLang.value }
    if (morphPattern.value) options.pattern = morphPattern.value
    const data = await api.getAnalysisMorphology(options)
    if (data) {
      morphResults.value = (data as unknown as { results?: MorphologyResult[] }).results || data || []
    }
  } finally {
    morphLoading.value = false
  }
}
</script>

<style scoped>
.analysis-header { margin-bottom: 1.25rem; }
.analysis-title { font-family: var(--font-ui); font-size: 1.5rem; margin-bottom: 0.25rem; }
.analysis-subtitle { font-size: 0.9rem; opacity: 0.6; }

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

.controls { display: flex; gap: 0.5rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.control-input {
  flex: 1;
  min-width: 150px;
  padding: 0.5rem 0.75rem;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  font-family: var(--font-ui);
  font-size: 0.9rem;
  background: var(--color-surface);
  color: var(--color-text);
}
.control-input.narrow { flex: 0; min-width: 100px; width: 120px; }
.control-select {
  padding: 0.5rem 0.6rem;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  font-family: var(--font-ui);
  font-size: 0.9rem;
  background: var(--color-surface);
  color: var(--color-text);
}
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

.results-table { width: 100%; border-collapse: collapse; }
.results-table th {
  font-family: var(--font-ui);
  font-size: 0.8rem;
  text-align: left;
  padding: 0.5rem;
  border-bottom: 2px solid var(--color-border);
  color: var(--color-accent);
}
.results-table td { padding: 0.5rem; border-bottom: 1px solid var(--color-border); font-size: 0.9rem; }
.strongs-link { color: var(--color-accent); text-decoration: none; font-family: var(--font-ui); font-weight: 600; }
.strongs-link:hover { text-decoration: underline; }
.word-col { font-weight: 600; }
.freq-col { font-family: var(--font-ui); font-weight: 700; }

.morph-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 0.75rem; }
.morph-card {
  padding: 0.75rem;
  border: 1px solid var(--color-border);
  border-radius: 6px;
}
.morph-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem; }
.morph-code { font-family: var(--font-ui); font-weight: 700; font-size: 0.85rem; }
.morph-count { font-family: var(--font-ui); font-size: 0.75rem; opacity: 0.5; }
.morph-desc { font-size: 0.85rem; line-height: 1.4; }
.morph-example { font-size: 0.8rem; opacity: 0.6; font-style: italic; margin-top: 0.2rem; }

.status-msg { font-family: var(--font-ui); font-size: 0.9rem; opacity: 0.6; padding: 1rem 0; }
</style>
