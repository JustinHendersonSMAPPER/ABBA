<template>
  <div class="reading-plans">
    <h1 class="page-title">Reading Plans</h1>

    <LoadingState v-if="api.loading.value" label="Loading plans…" />
    <div v-else-if="api.error.value" class="error">{{ api.error.value }}</div>

    <template v-else-if="!activePlan">
      <div v-if="plans.length" class="plan-grid">
        <div
          v-for="plan in plans"
          :key="plan.id"
          class="plan-card"
          @click="viewPlan(plan)"
        >
          <h3 class="plan-name">{{ plan.name }}</h3>
          <p v-if="plan.description" class="plan-desc">{{ plan.description }}</p>
          <div class="plan-meta">
            <span v-if="plan.duration">{{ plan.duration }} days</span>
            <span v-if="plan.category" class="plan-category">{{ plan.category }}</span>
          </div>
        </div>
      </div>
      <p v-else class="empty-state">No reading plans available yet.</p>
    </template>

    <div v-if="activePlan" class="plan-detail">
      <button class="back-btn" @click="activePlan = null; planDetail = null">
        &larr; All plans
      </button>

      <h2 class="detail-title">{{ activePlan.name }}</h2>
      <p v-if="activePlan.description" class="detail-desc">{{ activePlan.description }}</p>

      <div v-if="planDetail" class="plan-entries">
        <div v-for="entry in planDetail.entries || []" :key="entry.day" class="plan-entry">
          <span class="entry-day">Day {{ entry.day }}</span>
          <div class="entry-content">
            <h4 v-if="entry.title">{{ entry.title }}</h4>
            <div class="entry-readings">
              <router-link
                v-for="(reading, i) in entry.readings || []"
                :key="i"
                :to="readingRoute(reading)"
                class="entry-link"
              >
                {{ reading.reference || `${reading.book} ${reading.chapter}` }}
              </router-link>
            </div>
            <p v-if="entry.reflection" class="entry-reflection">{{ entry.reflection }}</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useApi } from '../composables/useApi'
import type { ReadingPlan } from '../types/api'
import LoadingState from '../components/LoadingState.vue'

interface PlanReading {
  book: string
  chapter: string | number
  verse?: string | number
  reference?: string
}

interface PlanEntry {
  day: number
  title?: string
  readings?: PlanReading[]
  reflection?: string
}

interface PlanDetail extends ReadingPlan {
  entries?: PlanEntry[]
}

const api = useApi()

const plans = ref<ReadingPlan[]>([])
const activePlan = ref<ReadingPlan | null>(null)
const planDetail = ref<PlanDetail | null>(null)

onMounted(async () => {
  const result = await api.getPlans() as Record<string, unknown> | null
  if (result) {
    plans.value = ((result as Record<string, unknown>).plans || result) as ReadingPlan[]
  }
})

async function viewPlan(plan: ReadingPlan): Promise<void> {
  activePlan.value = plan
  const result = await api.getPlan(plan.id)
  if (result) {
    planDetail.value = result as PlanDetail
  }
}

function readingRoute(reading: PlanReading): string {
  if (reading.verse) {
    return `/study/${reading.book}/${reading.chapter}/${reading.verse}`
  }
  return `/study/${reading.book}/${reading.chapter}`
}
</script>

<style scoped>
.page-title {
  font-family: var(--font-ui);
  font-size: 1.5rem;
  margin-bottom: 1.5rem;
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

.plan-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 1rem;
}

.plan-card {
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 1.25rem;
  cursor: pointer;
  transition: border-color 0.15s, box-shadow 0.15s;
  background: var(--color-surface);
}

.plan-card:hover {
  border-color: var(--color-accent);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

.plan-name {
  font-family: var(--font-ui);
  font-size: 1.05rem;
  margin-bottom: 0.35rem;
}

.plan-desc {
  font-size: 0.85rem;
  opacity: 0.7;
  line-height: 1.4;
  margin-bottom: 0.5rem;
}

.plan-meta {
  display: flex;
  gap: 0.5rem;
  font-size: 0.75rem;
  font-family: var(--font-ui);
  opacity: 0.5;
}

.plan-category {
  text-transform: capitalize;
}

.back-btn {
  margin-bottom: 1rem;
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

.detail-title {
  font-family: var(--font-ui);
  font-size: 1.25rem;
  margin-bottom: 0.35rem;
}

.detail-desc {
  font-size: 0.9rem;
  opacity: 0.7;
  margin-bottom: 1.25rem;
}

.plan-entry {
  display: flex;
  gap: 1rem;
  padding: 0.75rem 0;
  border-bottom: 1px solid var(--color-border);
}

.entry-day {
  font-family: var(--font-ui);
  font-size: 0.8rem;
  font-weight: 600;
  min-width: 55px;
  opacity: 0.5;
}

.entry-content h4 {
  font-size: 0.95rem;
  margin-bottom: 0.2rem;
}

.entry-readings {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin-bottom: 0.3rem;
}

.entry-link {
  font-size: 0.85rem;
  color: var(--color-accent);
  text-decoration: none;
  padding: 0.1rem 0.4rem;
  background: rgba(74, 111, 165, 0.08);
  border-radius: 3px;
}

.entry-link:hover {
  text-decoration: underline;
}

.entry-reflection {
  font-size: 0.85rem;
  opacity: 0.6;
  font-style: italic;
}
</style>
