<template>
  <div class="discourse-view" v-if="units && units.length">
    <h3 class="discourse-title">Discourse Structure</h3>
    <div v-for="unit in units" :key="unit.discourse_id" class="discourse-unit" :class="'prominence-' + unit.prominence">
      <div class="unit-header">
        <span class="unit-type">{{ unit.discourse_type }}</span>
        <span v-if="unit.function_label" class="unit-function">{{ unit.function_label }}</span>
        <span class="unit-range">{{ unit.start_chapter }}:{{ unit.start_verse }} - {{ unit.end_chapter }}:{{ unit.end_verse }}</span>
      </div>
      <p v-if="unit.description" class="unit-description">{{ unit.description }}</p>
      <span v-if="unit.relation_to_context" class="unit-relation">{{ unit.relation_to_context }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
interface DiscourseUnit {
  discourse_id: string
  discourse_type: string
  function_label?: string
  start_chapter: number
  start_verse: number
  end_chapter: number
  end_verse: number
  description?: string
  relation_to_context?: string
  prominence: number
}

withDefaults(defineProps<{
  units: DiscourseUnit[]
}>(), { units: () => [] })
</script>

<style scoped>
.discourse-view {
  margin: 1rem 0;
  padding: 1rem;
  background: rgba(0, 0, 0, 0.02);
  border-radius: 8px;
  border: 1px solid var(--color-border);
}
.discourse-title {
  font-family: var(--font-ui);
  font-size: 0.9rem;
  color: var(--color-accent);
  margin-bottom: 0.75rem;
}
.discourse-unit {
  padding: 0.5rem 0.75rem;
  margin-bottom: 0.5rem;
  border-left: 3px solid var(--color-accent);
  border-radius: 0 4px 4px 0;
  background: rgba(255, 255, 255, 0.5);
}
.prominence-0 { border-left-color: #bdc3c7; }
.prominence-1 { border-left-color: #3498db; }
.prominence-2 { border-left-color: #2ecc71; }
.prominence-3 { border-left-color: #e74c3c; }
.unit-header { display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; }
.unit-type {
  font-weight: 700;
  font-size: 0.8rem;
  text-transform: uppercase;
  padding: 0.1rem 0.4rem;
  border-radius: 4px;
  background: rgba(0, 0, 0, 0.06);
}
.unit-function { font-size: 0.85rem; font-weight: 600; }
.unit-range { font-size: 0.75rem; opacity: 0.5; margin-left: auto; }
.unit-description { font-size: 0.85rem; line-height: 1.5; margin-top: 0.35rem; }
.unit-relation {
  display: inline-block;
  margin-top: 0.25rem;
  font-size: 0.75rem;
  padding: 0.1rem 0.4rem;
  background: rgba(0, 0, 0, 0.04);
  border-radius: 8px;
  opacity: 0.6;
}
</style>
