<template>
  <span v-if="record" ref="rootEl" class="provenance-chip">
    <button
      class="chip-badge"
      :aria-expanded="open"
      :aria-label="`Provenance: ${tierLabel} — click for details`"
      @click="toggleOpen"
    >{{ tierLabel }}</button>
    <span v-if="open" class="chip-popover" role="tooltip">
      <span class="popover-source">{{ record.source }}<template v-if="record.source_detail"> — {{ record.source_detail }}</template></span>
      <span class="popover-rationale">{{ record.trust_rationale }}</span>
      <span v-if="record.confidence != null" class="popover-confidence">Confidence: {{ Math.round(record.confidence * 100) }}%</span>
      <button class="popover-close" aria-label="Close" @click.stop="() => { open = false; detachListeners() }">&#x2715;</button>
    </span>
  </span>
</template>

<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount, computed } from 'vue'
import { useApi } from '../composables/useApi'
import type { ProvenanceData } from '../types/api'

const props = defineProps<{
  entityType: string
  entityId: string | number
}>()

const api = useApi()
const record = ref<ProvenanceData | null>(null)
const open = ref(false)
const rootEl = ref<HTMLElement | null>(null)

const tierLabel = computed(() => {
  if (!record.value) return ''
  switch (record.value.trust_tier) {
    case 'A': return '📚 Sourced'
    case 'B': return '🤖 AI-assisted'
    case 'C': return '⏸ Deferred'
    default: return record.value.trust_tier
  }
})

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Escape' && open.value) {
    open.value = false
  }
}

function onOutsideClick(e: MouseEvent) {
  if (rootEl.value && !rootEl.value.contains(e.target as Node)) {
    open.value = false
  }
}

function attachListeners() {
  window.addEventListener('keydown', onKeydown)
  window.addEventListener('mousedown', onOutsideClick)
}

function detachListeners() {
  window.removeEventListener('keydown', onKeydown)
  window.removeEventListener('mousedown', onOutsideClick)
}

function toggleOpen() {
  open.value = !open.value
  if (open.value) {
    attachListeners()
  } else {
    detachListeners()
  }
}

onMounted(async () => {
  record.value = await api.getProvenance(props.entityType, String(props.entityId))
})

onBeforeUnmount(() => {
  detachListeners()
})
</script>

<style scoped>
.provenance-chip {
  display: inline-flex;
  align-items: center;
  position: relative;
  margin-left: 0.4rem;
  vertical-align: middle;
}

.chip-badge {
  display: inline-flex;
  align-items: center;
  font-size: 0.7rem;
  padding: 0.1rem 0.45rem;
  border-radius: 10px;
  border: 1px solid var(--color-border, #ccc);
  background: var(--color-surface, #fff);
  color: var(--color-text, #333);
  cursor: pointer;
  font-family: var(--font-ui, sans-serif);
  line-height: 1.4;
  white-space: nowrap;
}

.chip-badge:hover {
  border-color: var(--color-accent, #4a6fa5);
  color: var(--color-accent, #4a6fa5);
}

.chip-popover {
  position: absolute;
  top: calc(100% + 6px);
  left: 0;
  z-index: 200;
  background: var(--color-surface, #fff);
  border: 1px solid var(--color-border, #ccc);
  border-radius: 6px;
  padding: 0.6rem 0.75rem 0.5rem;
  min-width: 220px;
  max-width: 320px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
}

.popover-source {
  font-size: 0.8rem;
  font-weight: 600;
  font-family: var(--font-ui, sans-serif);
  color: var(--color-text, #333);
  word-break: break-word;
}

.popover-rationale {
  font-size: 0.78rem;
  line-height: 1.4;
  color: var(--color-text, #333);
  opacity: 0.8;
  word-break: break-word;
}

.popover-confidence {
  font-size: 0.75rem;
  font-family: var(--font-ui, sans-serif);
  opacity: 0.65;
}

.popover-close {
  position: absolute;
  top: 0.3rem;
  right: 0.4rem;
  background: none;
  border: none;
  cursor: pointer;
  font-size: 0.75rem;
  color: var(--color-text, #333);
  opacity: 0.5;
  line-height: 1;
  padding: 0;
}

.popover-close:hover {
  opacity: 1;
}

@media (prefers-color-scheme: dark) {
  .chip-popover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
  }
}
</style>
