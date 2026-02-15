<template>
  <div v-if="genre || hasStructures" class="literary-indicator">
    <span v-if="genre" class="genre-badge">{{ genre }}</span>
    <span
      v-for="(structure, i) in literaryStructures"
      :key="i"
      class="structure-badge"
      :title="structure.description || structure.type"
    >
      {{ structure.label || structure.type }}
    </span>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

interface LiteraryStructure {
  type: string
  label?: string
  description?: string
}

const props = withDefaults(defineProps<{
  genre: string
  literaryStructures: LiteraryStructure[]
}>(), { genre: '', literaryStructures: () => [] })

const hasStructures = computed(
  () => props.literaryStructures && props.literaryStructures.length > 0
)
</script>

<style scoped>
.literary-indicator {
  display: inline-flex;
  gap: 0.35rem;
  align-items: center;
  flex-wrap: wrap;
}

.genre-badge,
.structure-badge {
  font-family: var(--font-ui);
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  padding: 0.15rem 0.5rem;
  border-radius: 10px;
  line-height: 1.4;
}

.genre-badge {
  background: rgba(74, 111, 165, 0.12);
  color: var(--color-accent);
}

.structure-badge {
  background: rgba(100, 140, 80, 0.12);
  color: #4a7a3a;
}
</style>
