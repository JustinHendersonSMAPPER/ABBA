<template>
  <div class="depth-dial" role="radiogroup" aria-label="Reading depth level">
    <button
      v-for="level in levels"
      :key="level.value"
      :class="['depth-option', { active: modelValue === level.value }]"
      :aria-checked="modelValue === level.value"
      role="radio"
      :title="level.description"
      @click="$emit('update:modelValue', level.value)"
    >
      {{ level.label }}
    </button>
  </div>
</template>

<script setup lang="ts">
type DepthLevel = 'basic' | 'standard' | 'deep' | 'scholarly'

interface DepthOption {
  value: DepthLevel
  label: string
  description: string
}

withDefaults(defineProps<{
  modelValue: string
}>(), { modelValue: 'basic' })

defineEmits<{
  'update:modelValue': [value: string]
}>()

const levels: DepthOption[] = [
  { value: 'basic', label: 'Read', description: 'Clean reading experience' },
  { value: 'standard', label: 'Understand', description: 'Key words and context' },
  { value: 'deep', label: 'Study', description: 'Cross-references and structure' },
  { value: 'scholarly', label: 'Analyze', description: 'Full linguistic analysis' },
]
</script>

<style scoped>
.depth-dial {
  display: flex;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  overflow: hidden;
  margin-left: auto;
}

.depth-option {
  padding: 0.35rem 0.75rem;
  border: none;
  background: transparent;
  cursor: pointer;
  font-size: 0.8rem;
  font-family: var(--font-ui);
  color: var(--color-text);
  opacity: 0.6;
  transition: all 0.2s ease;
  border-right: 1px solid var(--color-border);
}

.depth-option:last-child {
  border-right: none;
}

.depth-option:hover {
  opacity: 0.85;
  background: rgba(74, 111, 165, 0.05);
}

.depth-option.active {
  opacity: 1;
  background: var(--color-accent);
  color: white;
  font-weight: 600;
}
</style>
