<template>
  <div class="depth-dial-wrapper">
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
    <p class="depth-caption">{{ currentCaption }}</p>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

type DepthLevel = 'basic' | 'standard' | 'deep' | 'scholarly'

interface DepthOption {
  value: DepthLevel
  label: string
  description: string
  caption: string
}

const props = withDefaults(defineProps<{
  modelValue: string
}>(), { modelValue: 'basic' })

defineEmits<{
  'update:modelValue': [value: string]
}>()

const levels: DepthOption[] = [
  { value: 'basic', label: 'Read', description: 'Clean reading experience', caption: 'Just the text' },
  { value: 'standard', label: 'Understand', description: 'Key words and context', caption: 'Word meanings & original language' },
  { value: 'deep', label: 'Study', description: 'Cross-references and structure', caption: 'Cross-references, context & syntax' },
  { value: 'scholarly', label: 'Analyze', description: 'Full linguistic analysis', caption: 'Manuscript variants, morphology & discourse' },
]

const currentCaption = computed(() => {
  const found = levels.find((l) => l.value === props.modelValue)
  return found ? found.caption : ''
})
</script>

<style scoped>
.depth-dial-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.2rem;
  margin-left: auto;
}

.depth-dial {
  display: flex;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  overflow: hidden;
}

.depth-caption {
  font-size: 0.7rem;
  font-family: var(--font-ui);
  color: var(--color-text);
  opacity: 0.55;
  text-align: center;
  line-height: 1.2;
  margin: 0;
  white-space: nowrap;
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
