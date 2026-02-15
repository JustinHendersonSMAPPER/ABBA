<template>
  <span class="translation-lens">
    <span
      v-for="(word, i) in words"
      :key="i"
      :class="['lens-word', { 'has-richness': isRich(i) }]"
      @mouseenter="activeTooltip = isRich(i) ? i : null"
      @mouseleave="activeTooltip = null"
      @click="isRich(i) && $emit('word-click', word, richFlags[i])"
    >
      {{ typeof word === 'string' ? word : (word as Record<string, unknown>).text || '' }}<template v-if="i < words.length - 1">{{ ' ' }}</template>
      <span v-if="activeTooltip === i" class="lens-tooltip">
        {{ richFlags[i].explanation || 'Rich meaning in original language' }}
      </span>
    </span>
  </span>
</template>

<script setup lang="ts">
import { ref } from 'vue'

interface RichFlag {
  richness: number
  explanation?: string
  [key: string]: unknown
}

const props = withDefaults(defineProps<{
  words: Array<string | Record<string, unknown>>
  richFlags: RichFlag[]
}>(), { richFlags: () => [] })

defineEmits<{
  'word-click': [word: string | Record<string, unknown>, flags: RichFlag]
}>()

const activeTooltip = ref<number | null>(null)

function isRich(index: number): boolean {
  return (
    props.richFlags &&
    props.richFlags[index] != null &&
    props.richFlags[index].richness > 0.5
  )
}
</script>

<style scoped>
.translation-lens {
  font-family: var(--font-reading);
}

.lens-word {
  position: relative;
  cursor: default;
}

.lens-word.has-richness {
  text-decoration: underline;
  text-decoration-color: rgba(74, 111, 165, 0.35);
  text-decoration-thickness: 1.5px;
  text-underline-offset: 3px;
  cursor: pointer;
}

.lens-word.has-richness:hover {
  text-decoration-color: var(--color-accent);
}

.lens-tooltip {
  position: absolute;
  bottom: 100%;
  left: 50%;
  transform: translateX(-50%);
  background: var(--color-text);
  color: var(--color-bg);
  font-size: 0.75rem;
  font-family: var(--font-ui);
  padding: 0.35rem 0.6rem;
  border-radius: 4px;
  white-space: nowrap;
  max-width: 260px;
  overflow: hidden;
  text-overflow: ellipsis;
  z-index: 20;
  pointer-events: none;
  margin-bottom: 4px;
  line-height: 1.3;
}

.lens-tooltip::after {
  content: '';
  position: absolute;
  top: 100%;
  left: 50%;
  transform: translateX(-50%);
  border: 4px solid transparent;
  border-top-color: var(--color-text);
}
</style>
