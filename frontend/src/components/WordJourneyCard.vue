<template>
  <div v-if="detail" class="word-journey-card">
    <button class="card-close" @click="$emit('close')" aria-label="Close">&times;</button>

    <div class="card-header">
      <span class="original-text">{{ detail.original_text || detail.transliteration || '' }}</span>
      <span v-if="detail.transliteration && detail.original_text" class="transliteration">{{ detail.transliteration }}</span>
    </div>

    <div class="card-gloss">
      {{ detail.english_gloss }}
    </div>

    <dl class="card-details">
      <template v-if="detail.strongs_number">
        <dt>Strong's</dt>
        <dd>{{ detail.strongs_number }}</dd>
      </template>
      <template v-if="detail.morphology_code">
        <dt>Morphology</dt>
        <dd>{{ detail.morphology_description || detail.morphology_code }}</dd>
      </template>
      <template v-if="detail.part_of_speech">
        <dt>Part of Speech</dt>
        <dd>{{ detail.part_of_speech }}</dd>
      </template>
      <template v-if="detail.language">
        <dt>Language</dt>
        <dd>{{ detail.language }}</dd>
      </template>
    </dl>

    <router-link
      v-if="detail.strongs_number"
      :to="{ name: 'lexicon', params: { strongs: detail.strongs_number } }"
      class="learn-more"
    >
      Learn more &rarr;
    </router-link>
  </div>
</template>

<script setup lang="ts">
interface WordDetail {
  original_text?: string
  transliteration?: string
  english_gloss?: string
  strongs_number?: string
  morphology_code?: string
  morphology_description?: string
  part_of_speech?: string
  language?: string
}

withDefaults(defineProps<{
  detail: WordDetail | null
}>(), { detail: null })

defineEmits<{
  close: []
}>()
</script>

<style scoped>
.word-journey-card {
  position: relative;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 1.25rem;
  max-width: 340px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
  font-family: var(--font-ui);
}

.card-close {
  position: absolute;
  top: 0.5rem;
  right: 0.75rem;
  background: none;
  border: none;
  font-size: 1.25rem;
  cursor: pointer;
  color: var(--color-text);
  opacity: 0.5;
}

.card-close:hover {
  opacity: 1;
}

.card-header {
  margin-bottom: 0.5rem;
}

.original-text {
  font-size: 1.5rem;
  font-weight: 700;
  display: block;
  line-height: 1.3;
}

.transliteration {
  font-size: 0.9rem;
  font-style: italic;
  opacity: 0.7;
}

.card-gloss {
  font-size: 1rem;
  margin-bottom: 0.75rem;
  color: var(--color-accent);
  font-weight: 500;
}

.card-details {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 0.25rem 0.75rem;
  font-size: 0.85rem;
  margin-bottom: 0.75rem;
}

.card-details dt {
  font-weight: 600;
  opacity: 0.6;
}

.card-details dd {
  margin: 0;
}

.learn-more {
  display: inline-block;
  font-size: 0.85rem;
  color: var(--color-accent);
  text-decoration: none;
}

.learn-more:hover {
  text-decoration: underline;
}
</style>
