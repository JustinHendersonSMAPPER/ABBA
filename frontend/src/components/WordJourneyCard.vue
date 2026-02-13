<template>
  <div v-if="detail" class="word-journey-card">
    <button class="card-close" @click="$emit('close')" aria-label="Close">&times;</button>

    <div class="card-header">
      <span class="original-text">{{ detail.original }}</span>
      <span v-if="detail.transliteration" class="transliteration">{{ detail.transliteration }}</span>
    </div>

    <div class="card-gloss">
      {{ detail.gloss }}
    </div>

    <dl class="card-details">
      <template v-if="detail.strongs">
        <dt>Strong's</dt>
        <dd>{{ detail.strongs }}</dd>
      </template>
      <template v-if="detail.morphology">
        <dt>Morphology</dt>
        <dd>{{ detail.morphology }}</dd>
      </template>
      <template v-if="detail.semantic_domain">
        <dt>Domain</dt>
        <dd>{{ detail.semantic_domain }}</dd>
      </template>
      <template v-if="detail.occurrences != null">
        <dt>Occurrences</dt>
        <dd>{{ detail.occurrences }} times in the Bible</dd>
      </template>
    </dl>

    <router-link
      v-if="detail.strongs"
      :to="`/study/lexicon/${detail.strongs}`"
      class="learn-more"
    >
      Learn more &rarr;
    </router-link>
  </div>
</template>

<script setup>
defineProps({
  detail: {
    type: Object,
    default: null,
    // Expected shape:
    // { original, transliteration, gloss, strongs, morphology, semantic_domain, occurrences }
  },
})

defineEmits(['close'])
</script>

<style scoped>
.word-journey-card {
  position: relative;
  background: white;
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
