<template>
  <div class="manuscript-variants" v-if="variants && variants.length">
    <h3 class="variants-title">Manuscript Variants</h3>
    <div v-for="v in variants" :key="v.variant_id" class="variant-card" :class="'variant-' + v.significance">
      <div class="variant-header">
        <span class="variant-type">{{ v.variant_type }}</span>
        <span class="variant-significance" :class="'sig-' + v.significance">{{ v.significance }}</span>
        <ProvenanceChip
          v-if="v.id != null"
          entity-type="manuscript_variant"
          :entity-id="v.id"
        />
      </div>
      <div v-if="v.base_text" class="variant-row">
        <span class="label">Base text:</span>
        <span class="text base">{{ v.base_text }}</span>
      </div>
      <div v-if="v.variant_text" class="variant-row">
        <span class="label">Variant:</span>
        <span class="text variant">{{ v.variant_text }}</span>
      </div>
      <p v-if="v.explanation" class="variant-explanation">{{ v.explanation }}</p>
      <p v-if="v.manuscripts" class="variant-manuscripts">Manuscripts: {{ v.manuscripts }}</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import ProvenanceChip from './ProvenanceChip.vue'

interface Variant {
  id?: number
  variant_id: string
  variant_type: string
  significance: string
  base_text?: string
  variant_text?: string
  explanation?: string
  manuscripts?: string
}

withDefaults(defineProps<{
  variants: Variant[]
}>(), { variants: () => [] })
</script>

<style scoped>
.manuscript-variants {
  margin: 1rem 0;
  padding: 1rem;
  background: rgba(0, 0, 0, 0.02);
  border-radius: 8px;
  border: 1px solid var(--color-border);
}
.variants-title {
  font-family: var(--font-ui);
  font-size: 0.9rem;
  color: var(--color-accent);
  margin-bottom: 0.75rem;
}
.variant-card {
  padding: 0.75rem;
  margin-bottom: 0.5rem;
  border-radius: 6px;
  border-left: 3px solid #ccc;
}
.variant-major { border-left-color: #e74c3c; }
.variant-minor { border-left-color: #f39c12; }
.variant-orthographic { border-left-color: #bdc3c7; }
.variant-header { display: flex; gap: 0.5rem; align-items: center; margin-bottom: 0.5rem; }
.variant-type { font-weight: 600; font-size: 0.85rem; text-transform: capitalize; }
.variant-significance {
  font-size: 0.7rem;
  padding: 0.1rem 0.4rem;
  border-radius: 8px;
  text-transform: uppercase;
  font-weight: 700;
}
.sig-major { background: #fde8e8; color: #c0392b; }
.sig-minor { background: #fef9e7; color: #d68910; }
.sig-orthographic { background: #f0f0f0; color: #888; }
.variant-row { display: flex; gap: 0.5rem; margin-bottom: 0.25rem; font-size: 0.85rem; }
.label { font-weight: 600; min-width: 80px; opacity: 0.6; }
.text.base { font-family: var(--font-reading); }
.text.variant { font-family: var(--font-reading); font-style: italic; }
.variant-explanation { font-size: 0.85rem; line-height: 1.5; margin-top: 0.5rem; }
.variant-manuscripts { font-size: 0.75rem; opacity: 0.5; margin-top: 0.25rem; }
</style>
