<template>
  <div class="loading-state" role="status" aria-live="polite" :aria-label="label">
    <div class="spinner" aria-hidden="true">
      <div class="spinner-ring"></div>
    </div>
    <span class="loading-label">{{ label }}</span>
  </div>
</template>

<script setup lang="ts">
withDefaults(defineProps<{
  label?: string
}>(), { label: 'Loading…' })
</script>

<style scoped>
.loading-state {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  font-family: var(--font-ui);
  font-size: 0.9rem;
  opacity: 0.7;
  padding: 2rem 0;
}

.spinner {
  width: 18px;
  height: 18px;
  flex-shrink: 0;
}

.spinner-ring {
  width: 18px;
  height: 18px;
  border: 2px solid var(--color-border);
  border-top-color: var(--color-accent);
  border-radius: 50%;
  animation: spin 0.75s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.loading-label {
  color: var(--color-text);
}
</style>
