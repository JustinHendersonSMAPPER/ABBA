<template>
  <div class="semantic-domains" v-if="domains && domains.length">
    <span
      v-for="d in domains"
      :key="d.domain_code"
      class="domain-badge"
      :title="d.domain_name + ' (' + d.domain_code + ')'"
    >
      {{ d.domain_name }}
    </span>
  </div>
</template>

<script setup>
defineProps({
  domains: { type: Array, default: () => [] },
})
</script>

<style scoped>
.semantic-domains { display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 0.25rem 0; }
.domain-badge {
  font-family: var(--font-ui);
  font-size: 0.7rem;
  padding: 0.15rem 0.5rem;
  background: rgba(52, 152, 219, 0.1);
  color: #2980b9;
  border-radius: 12px;
  border: 1px solid rgba(52, 152, 219, 0.2);
  cursor: default;
}
</style>
