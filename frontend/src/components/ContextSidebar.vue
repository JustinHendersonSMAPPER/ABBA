<template>
  <aside :class="['context-sidebar', { collapsed }]">
    <button class="sidebar-toggle" @click="collapsed = !collapsed" :aria-label="collapsed ? 'Open sidebar' : 'Close sidebar'">
      {{ collapsed ? '\u25C0' : '\u25B6' }}
    </button>

    <div v-if="!collapsed" class="sidebar-content">
      <section v-if="culturalContext && culturalContext.length" class="sidebar-section">
        <h3 class="section-title">Cultural Context</h3>
        <div v-for="(item, i) in culturalContext" :key="'ctx-' + i" class="context-item">
          <strong v-if="item.title">{{ item.title }}</strong>
          <p>{{ item.text }}</p>
        </div>
      </section>

      <section v-if="crossReferences && crossReferences.length" class="sidebar-section">
        <h3 class="section-title">Cross-References</h3>
        <ul class="cross-ref-list">
          <li v-for="(ref, i) in crossReferences" :key="'ref-' + i" class="cross-ref-item">
            <router-link
              v-if="ref.book && ref.chapter && ref.verse"
              :to="`/study/${ref.book}/${ref.chapter}/${ref.verse}`"
              class="cross-ref-link"
            >
              {{ ref.label || `${ref.book} ${ref.chapter}:${ref.verse}` }}
            </router-link>
            <span v-else>{{ ref.label || ref }}</span>
            <span v-if="ref.note" class="cross-ref-note">{{ ref.note }}</span>
          </li>
        </ul>
      </section>

      <section v-if="literaryStructures && literaryStructures.length" class="sidebar-section">
        <h3 class="section-title">Literary Structure</h3>
        <div v-for="(structure, i) in literaryStructures" :key="'lit-' + i" class="structure-item">
          <span class="structure-type">{{ structure.type }}</span>
          <p v-if="structure.description">{{ structure.description }}</p>
          <ul v-if="structure.elements && structure.elements.length" class="structure-elements">
            <li v-for="(el, j) in structure.elements" :key="'el-' + j">{{ el }}</li>
          </ul>
        </div>
      </section>

      <p v-if="isEmpty" class="sidebar-empty">
        No additional context available for this passage.
      </p>
    </div>
  </aside>
</template>

<script setup>
import { ref, computed } from 'vue'

const props = defineProps({
  culturalContext: { type: Array, default: () => [] },
  crossReferences: { type: Array, default: () => [] },
  literaryStructures: { type: Array, default: () => [] },
})

const collapsed = ref(false)

const isEmpty = computed(() => {
  return (
    (!props.culturalContext || props.culturalContext.length === 0) &&
    (!props.crossReferences || props.crossReferences.length === 0) &&
    (!props.literaryStructures || props.literaryStructures.length === 0)
  )
})
</script>

<style scoped>
.context-sidebar {
  position: fixed;
  right: 0;
  top: 0;
  bottom: 0;
  width: 320px;
  background: var(--color-bg);
  border-left: 1px solid var(--color-border);
  padding: 1rem;
  overflow-y: auto;
  transition: width 0.2s ease;
  z-index: 10;
}

.context-sidebar.collapsed {
  width: 40px;
  padding: 0.5rem;
}

.sidebar-toggle {
  display: block;
  background: none;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  cursor: pointer;
  padding: 0.25rem 0.5rem;
  margin-bottom: 1rem;
  font-size: 0.75rem;
  color: var(--color-text);
}

.sidebar-content {
  padding-top: 0.5rem;
}

.sidebar-section {
  margin-bottom: 1.5rem;
}

.section-title {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--color-accent);
  margin-bottom: 0.5rem;
  font-family: var(--font-ui);
}

.context-item p {
  font-size: 0.875rem;
  line-height: 1.5;
  margin-top: 0.25rem;
}

.cross-ref-list {
  list-style: none;
}

.cross-ref-item {
  padding: 0.25rem 0;
  font-size: 0.875rem;
}

.cross-ref-link {
  color: var(--color-accent);
  text-decoration: none;
}

.cross-ref-link:hover {
  text-decoration: underline;
}

.cross-ref-note {
  display: block;
  font-size: 0.8rem;
  opacity: 0.7;
  margin-top: 0.15rem;
}

.structure-type {
  display: inline-block;
  font-size: 0.75rem;
  font-weight: 600;
  background: rgba(74, 111, 165, 0.1);
  padding: 0.1rem 0.4rem;
  border-radius: 3px;
  margin-bottom: 0.25rem;
}

.structure-elements {
  list-style: none;
  padding-left: 0.75rem;
  font-size: 0.85rem;
  border-left: 2px solid var(--color-border);
  margin-top: 0.25rem;
}

.structure-elements li {
  padding: 0.15rem 0;
}

.sidebar-empty {
  font-size: 0.85rem;
  opacity: 0.5;
  font-style: italic;
}
</style>
