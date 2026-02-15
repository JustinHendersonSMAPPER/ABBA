<template>
  <div class="syntax-tree-view" v-if="tree && tree.root_nodes && tree.root_nodes.length">
    <h3 class="tree-title">Clause Structure</h3>
    <div class="tree-container">
      <SyntaxNodeComponent
        v-for="node in tree.root_nodes"
        :key="node.node_id"
        :node="node"
        :depth="0"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import SyntaxNodeComponent from './SyntaxNodeComponent.vue'

interface SyntaxTree {
  root_nodes: Array<{
    node_id: string
    node_type?: string
    [key: string]: unknown
  }>
}

withDefaults(defineProps<{
  tree: SyntaxTree | null
}>(), { tree: null })
</script>

<style scoped>
.syntax-tree-view {
  margin: 1rem 0;
  padding: 1rem;
  background: rgba(0, 0, 0, 0.02);
  border-radius: 8px;
  border: 1px solid var(--color-border);
}
.tree-title {
  font-family: var(--font-ui);
  font-size: 0.9rem;
  color: var(--color-accent);
  margin-bottom: 0.75rem;
}
.tree-container {
  font-family: var(--font-mono, 'Courier New', monospace);
  font-size: 0.85rem;
  line-height: 1.6;
}
</style>
