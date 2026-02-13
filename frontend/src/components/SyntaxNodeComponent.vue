<template>
  <div class="syntax-node" :style="{ marginLeft: depth * 1.5 + 'rem' }">
    <div class="node-header" :class="['node-' + node.node_type]">
      <span class="node-type">{{ node.node_type }}</span>
      <span v-if="node.role" class="node-role">[{{ node.role }}]</span>
      <span v-if="node.clause_type" class="node-clause">({{ node.clause_type }})</span>
      <span v-if="node.text_content" class="node-text">{{ node.text_content }}</span>
    </div>
    <SyntaxNodeComponent
      v-for="child in (node.children || [])"
      :key="child.node_id"
      :node="child"
      :depth="depth + 1"
    />
  </div>
</template>

<script setup>
defineProps({
  node: { type: Object, required: true },
  depth: { type: Number, default: 0 },
})
</script>

<style scoped>
.syntax-node { margin: 0.15rem 0; }
.node-header {
  display: inline-flex;
  gap: 0.4rem;
  align-items: center;
  padding: 0.15rem 0.4rem;
  border-radius: 4px;
  font-size: 0.8rem;
}
.node-sentence { background: rgba(52, 152, 219, 0.1); }
.node-clause { background: rgba(46, 204, 113, 0.1); }
.node-phrase { background: rgba(155, 89, 182, 0.1); }
.node-word { background: rgba(241, 196, 15, 0.1); }
.node-type { font-weight: 700; text-transform: uppercase; font-size: 0.7rem; opacity: 0.6; }
.node-role { color: var(--color-accent); font-weight: 600; }
.node-clause { font-style: italic; opacity: 0.7; }
.node-text { font-family: var(--font-reading); }
</style>
