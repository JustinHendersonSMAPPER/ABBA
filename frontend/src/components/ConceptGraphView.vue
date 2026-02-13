<template>
  <div class="concept-graph-view" v-if="graph">
    <h3 class="graph-title">Concept Relationships: {{ graph.center_concept }}</h3>
    <div class="graph-container">
      <svg :viewBox="viewBox" class="graph-svg">
        <!-- Edges -->
        <line
          v-for="(edge, i) in edges"
          :key="'e-' + i"
          :x1="edge.x1" :y1="edge.y1"
          :x2="edge.x2" :y2="edge.y2"
          class="graph-edge"
          :stroke-width="edge.weight * 2"
          :stroke-opacity="0.3 + edge.weight * 0.4"
        />
        <!-- Edge labels -->
        <text
          v-for="(edge, i) in edges"
          :key="'el-' + i"
          :x="(edge.x1 + edge.x2) / 2"
          :y="(edge.y1 + edge.y2) / 2 - 5"
          class="edge-label"
        >{{ edge.type }}</text>
        <!-- Nodes -->
        <g v-for="(node, i) in positions" :key="'n-' + i">
          <circle
            :cx="node.x" :cy="node.y"
            :r="node.isCenter ? 28 : 20"
            :class="['graph-node', node.isCenter ? 'center-node' : 'related-node']"
          />
          <text
            :x="node.x" :y="node.y + 4"
            class="node-label"
            :class="{ 'center-label': node.isCenter }"
          >{{ node.name }}</text>
        </g>
      </svg>
    </div>
    <div v-if="graph.relationships && graph.relationships.length" class="relationship-list">
      <div v-for="(rel, i) in graph.relationships" :key="i" class="rel-row">
        <span class="rel-source">{{ rel.source_concept }}</span>
        <span class="rel-arrow">{{ relationshipArrow(rel.relationship_type) }}</span>
        <span class="rel-target">{{ rel.target_concept }}</span>
        <span class="rel-type">({{ rel.relationship_type }})</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, defineProps } from 'vue'

const props = defineProps({
  graph: { type: Object, default: null },
})

const viewBox = computed(() => '0 0 500 400')

const positions = computed(() => {
  if (!props.graph || !props.graph.nodes) return []
  const nodes = props.graph.nodes
  const cx = 250, cy = 200
  return nodes.map((n, i) => {
    if (n.is_center) return { ...n, x: cx, y: cy, isCenter: true }
    const angle = (2 * Math.PI * i) / Math.max(nodes.length - 1, 1)
    const r = 140
    return { ...n, x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle), isCenter: false }
  })
})

const edges = computed(() => {
  if (!props.graph || !props.graph.relationships) return []
  const posMap = {}
  positions.value.forEach(p => { posMap[p.name] = p })
  return props.graph.relationships.map(rel => {
    const s = posMap[rel.source_concept] || { x: 250, y: 200 }
    const t = posMap[rel.target_concept] || { x: 250, y: 200 }
    return { x1: s.x, y1: s.y, x2: t.x, y2: t.y, type: rel.relationship_type, weight: rel.weight || 0.5 }
  })
})

function relationshipArrow(type) {
  const arrows = { synonym: '=', antithetical: '<>', causal: '->', enables: '=>', contrast: '|', temporal: '>>' }
  return arrows[type] || '--'
}
</script>

<style scoped>
.concept-graph-view {
  margin: 1rem 0;
  padding: 1rem;
  background: rgba(0, 0, 0, 0.02);
  border-radius: 8px;
  border: 1px solid var(--color-border);
}
.graph-title {
  font-family: var(--font-ui);
  font-size: 0.9rem;
  color: var(--color-accent);
  margin-bottom: 0.75rem;
}
.graph-container {
  width: 100%;
  max-width: 500px;
  margin: 0 auto;
}
.graph-svg { width: 100%; height: auto; }
.graph-edge { stroke: var(--color-accent); }
.graph-node { fill: #ecf0f1; stroke: var(--color-accent); stroke-width: 2; }
.center-node { fill: var(--color-accent); }
.related-node { fill: #f8f9fa; }
.node-label {
  font-family: var(--font-ui);
  font-size: 10px;
  text-anchor: middle;
  fill: #333;
}
.center-label { fill: white; font-weight: 700; }
.edge-label { font-family: var(--font-ui); font-size: 8px; fill: #888; text-anchor: middle; }

.relationship-list { margin-top: 1rem; padding-top: 0.75rem; border-top: 1px solid var(--color-border); }
.rel-row { display: flex; gap: 0.5rem; align-items: center; padding: 0.2rem 0; font-size: 0.8rem; }
.rel-source, .rel-target { font-weight: 600; }
.rel-arrow { opacity: 0.4; font-family: var(--font-mono, monospace); }
.rel-type { opacity: 0.5; font-size: 0.75rem; }
</style>
