<template>
  <div id="abba-app" :class="{ 'dark-mode': isDark }">
    <nav class="app-nav">
      <router-link to="/" class="nav-brand">ABBA</router-link>
      <div class="nav-links">
        <router-link to="/">Read</router-link>
        <router-link to="/topics">Topics</router-link>
        <router-link to="/plans">Plans</router-link>
      </div>
      <DepthDial v-model="depthLevel" />
    </nav>
    <main class="app-main">
      <router-view :depth="depthLevel" />
    </main>
    <ContextSidebar v-if="depthLevel !== 'basic'" />
  </div>
</template>

<script setup>
import { ref } from 'vue'
import DepthDial from './components/DepthDial.vue'
import ContextSidebar from './components/ContextSidebar.vue'

const depthLevel = ref('basic')
const isDark = ref(false)
</script>

<style>
:root {
  --color-bg: #fafaf8;
  --color-text: #2d2d2d;
  --color-accent: #4a6fa5;
  --color-border: #e0ddd5;
  --font-reading: 'Georgia', serif;
  --font-ui: system-ui, -apple-system, sans-serif;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: var(--font-ui);
  background: var(--color-bg);
  color: var(--color-text);
}

.app-nav {
  display: flex;
  align-items: center;
  padding: 0.75rem 1.5rem;
  border-bottom: 1px solid var(--color-border);
  gap: 1.5rem;
}

.nav-brand {
  font-weight: 700;
  font-size: 1.25rem;
  text-decoration: none;
  color: var(--color-accent);
}

.nav-links { display: flex; gap: 1rem; }
.nav-links a {
  text-decoration: none;
  color: var(--color-text);
  opacity: 0.7;
}
.nav-links a.router-link-active { opacity: 1; font-weight: 600; }

.app-main {
  max-width: 720px;
  margin: 2rem auto;
  padding: 0 1rem;
  font-family: var(--font-reading);
  line-height: 1.8;
}
</style>
