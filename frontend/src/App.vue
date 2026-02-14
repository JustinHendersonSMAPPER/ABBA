<template>
  <div id="abba-app" :class="{ 'dark-mode': isDark }">
    <nav class="app-nav">
      <router-link to="/" class="nav-brand">ABBA</router-link>
      <div class="nav-links">
        <router-link to="/">Read</router-link>
        <router-link to="/topics">Topics</router-link>
        <router-link to="/plans">Plans</router-link>
        <router-link to="/discover">Discover</router-link>
      </div>
      <div class="nav-actions">
        <DepthDial v-model="depthLevel" />
        <button class="dark-toggle" :aria-label="isDark ? 'Light mode' : 'Dark mode'" @click="isDark = !isDark">
          {{ isDark ? 'Light' : 'Dark' }}
        </button>
      </div>
    </nav>
    <main class="app-main">
      <router-view :depth="depthLevel" />
    </main>
    <ContextSidebar v-if="depthLevel !== 'basic'" />
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
import DepthDial from './components/DepthDial.vue'
import ContextSidebar from './components/ContextSidebar.vue'

const depthLevel = ref('basic')
const isDark = ref(false)

// Persist dark mode preference
const saved = typeof localStorage !== 'undefined' && localStorage.getItem('abba-dark')
if (saved === 'true') isDark.value = true

watch(isDark, (v) => {
  if (typeof localStorage !== 'undefined') localStorage.setItem('abba-dark', String(v))
})
</script>

<style>
:root {
  --color-bg: #fafaf8;
  --color-text: #2d2d2d;
  --color-accent: #4a6fa5;
  --color-border: #e0ddd5;
  --color-surface: #ffffff;
  --font-reading: 'Georgia', serif;
  --font-ui: system-ui, -apple-system, sans-serif;
}

.dark-mode {
  --color-bg: #1a1a2e;
  --color-text: #e0e0e0;
  --color-accent: #6b8fc4;
  --color-border: #3a3a4a;
  --color-surface: #262640;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: var(--font-ui);
  background: var(--color-bg);
  color: var(--color-text);
  transition: background 0.2s, color 0.2s;
}

.app-nav {
  display: flex;
  align-items: center;
  padding: 0.75rem 1.5rem;
  border-bottom: 1px solid var(--color-border);
  gap: 1.5rem;
  flex-wrap: wrap;
}

.nav-brand {
  font-weight: 700;
  font-size: 1.25rem;
  text-decoration: none;
  color: var(--color-accent);
}

.nav-links {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
}

.nav-links a {
  text-decoration: none;
  color: var(--color-text);
  opacity: 0.7;
  font-size: 0.9rem;
}

.nav-links a.router-link-active {
  opacity: 1;
  font-weight: 600;
}

.nav-actions {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-left: auto;
}

.dark-toggle {
  padding: 0.3rem 0.6rem;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  background: var(--color-surface);
  color: var(--color-text);
  cursor: pointer;
  font-size: 0.8rem;
  font-family: var(--font-ui);
}

.dark-toggle:hover {
  border-color: var(--color-accent);
}

.app-main {
  max-width: 720px;
  margin: 2rem auto;
  padding: 0 1rem;
  font-family: var(--font-reading);
  line-height: 1.8;
}

/* Responsive adjustments */
@media (max-width: 600px) {
  .app-nav {
    padding: 0.5rem 0.75rem;
    gap: 0.75rem;
  }

  .nav-links {
    gap: 0.5rem;
  }

  .nav-links a {
    font-size: 0.8rem;
  }

  .app-main {
    margin: 1rem auto;
    padding: 0 0.5rem;
  }
}
</style>
