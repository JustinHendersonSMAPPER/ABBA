<template>
  <div id="abba-app" :class="{ 'dark-mode': isDark }">
    <nav class="app-nav">
      <router-link to="/" class="nav-brand">ABBA</router-link>
      <button class="nav-hamburger" @click="mobileNavOpen = !mobileNavOpen">Menu</button>
      <div class="nav-links" :class="{ open: mobileNavOpen }">
        <router-link to="/">Read</router-link>
        <router-link to="/search">Search</router-link>
        <router-link to="/topics">Topics</router-link>
        <router-link to="/plans">Plans</router-link>
        <router-link to="/discover">Discover</router-link>
        <div class="nav-more">
          <button class="nav-more-btn" @click="showMore = !showMore">More</button>
          <div v-if="showMore" class="nav-dropdown" @mouseleave="showMore = false">
            <router-link to="/compare" @click="showMore = false">Compare</router-link>
            <router-link to="/lexicon" @click="showMore = false">Words</router-link>
            <router-link to="/domains" @click="showMore = false">Domains</router-link>
            <router-link to="/collections" @click="showMore = false">Collections</router-link>
            <router-link to="/community" @click="showMore = false">Community</router-link>
            <router-link to="/analysis" @click="showMore = false">Analysis</router-link>
          </div>
        </div>
      </div>
      <div class="nav-actions">
        <form class="nav-search" @submit.prevent="goSearch">
          <input
            v-model="searchQuery"
            type="text"
            placeholder="Search the Bible..."
            class="nav-search-input"
          />
          <button type="submit" class="nav-search-btn" :disabled="!searchQuery.trim()">Go</button>
        </form>
        <DepthDial v-model="depthLevel" />
        <button class="dark-toggle" :aria-label="isDark ? 'Light mode' : 'Dark mode'" @click="isDark = !isDark">
          {{ isDark ? 'Light' : 'Dark' }}
        </button>
      </div>
    </nav>
    <div class="app-layout">
      <main class="app-main" :class="{ 'has-sidebar': depthLevel !== 'basic' }">
        <router-view :depth="depthLevel" />
      </main>
      <ContextSidebar
        v-if="depthLevel !== 'basic'"
        :culturalContext="contextStore.culturalContext"
        :crossReferences="contextStore.crossReferences"
        :literaryStructures="contextStore.literaryStructures"
      />
    </div>
    <OnboardingOverlay ref="onboardingRef" />
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import DepthDial from './components/DepthDial.vue'
import ContextSidebar from './components/ContextSidebar.vue'
import OnboardingOverlay from './components/OnboardingOverlay.vue'
import { useContextStore } from './stores/context'

const router = useRouter()
const contextStore = useContextStore()

const depthLevel = ref<string>('basic')
const isDark = ref(false)
const searchQuery = ref('')
const showMore = ref(false)
const mobileNavOpen = ref(false)

// Persist dark mode preference
const saved = typeof localStorage !== 'undefined' && localStorage.getItem('abba-dark')
if (saved === 'true') isDark.value = true

watch(isDark, (v) => {
  if (typeof localStorage !== 'undefined') localStorage.setItem('abba-dark', String(v))
})

function goSearch() {
  if (!searchQuery.value.trim()) return
  router.push({ name: 'search', query: { q: searchQuery.value.trim() } })
  searchQuery.value = ''
}
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
  gap: 1rem;
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
  gap: 0.75rem;
  flex-wrap: wrap;
}

.nav-links a {
  text-decoration: none;
  color: var(--color-text);
  opacity: 0.7;
  font-size: 0.85rem;
}

.nav-links a.router-link-active {
  opacity: 1;
  font-weight: 600;
}

.nav-actions {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-left: auto;
}

.nav-search {
  display: flex;
  gap: 0;
}

.nav-search-input {
  padding: 0.3rem 0.5rem;
  border: 1px solid var(--color-border);
  border-right: none;
  border-radius: 4px 0 0 4px;
  font-size: 0.8rem;
  font-family: var(--font-ui);
  width: 160px;
  background: var(--color-surface);
  color: var(--color-text);
}

.nav-search-input:focus {
  outline: none;
  border-color: var(--color-accent);
}

.nav-search-btn {
  padding: 0.3rem 0.5rem;
  border: 1px solid var(--color-border);
  border-radius: 0 4px 4px 0;
  background: var(--color-accent);
  color: white;
  font-size: 0.8rem;
  font-family: var(--font-ui);
  cursor: pointer;
}

.nav-search-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
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

.app-layout {
  display: flex;
}

.app-main {
  flex: 1;
  max-width: 720px;
  margin: 2rem auto;
  padding: 0 1rem;
  font-family: var(--font-reading);
  line-height: 1.8;
}

.app-main.has-sidebar {
  margin-right: 340px;
}

/* Responsive adjustments */
@media (max-width: 900px) {
  .app-main.has-sidebar {
    margin-right: 0;
  }
}

@media (max-width: 600px) {
  .app-nav {
    padding: 0.5rem 0.75rem;
    gap: 0.5rem;
  }

  .nav-links {
    gap: 0.4rem;
  }

  .nav-links a {
    font-size: 0.75rem;
  }

  .nav-search-input {
    width: 100px;
  }

  .app-main {
    margin: 1rem auto;
    padding: 0 0.5rem;
  }
}

.nav-more { position: relative; }
.nav-more-btn { background: none; border: none; font-size: 0.85rem; cursor: pointer; color: var(--color-text); opacity: 0.7; font-family: var(--font-ui); padding: 0; }
.nav-more-btn:hover { opacity: 1; }
.nav-dropdown { position: absolute; top: 100%; left: 0; background: var(--color-surface); border: 1px solid var(--color-border); border-radius: 6px; padding: 0.5rem 0; min-width: 140px; z-index: 50; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
.nav-dropdown a { display: block; padding: 0.4rem 1rem; text-decoration: none; color: var(--color-text); font-size: 0.85rem; }
.nav-dropdown a:hover { background: rgba(74, 111, 165, 0.08); }
.nav-dropdown a.router-link-active { font-weight: 600; }

.nav-hamburger { display: none; background: none; border: 1px solid var(--color-border); border-radius: 4px; padding: 0.3rem 0.6rem; font-family: var(--font-ui); font-size: 0.8rem; cursor: pointer; color: var(--color-text); }
@media (max-width: 600px) {
  .nav-hamburger { display: block; }
  .nav-links { display: none; flex-direction: column; width: 100%; }
  .nav-links.open { display: flex; }
}
</style>
