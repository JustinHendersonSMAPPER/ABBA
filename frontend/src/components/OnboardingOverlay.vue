<template>
  <div v-if="visible" class="onboarding-overlay" @click.self="dismiss">
    <div class="onboarding-modal">
      <div class="step-indicators">
        <span v-for="s in totalSteps" :key="s" class="step-dot" :class="{ active: s === step }" />
      </div>

      <div v-if="step === 1" class="step-content">
        <h2 class="step-title">Welcome to ABBA</h2>
        <p class="step-text">
          ABBA helps you study the Bible with original-language insights,
          cross-references, and cultural context.
        </p>
        <p class="step-text">
          Use the <strong>Depth Dial</strong> in the top bar to control how much detail you see:
        </p>
        <ul class="depth-list">
          <li><strong>Basic</strong> -- Just the text</li>
          <li><strong>Standard</strong> -- Word meanings and translations</li>
          <li><strong>Deep</strong> -- Cultural context, cross-references, syntax</li>
          <li><strong>Scholarly</strong> -- Manuscript variants, morphology, discourse</li>
        </ul>
      </div>

      <div v-if="step === 2" class="step-content">
        <h2 class="step-title">Start with What Matters</h2>
        <p class="step-text">
          Not sure where to begin? Try one of these:
        </p>
        <div class="start-options">
          <router-link to="/topics" class="start-btn" @click="dismiss">Life Topics</router-link>
          <router-link to="/plans" class="start-btn" @click="dismiss">Reading Plans</router-link>
          <router-link to="/discover" class="start-btn" @click="dismiss">Discover Concepts</router-link>
        </div>
      </div>

      <div v-if="step === 3" class="step-content">
        <h2 class="step-title">Or Just Read</h2>
        <p class="step-text">
          Click any verse number while reading to dive deeper.
          Save verses to collections and share them with others.
        </p>
        <router-link to="/" class="start-btn primary" @click="dismiss">Start Reading</router-link>
      </div>

      <div class="step-actions">
        <button v-if="step > 1" class="nav-btn" @click="step--">Back</button>
        <span class="spacer" />
        <button v-if="step < totalSteps" class="nav-btn primary" @click="step++">Next</button>
        <button v-else class="nav-btn primary" @click="dismiss">Get Started</button>
      </div>

      <button class="skip-btn" @click="dismiss">Skip</button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'

const STORAGE_KEY = 'abba-onboarded'

const visible = ref(false)
const step = ref(1)
const totalSteps = 3

onMounted(() => {
  if (typeof localStorage !== 'undefined') {
    const onboarded = localStorage.getItem(STORAGE_KEY)
    if (onboarded !== 'true') {
      visible.value = true
    }
  }
})

function dismiss(): void {
  visible.value = false
  if (typeof localStorage !== 'undefined') {
    localStorage.setItem(STORAGE_KEY, 'true')
  }
}

defineExpose({ show: () => { visible.value = true; step.value = 1 } })
</script>

<style scoped>
.onboarding-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 200;
}
.onboarding-modal {
  background: var(--color-surface);
  border-radius: 12px;
  padding: 2rem;
  max-width: 440px;
  width: 90%;
  position: relative;
}
.step-indicators {
  display: flex;
  justify-content: center;
  gap: 0.5rem;
  margin-bottom: 1.5rem;
}
.step-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--color-border);
}
.step-dot.active {
  background: var(--color-accent);
}
.step-title {
  font-family: var(--font-ui);
  font-size: 1.25rem;
  margin-bottom: 0.75rem;
  text-align: center;
}
.step-text {
  font-size: 0.9rem;
  line-height: 1.6;
  margin-bottom: 0.75rem;
}
.depth-list {
  list-style: none;
  font-size: 0.85rem;
  line-height: 1.8;
  padding-left: 0;
}
.depth-list li {
  padding: 0.15rem 0;
}
.start-options {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin-top: 0.75rem;
}
.start-btn {
  display: block;
  padding: 0.6rem 1rem;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  text-decoration: none;
  font-family: var(--font-ui);
  font-size: 0.9rem;
  color: var(--color-text);
  text-align: center;
  cursor: pointer;
}
.start-btn:hover {
  border-color: var(--color-accent);
  color: var(--color-accent);
}
.start-btn.primary {
  background: var(--color-accent);
  color: white;
  border-color: var(--color-accent);
}
.step-actions {
  display: flex;
  align-items: center;
  margin-top: 1.5rem;
}
.spacer { flex: 1; }
.nav-btn {
  padding: 0.4rem 1rem;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  background: none;
  font-family: var(--font-ui);
  font-size: 0.85rem;
  cursor: pointer;
  color: var(--color-text);
}
.nav-btn.primary {
  background: var(--color-accent);
  color: white;
  border-color: var(--color-accent);
}
.skip-btn {
  position: absolute;
  top: 0.75rem;
  right: 0.75rem;
  background: none;
  border: none;
  font-family: var(--font-ui);
  font-size: 0.75rem;
  opacity: 0.4;
  cursor: pointer;
  color: var(--color-text);
}
.skip-btn:hover { opacity: 0.8; }
</style>
