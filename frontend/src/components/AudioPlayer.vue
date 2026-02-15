<template>
  <div class="audio-player" v-if="audio">
    <div class="audio-info">
      <span class="audio-label">{{ audio.title || 'Chapter Audio' }}</span>
      <span v-if="audio.reader" class="audio-reader">{{ audio.reader }}</span>
    </div>
    <audio
      v-if="audio.url || audio.audio_url"
      :src="audio.url || audio.audio_url"
      controls
      preload="none"
      class="audio-element"
    />
    <p v-else class="audio-unavailable">Audio not available for this chapter.</p>
  </div>
</template>

<script setup lang="ts">
import type { AudioResource } from '../types/api'

defineProps<{
  audio: AudioResource | null
}>()
</script>

<style scoped>
.audio-player {
  margin-bottom: 1rem;
  padding: 0.5rem 0.75rem;
  background: rgba(74, 111, 165, 0.06);
  border-radius: 6px;
  border: 1px solid var(--color-border);
}
.audio-info {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.35rem;
}
.audio-label {
  font-family: var(--font-ui);
  font-size: 0.8rem;
  font-weight: 600;
}
.audio-reader {
  font-family: var(--font-ui);
  font-size: 0.75rem;
  opacity: 0.5;
}
.audio-element {
  width: 100%;
  height: 32px;
}
.audio-unavailable {
  font-family: var(--font-ui);
  font-size: 0.8rem;
  opacity: 0.5;
}
</style>
