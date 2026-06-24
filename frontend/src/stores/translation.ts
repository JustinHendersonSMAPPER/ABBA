import { defineStore } from 'pinia'
import { ref } from 'vue'

const STORAGE_KEY = 'abba-translation'
const DEFAULT_ID = 'BSB'

export const useTranslationStore = defineStore('translation', () => {
  const stored = localStorage.getItem(STORAGE_KEY)
  const current = ref<string>(stored ?? DEFAULT_ID)

  function setCurrent(id: string): void {
    current.value = id
    localStorage.setItem(STORAGE_KEY, id)
  }

  return { current, setCurrent }
})
