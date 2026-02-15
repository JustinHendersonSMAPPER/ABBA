import { defineStore } from 'pinia'
import { ref } from 'vue'

interface ContextItem {
  title?: string
  text?: string
  [key: string]: unknown
}

interface CrossRef {
  book?: string
  chapter?: string | number
  verse?: string | number
  label?: string
  note?: string
  [key: string]: unknown
}

interface LiteraryStructure {
  type: string
  description?: string
  elements?: string[]
  [key: string]: unknown
}

interface SetContextOptions {
  cultural?: ContextItem[]
  crossRefs?: CrossRef[]
  literary?: LiteraryStructure[]
  reference?: string | null
}

export const useContextStore = defineStore('context', () => {
  const culturalContext = ref<ContextItem[]>([])
  const crossReferences = ref<CrossRef[]>([])
  const literaryStructures = ref<LiteraryStructure[]>([])
  const currentRef = ref<string | null>(null)

  function setContext({ cultural = [], crossRefs = [], literary = [], reference = null }: SetContextOptions = {}) {
    culturalContext.value = cultural
    crossReferences.value = crossRefs
    literaryStructures.value = literary
    currentRef.value = reference
  }

  function clear() {
    culturalContext.value = []
    crossReferences.value = []
    literaryStructures.value = []
    currentRef.value = null
  }

  return { culturalContext, crossReferences, literaryStructures, currentRef, setContext, clear }
})
