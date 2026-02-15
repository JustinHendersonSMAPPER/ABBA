import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useContextStore = defineStore('context', () => {
  const culturalContext = ref([])
  const crossReferences = ref([])
  const literaryStructures = ref([])
  const currentRef = ref(null)

  function setContext({ cultural = [], crossRefs = [], literary = [], reference = null } = {}) {
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
