import { ref } from 'vue'

const BASE_URL = '/api/v1'

async function request(url, options = {}) {
  const response = await fetch(`${BASE_URL}${url}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  })
  if (!response.ok) {
    const body = await response.text().catch(() => '')
    throw new Error(`API ${response.status}: ${body || response.statusText}`)
  }
  return response.json()
}

export function useApi() {
  const loading = ref(false)
  const error = ref(null)

  async function call(fn) {
    loading.value = true
    error.value = null
    try {
      return await fn()
    } catch (err) {
      error.value = err.message || 'Unknown error'
      return null
    } finally {
      loading.value = false
    }
  }

  function getVerse(book, chapter, verse, depth = 'basic') {
    return call(() =>
      request(`/verses/${encodeURIComponent(book)}/${chapter}/${verse}?depth=${depth}`)
    )
  }

  function getChapter(book, chapter, depth = 'basic') {
    return call(() =>
      request(`/verses/${encodeURIComponent(book)}/${chapter}?depth=${depth}`)
    )
  }

  function searchText(query, options = {}) {
    const params = new URLSearchParams({ q: query, ...options })
    return call(() => request(`/search?${params}`))
  }

  function getTopics() {
    return call(() => request('/life-topics'))
  }

  function getTopic(topicId) {
    return call(() => request(`/life-topics/${encodeURIComponent(topicId)}`))
  }

  function getPlans() {
    return call(() => request('/reading-plans'))
  }

  function getPlan(planId) {
    return call(() => request(`/reading-plans/${encodeURIComponent(planId)}`))
  }

  function getBooks() {
    return call(() => request('/books'))
  }

  function getWordDetail(strongsNumber) {
    return call(() => request(`/lexicon/${encodeURIComponent(strongsNumber)}`))
  }

  function getCrossReferences(book, chapter, verse) {
    return call(() =>
      request(`/verses/${encodeURIComponent(book)}/${chapter}/${verse}/cross-references`)
    )
  }

  function getContext(book, chapter, verse) {
    return call(() =>
      request(`/verses/${encodeURIComponent(book)}/${chapter}/${verse}/context`)
    )
  }

  return {
    loading,
    error,
    getVerse,
    getChapter,
    searchText,
    getTopics,
    getTopic,
    getPlans,
    getPlan,
    getBooks,
    getWordDetail,
    getCrossReferences,
    getContext,
  }
}
