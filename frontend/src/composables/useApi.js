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
    return call(() => request(`/search/text?${params}`))
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

  // --- Phase 9 API methods ---

  function discoverConcepts(query) {
    return call(() => request(`/discover?q=${encodeURIComponent(query)}`))
  }

  function getSemanticDomains(parent) {
    const params = parent ? `?parent=${encodeURIComponent(parent)}` : ''
    return call(() => request(`/semantic-domains${params}`))
  }

  function getWordDomains(strongsNumber) {
    return call(() => request(`/words/${encodeURIComponent(strongsNumber)}/domains`))
  }

  function getSyntaxTree(bookId, chapter, verse) {
    return call(() => request(`/syntax/${bookId}/${chapter}/${verse}`))
  }

  function getDiscourseUnits(bookId, chapter, verse) {
    return call(() => request(`/discourse/${bookId}/${chapter}/${verse}`))
  }

  function getBookDiscourse(bookId) {
    return call(() => request(`/discourse/${bookId}`))
  }

  function getManuscriptVariants(bookId, chapter, verse) {
    return call(() => request(`/variants/${bookId}/${chapter}/${verse}`))
  }

  function getSignificantVariants() {
    return call(() => request('/variants/significant'))
  }

  function multilingualSearch(query, sourceLang = 'en', translations = null) {
    const params = new URLSearchParams({ q: query, source_lang: sourceLang })
    if (translations) params.set('target_translations', translations)
    return call(() => request(`/search/multilingual?${params}`))
  }

  function createContribution(data) {
    return call(() => request('/community/contributions', {
      method: 'POST', body: JSON.stringify(data),
    }))
  }

  function listContributions(status, bookId) {
    const params = new URLSearchParams()
    if (status) params.set('status', status)
    if (bookId) params.set('book_id', bookId)
    return call(() => request(`/community/contributions?${params}`))
  }

  function reviewContribution(contributionId, decision, note) {
    return call(() => request(`/community/contributions/${contributionId}/review`, {
      method: 'POST', body: JSON.stringify({ decision, review_note: note }),
    }))
  }

  function createConceptProposal(data) {
    return call(() => request('/concepts/proposals', {
      method: 'POST', body: JSON.stringify(data),
    }))
  }

  function listConceptProposals(status) {
    const params = status ? `?status=${encodeURIComponent(status)}` : ''
    return call(() => request(`/concepts/proposals${params}`))
  }

  function getConceptGraph(conceptName, depth = 1) {
    return call(() => request(`/graph/${encodeURIComponent(conceptName)}?depth=${depth}`))
  }

  function submitConceptFeedback(conceptName, verseId, feedbackType) {
    return call(() => request(
      `/concepts/${encodeURIComponent(conceptName)}/feedback?verse_id=${encodeURIComponent(verseId)}&feedback_type=${feedbackType}`,
      { method: 'POST' }
    ))
  }

  function getAudioResource(bookId, chapter, translationId = 'engbsb') {
    return call(() => request(`/audio/${bookId}/${chapter}?translation_id=${translationId}`))
  }

  function mobileSync(data) {
    return call(() => request('/mobile/sync', {
      method: 'POST', body: JSON.stringify(data),
    }))
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
    // Phase 9
    discoverConcepts,
    getSemanticDomains,
    getWordDomains,
    getSyntaxTree,
    getDiscourseUnits,
    getBookDiscourse,
    getManuscriptVariants,
    getSignificantVariants,
    multilingualSearch,
    createContribution,
    listContributions,
    reviewContribution,
    createConceptProposal,
    listConceptProposals,
    getConceptGraph,
    submitConceptFeedback,
    getAudioResource,
    mobileSync,
  }
}
