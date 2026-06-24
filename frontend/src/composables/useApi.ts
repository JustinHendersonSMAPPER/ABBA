import { ref } from 'vue'
import type { Ref } from 'vue'
import type {
  AudioResource,
  BookInfo,
  CollectionInfo,
  CollectionItem,
  ConceptDiscoveryResult,
  ConceptGraph,
  ConceptProposal,
  Contribution,
  FrequencyResult,
  GenreShift,
  LexiconEntry,
  MorphologyResult,
  PassageInfo,
  ProvenanceData,
  ReadingPlan,
  ReadingPlanDetail,
  SearchResult,
  SemanticDomain,
  ShareData,
  TopicDetail,
  TopicSummary,
  VerseResponse,
  WordDomainResult,
  WordExplanation,
} from '../types/api'

export const DEFAULT_TRANSLATION = 'BSB'

const BASE_URL = '/api/v1'

async function request<T>(url: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${BASE_URL}${url}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  })
  if (!response.ok) {
    const body = await response.text().catch(() => '')
    throw new Error(`API ${response.status}: ${body || response.statusText}`)
  }
  return response.json() as Promise<T>
}

export interface UseApiReturn {
  loading: Ref<boolean>
  error: Ref<string | null>
  DEFAULT_TRANSLATION: string
  getProvenance: (entityType: string, entityId: string) => Promise<ProvenanceData | null>
  getVerse: (translationId: string, bookId: number, chapter: string | number, verse: string | number, depth?: string) => Promise<VerseResponse | null>
  getChapter: (translationId: string, bookId: number, chapter: string | number, depth?: string) => Promise<VerseResponse[] | null>
  searchText: (query: string, options?: Record<string, string>) => Promise<SearchResult[] | null>
  getTopics: () => Promise<TopicSummary[] | null>
  getTopic: (topicId: string) => Promise<TopicDetail | null>
  searchTopics: (query: string) => Promise<TopicSummary[] | null>
  getPlans: () => Promise<ReadingPlan[] | null>
  getPlan: (planId: string) => Promise<ReadingPlanDetail | null>
  getBooks: () => Promise<BookInfo[] | null>
  getWordDetail: (strongsNumber: string) => Promise<LexiconEntry | null>
  discoverConcepts: (query: string) => Promise<ConceptDiscoveryResult | null>
  getSemanticDomains: (parent?: string | null) => Promise<SemanticDomain[] | null>
  getDomainWords: (domainCode: string) => Promise<WordDomainResult[] | null>
  getWordDomains: (strongsNumber: string) => Promise<WordDomainResult | null>
  getSyntaxTree: (bookId: string, chapter: string | number, verse: string | number) => Promise<VerseResponse['syntax_tree']>
  getDiscourseUnits: (bookId: string, chapter: string | number, verse: string | number) => Promise<{ units?: unknown[] } | null>
  getBookDiscourse: (bookId: string) => Promise<unknown[] | null>
  getManuscriptVariants: (bookId: string, chapter: string | number, verse: string | number) => Promise<{ variants?: unknown[] } | null>
  getSignificantVariants: () => Promise<unknown[] | null>
  multilingualSearch: (query: string, sourceLang?: string, translations?: string | null) => Promise<SearchResult[] | null>
  createContribution: (data: Record<string, unknown>) => Promise<Contribution | null>
  listContributions: (status?: string, bookId?: string) => Promise<Contribution[] | null>
  reviewContribution: (contributionId: string, decision: string, note: string) => Promise<{ status: string } | null>
  createConceptProposal: (data: Record<string, unknown>) => Promise<ConceptProposal | null>
  listConceptProposals: (status?: string) => Promise<ConceptProposal[] | null>
  getConceptGraph: (conceptName: string, depth?: number) => Promise<ConceptGraph | null>
  submitConceptFeedback: (conceptName: string, verseId: string, feedbackType: string) => Promise<{ status: string } | null>
  getConceptFeedbackSummary: (conceptName: string) => Promise<Record<string, unknown> | null>
  getAudioResource: (bookId: string, chapter: string | number, translationId?: string) => Promise<AudioResource | null>
  mobileSync: (data: Record<string, unknown>) => Promise<Record<string, unknown> | null>
  searchStrongs: (strongsNumber: string, limit?: number) => Promise<SearchResult[] | null>
  semanticSearch: (query: string, options?: Record<string, string>) => Promise<SearchResult[] | null>
  compareTranslations: (book: string, chapter: string | number, verse: string | number, translations: string[]) => Promise<Record<string, unknown> | null>
  createNote: (bookId: string, chapter: string | number, verse: string | number, content: string, noteType?: string) => Promise<Record<string, unknown> | null>
  getNotes: (bookId: string, chapter: string | number, verse: string | number) => Promise<Record<string, unknown>[] | null>
  deleteNote: (noteId: string) => Promise<Record<string, unknown> | null>
  createCollection: (name: string, description?: string) => Promise<CollectionInfo | null>
  getCollections: () => Promise<CollectionInfo[] | null>
  addToCollection: (collectionId: string, bookId: string, chapter: string | number, verse: string | number, note?: string) => Promise<Record<string, unknown> | null>
  getCollectionItems: (collectionId: string) => Promise<CollectionItem[] | null>
  deleteCollection: (collectionId: string) => Promise<Record<string, unknown> | null>
  createShare: (shareType: string, title: string, content: Record<string, unknown>) => Promise<ShareData | null>
  getShare: (token: string) => Promise<ShareData | null>
  exportVerse: (translationId: string, bookId: string, chapter: string | number, verse: string | number, format?: string) => Promise<Record<string, unknown> | null>
  getWordExplanation: (strongsNumber: string) => Promise<WordExplanation | null>
  getGenreShifts: (bookId: string) => Promise<GenreShift[] | null>
  getPassages: (bookId: string, chapter: string | number) => Promise<PassageInfo[] | null>
  getAnalysisFrequency: (options?: Record<string, string>) => Promise<FrequencyResult[] | null>
  getAnalysisMorphology: (options?: Record<string, string>) => Promise<MorphologyResult[] | null>
  getAnalysisSemanticDomain: (domain: string) => Promise<unknown[] | null>
}

export function useApi(): UseApiReturn {
  const loading: Ref<boolean> = ref(false)
  const error: Ref<string | null> = ref(null)
  let _pendingCount = 0

  async function call<T>(fn: () => Promise<T>): Promise<T | null> {
    _pendingCount++
    if (_pendingCount === 1) loading.value = true
    error.value = null
    try {
      return await fn()
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : 'Unknown error'
      return null
    } finally {
      _pendingCount--
      if (_pendingCount === 0) loading.value = false
    }
  }

  function getVerse(translationId: string, bookId: number, chapter: string | number, verse: string | number, depth = 'basic'): Promise<VerseResponse | null> {
    return call(() =>
      request<VerseResponse>(`/verses/${encodeURIComponent(translationId)}/${bookId}/${chapter}/${verse}?depth=${depth}`)
    )
  }

  function getChapter(translationId: string, bookId: number, chapter: string | number, depth = 'basic'): Promise<VerseResponse[] | null> {
    return call(() =>
      request<VerseResponse[]>(`/verses/${encodeURIComponent(translationId)}/${bookId}/${chapter}?depth=${depth}`)
    )
  }

  function searchText(query: string, options: Record<string, string> = {}): Promise<SearchResult[] | null> {
    const params = new URLSearchParams({ q: query, ...options })
    return call(() => request<SearchResult[]>(`/search/text?${params}`))
  }

  function getTopics(): Promise<TopicSummary[] | null> {
    return call(() => request<TopicSummary[]>('/life-topics'))
  }

  function getTopic(topicId: string): Promise<TopicDetail | null> {
    return call(() => request<TopicDetail>(`/life-topics/${encodeURIComponent(topicId)}`))
  }

  function searchTopics(query: string): Promise<TopicSummary[] | null> {
    return call(() => request<TopicSummary[]>(`/life-topics/search?q=${encodeURIComponent(query)}`))
  }

  function getPlans(): Promise<ReadingPlan[] | null> {
    return call(() => request<ReadingPlan[]>('/reading-plans'))
  }

  function getPlan(planId: string): Promise<ReadingPlanDetail | null> {
    return call(() => request<ReadingPlanDetail>(`/reading-plans/${encodeURIComponent(planId)}`))
  }

  function getBooks(): Promise<BookInfo[] | null> {
    return call(() => request<BookInfo[]>('/books'))
  }

  function getWordDetail(strongsNumber: string): Promise<LexiconEntry | null> {
    return call(() => request<LexiconEntry>(`/lexicon/${encodeURIComponent(strongsNumber)}`))
  }

  // Phase 9 API methods

  function discoverConcepts(query: string): Promise<ConceptDiscoveryResult | null> {
    return call(() => request<ConceptDiscoveryResult>(`/discover?q=${encodeURIComponent(query)}`))
  }

  function getSemanticDomains(parent?: string | null): Promise<SemanticDomain[] | null> {
    const params = parent ? `?parent=${encodeURIComponent(parent)}` : ''
    return call(() => request<SemanticDomain[]>(`/semantic-domains${params}`))
  }

  function getDomainWords(domainCode: string): Promise<WordDomainResult[] | null> {
    return call(() => request<WordDomainResult[]>(`/semantic-domains/${encodeURIComponent(domainCode)}/words`))
  }

  function getWordDomains(strongsNumber: string): Promise<WordDomainResult | null> {
    return call(() => request<WordDomainResult>(`/words/${encodeURIComponent(strongsNumber)}/domains`))
  }

  function getSyntaxTree(bookId: string, chapter: string | number, verse: string | number): Promise<VerseResponse['syntax_tree']> {
    return call(() => request<NonNullable<VerseResponse['syntax_tree']>>(`/syntax/${bookId}/${chapter}/${verse}`))
  }

  function getDiscourseUnits(bookId: string, chapter: string | number, verse: string | number): Promise<{ units?: unknown[] } | null> {
    return call(() => request<{ units?: unknown[] }>(`/discourse/${bookId}/${chapter}/${verse}`))
  }

  function getBookDiscourse(bookId: string): Promise<unknown[] | null> {
    return call(() => request<unknown[]>(`/discourse/${bookId}`))
  }

  function getManuscriptVariants(bookId: string, chapter: string | number, verse: string | number): Promise<{ variants?: unknown[] } | null> {
    return call(() => request<{ variants?: unknown[] }>(`/variants/${bookId}/${chapter}/${verse}`))
  }

  function getSignificantVariants(): Promise<unknown[] | null> {
    return call(() => request<unknown[]>('/variants/significant'))
  }

  function multilingualSearch(query: string, sourceLang = 'en', translations: string | null = null): Promise<SearchResult[] | null> {
    const params = new URLSearchParams({ q: query, source_lang: sourceLang })
    if (translations) params.set('target_translations', translations)
    return call(() => request<SearchResult[]>(`/search/multilingual?${params}`))
  }

  function createContribution(data: Record<string, unknown>): Promise<Contribution | null> {
    return call(() => request<Contribution>('/community/contributions', {
      method: 'POST', body: JSON.stringify(data),
    }))
  }

  function listContributions(status?: string, bookId?: string): Promise<Contribution[] | null> {
    const params = new URLSearchParams()
    if (status) params.set('status', status)
    if (bookId) params.set('book_id', bookId)
    return call(() => request<Contribution[]>(`/community/contributions?${params}`))
  }

  function reviewContribution(contributionId: string, decision: string, note: string): Promise<{ status: string } | null> {
    return call(() => request<{ status: string }>(`/community/contributions/${contributionId}/review`, {
      method: 'POST', body: JSON.stringify({ decision, review_note: note }),
    }))
  }

  function createConceptProposal(data: Record<string, unknown>): Promise<ConceptProposal | null> {
    return call(() => request<ConceptProposal>('/concepts/proposals', {
      method: 'POST', body: JSON.stringify(data),
    }))
  }

  function listConceptProposals(status?: string): Promise<ConceptProposal[] | null> {
    const params = status ? `?status=${encodeURIComponent(status)}` : ''
    return call(() => request<ConceptProposal[]>(`/concepts/proposals${params}`))
  }

  function getConceptGraph(conceptName: string, depth = 1): Promise<ConceptGraph | null> {
    return call(() => request<ConceptGraph>(`/graph/${encodeURIComponent(conceptName)}?depth=${depth}`))
  }

  function submitConceptFeedback(conceptName: string, verseId: string, feedbackType: string): Promise<{ status: string } | null> {
    return call(() => request<{ status: string }>(
      `/concepts/${encodeURIComponent(conceptName)}/feedback?verse_id=${encodeURIComponent(verseId)}&feedback_type=${feedbackType}`,
      { method: 'POST' }
    ))
  }

  function getConceptFeedbackSummary(conceptName: string): Promise<Record<string, unknown> | null> {
    return call(() => request<Record<string, unknown>>(`/concepts/${encodeURIComponent(conceptName)}/feedback/summary`))
  }

  function getAudioResource(bookId: string, chapter: string | number, translationId = DEFAULT_TRANSLATION): Promise<AudioResource | null> {
    return call(() => request<AudioResource>(`/audio/${bookId}/${chapter}?translation_id=${translationId}`))
  }

  function mobileSync(data: Record<string, unknown>): Promise<Record<string, unknown> | null> {
    return call(() => request<Record<string, unknown>>('/mobile/sync', {
      method: 'POST', body: JSON.stringify(data),
    }))
  }

  // Search

  function searchStrongs(strongsNumber: string, limit = 50): Promise<SearchResult[] | null> {
    const params = new URLSearchParams({ limit: String(limit) })
    return call(() => request<SearchResult[]>(`/search/strongs/${encodeURIComponent(strongsNumber)}?${params}`))
  }

  function semanticSearch(query: string, options: Record<string, string> = {}): Promise<SearchResult[] | null> {
    const params = new URLSearchParams({ q: query, ...options })
    return call(() => request<SearchResult[]>(`/search/semantic?${params}`))
  }

  // Translation Comparison

  function compareTranslations(book: string, chapter: string | number, verse: string | number, translations: string[]): Promise<Record<string, unknown> | null> {
    const params = new URLSearchParams()
    translations.forEach(t => params.append('translations', t))
    return call(() => request<Record<string, unknown>>(`/compare/${book}/${chapter}/${verse}?${params}`))
  }

  // Notes

  function createNote(bookId: string, chapter: string | number, verse: string | number, content: string, noteType = 'personal'): Promise<Record<string, unknown> | null> {
    return call(() => request<Record<string, unknown>>(`/notes/${bookId}/${chapter}/${verse}`, {
      method: 'POST', body: JSON.stringify({ content, note_type: noteType }),
    }))
  }

  function getNotes(bookId: string, chapter: string | number, verse: string | number): Promise<Record<string, unknown>[] | null> {
    return call(() => request<Record<string, unknown>[]>(`/notes/${bookId}/${chapter}/${verse}`))
  }

  function deleteNote(noteId: string): Promise<Record<string, unknown> | null> {
    return call(() => request<Record<string, unknown>>(`/notes/${noteId}`, { method: 'DELETE' }))
  }

  // Collections

  function createCollection(name: string, description = ''): Promise<CollectionInfo | null> {
    return call(() => request<CollectionInfo>('/collections', {
      method: 'POST', body: JSON.stringify({ name, description }),
    }))
  }

  function getCollections(): Promise<CollectionInfo[] | null> {
    return call(() => request<CollectionInfo[]>('/collections'))
  }

  function addToCollection(collectionId: string, bookId: string, chapter: string | number, verse: string | number, note = ''): Promise<Record<string, unknown> | null> {
    return call(() => request<Record<string, unknown>>(`/collections/${collectionId}/items`, {
      method: 'POST', body: JSON.stringify({ book_id: bookId, chapter, verse, note }),
    }))
  }

  function getCollectionItems(collectionId: string): Promise<CollectionItem[] | null> {
    return call(() => request<CollectionItem[]>(`/collections/${collectionId}/items`))
  }

  function deleteCollection(collectionId: string): Promise<Record<string, unknown> | null> {
    return call(() => request<Record<string, unknown>>(`/collections/${collectionId}`, { method: 'DELETE' }))
  }

  // Sharing & Export

  function createShare(shareType: string, title: string, content: Record<string, unknown>): Promise<ShareData | null> {
    return call(() => request<ShareData>('/share', {
      method: 'POST', body: JSON.stringify({ share_type: shareType, title, content }),
    }))
  }

  function getShare(token: string): Promise<ShareData | null> {
    return call(() => request<ShareData>(`/share/${encodeURIComponent(token)}`))
  }

  function exportVerse(translationId: string, bookId: string, chapter: string | number, verse: string | number, format = 'markdown'): Promise<Record<string, unknown> | null> {
    return call(() => request<Record<string, unknown>>(
      `/export/verse/${encodeURIComponent(translationId)}/${bookId}/${chapter}/${verse}?format=${format}`
    ))
  }

  // Word Explanations — resolves null on 404 without setting global error (empty table is normal)
  async function getWordExplanation(strongsNumber: string): Promise<WordExplanation | null> {
    try {
      const response = await fetch(`${BASE_URL}/word-explanations/${encodeURIComponent(strongsNumber)}`, {
        headers: { 'Content-Type': 'application/json' },
      })
      if (!response.ok) return null
      return response.json() as Promise<WordExplanation>
    } catch {
      return null
    }
  }

  // Genre Shifts

  function getGenreShifts(bookId: string): Promise<GenreShift[] | null> {
    return call(() => request<GenreShift[]>(`/genre-shifts/${bookId}`))
  }

  // Passages

  function getPassages(bookId: string, chapter: string | number): Promise<PassageInfo[] | null> {
    return call(() => request<PassageInfo[]>(`/passages/${bookId}/${chapter}`))
  }

  // Analysis endpoints

  function getAnalysisFrequency(options: Record<string, string> = {}): Promise<FrequencyResult[] | null> {
    const params = new URLSearchParams(options)
    return call(() => request<FrequencyResult[]>(`/analysis/frequency?${params}`))
  }

  function getAnalysisMorphology(options: Record<string, string> = {}): Promise<MorphologyResult[] | null> {
    const params = new URLSearchParams(options)
    return call(() => request<MorphologyResult[]>(`/analysis/morphology?${params}`))
  }

  function getAnalysisSemanticDomain(domain: string): Promise<unknown[] | null> {
    return call(() => request<unknown[]>(`/analysis/semantic-domain/${encodeURIComponent(domain)}`))
  }

  // Provenance — resolves null on 404 without setting global error (missing provenance is normal)
  async function getProvenance(entityType: string, entityId: string): Promise<ProvenanceData | null> {
    try {
      const response = await fetch(`${BASE_URL}/provenance/${encodeURIComponent(entityType)}/${encodeURIComponent(entityId)}`, {
        headers: { 'Content-Type': 'application/json' },
      })
      if (response.status === 404) return null
      if (!response.ok) return null
      return response.json() as Promise<ProvenanceData>
    } catch {
      return null
    }
  }

  return {
    loading,
    error,
    DEFAULT_TRANSLATION,
    getProvenance,
    getVerse,
    getChapter,
    searchText,
    getTopics,
    getTopic,
    searchTopics,
    getPlans,
    getPlan,
    getBooks,
    getWordDetail,
    discoverConcepts,
    getSemanticDomains,
    getDomainWords,
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
    getConceptFeedbackSummary,
    getAudioResource,
    mobileSync,
    searchStrongs,
    semanticSearch,
    compareTranslations,
    createNote,
    getNotes,
    deleteNote,
    createCollection,
    getCollections,
    addToCollection,
    getCollectionItems,
    deleteCollection,
    createShare,
    getShare,
    exportVerse,
    getWordExplanation,
    getGenreShifts,
    getPassages,
    getAnalysisFrequency,
    getAnalysisMorphology,
    getAnalysisSemanticDomain,
  }
}
