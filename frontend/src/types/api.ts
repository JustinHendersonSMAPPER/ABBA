/** Backend VerseResponse shape (from /api/v1/verses/{translation_id}/{book_id}/...) */
export interface VerseResponse {
  reference: string
  book_name: string
  chapter: number
  verse: number
  text: string
  translation_id: string
  words?: WordData[] | null
  richness_flags?: RichnessFlag[] | null
  cultural_context?: CulturalNote[] | null
  dictionary_context?: DictionaryContextItem[] | null
  cross_references?: CrossRef[] | null
  passage_info?: PassageInfo | null
  literary_structures?: LiteraryStructure[] | null
  concepts?: Record<string, unknown>[] | null
  surrounding_context?: { previous_verse?: string | null; next_verse?: string | null } | null
  speaker?: Record<string, unknown> | null
  genre?: string | null
  is_descriptive?: boolean | null
  parallel_passages?: Record<string, unknown>[] | null
  manuscript_variants?: ManuscriptVariant[] | null
  syntax_tree?: SyntaxTree | null
  discourse_units?: DiscourseUnit[] | null
  semantic_domains?: SemanticDomain[] | null
}

export interface RichnessFlag {
  richness?: number
  explanation: string
}

export interface CulturalNote {
  text: string
  category?: string
}

/** A public-domain dictionary entry that cites this verse (Tier-A historical context) */
export interface DictionaryContextItem {
  entry_id: number
  headword: string
  article: string
  source: string
  provenance_entity_id: string
  match_method: string
}

export interface CrossRef {
  book_id?: number
  book_name?: string
  chapter?: number
  verse?: number
  label?: string
  note?: string
  id?: number
  text?: string
  confidence?: number
}

/** Public data-coverage + provenance snapshot (GET /stats) */
export interface DataStats {
  translations: number
  verses: number
  cross_reference_candidates: number
  cross_references: number
  cross_references_explained: number
  cross_references_unexplained: number
  explained_coverage_pct: number
  cross_references_by_tier: Record<string, number>
  cross_references_by_source: Record<string, number>
  avg_confidence: number | null
  provenance_records: number
  dictionary_entries: number
}

/** Core verse data returned by the verse endpoint */
export interface VerseData {
  book?: string
  chapter?: number
  verse?: number
  text?: string
  words?: WordData[]
  richness_flags?: string[]
  parallel_translations?: TranslationEntry[]
  genre?: string
  literary_structures?: LiteraryStructure[]
  semantic_domains?: SemanticDomain[]
  syntax_tree?: SyntaxTree | null
  discourse_units?: DiscourseUnit[]
  manuscript_variants?: ManuscriptVariant[]
  primary_concept?: string
}

export interface WordData {
  word_num?: number
  original_text?: string
  transliteration?: string
  english_gloss?: string
  strongs_number?: string
  morphology_code?: string
  morphology_description?: string
  part_of_speech?: string
  language?: string
  /** legacy / UI-only fields kept for TranslationLens display */
  text?: string
}

export interface TranslationEntry {
  name: string
  text: string
  translation_id?: string
}

export interface LiteraryStructure {
  type: string
  description?: string
}

export interface SemanticDomain {
  domain_code: string
  domain_name: string
  name?: string
  description?: string
  child_count?: number
}

export interface SyntaxTree {
  root_nodes: SyntaxNode[]
}

export interface SyntaxNode {
  node_id: string
  label?: string
  children?: SyntaxNode[]
  text?: string
}

export interface DiscourseUnit {
  discourse_id: string
  discourse_type: string
  function_label?: string
  start_chapter: number
  start_verse: number
  end_chapter: number
  end_verse: number
  description?: string
  relation_to_context?: string
  prominence: number
}

export interface ManuscriptVariant {
  id?: number
  variant_id: string
  variant_type: string
  significance: string
  base_text?: string
  variant_text?: string
  explanation?: string
  manuscripts?: string
}

export interface BookInfo {
  book_id: number
  name: string
  common_name?: string
  testament: string
  chapter_count: number
  primary_genre?: string
  secondary_genres?: string[]
  author_traditional?: string
  date_range?: string
  original_audience?: string
  literary_features?: string[]
  reading_context?: string
  canonical_section?: string
}

export interface CrossReference {
  id?: number
  book: string
  chapter: number
  verse: number
  label?: string
  note?: string
}

export interface ContextData {
  cultural?: Array<{ text: string } | string>
  historical?: string
}

export interface AudioResource {
  url?: string
  audio_url?: string
  title?: string
  reader?: string
}

export interface ShareData {
  token: string
  share_type: string
  title: string
  content: Record<string, unknown>
  created_at?: string
}

export interface SearchResult {
  reference?: string
  book_id?: number
  book_name?: string
  chapter?: number
  verse?: number
  text?: string
  snippet?: string
  match_type?: string
  explanation?: string
  score?: number
  translation_id?: string
}

export interface ProvenanceData {
  entity_type: string
  entity_id: string
  source: string
  source_detail?: string
  trust_tier: 'A' | 'B' | 'C'
  trust_rationale: string
  generated_by?: string
  grounding: Record<string, unknown>
  confidence?: number
  pipeline_version: string
}

export interface TopicSummary {
  id: string
  slug?: string
  name: string
  description?: string
  category?: string
  verse_count?: number
  icon?: string
}

export interface TopicDetail extends TopicSummary {
  steps?: StudyStep[]
}

export interface StudyStep {
  title: string
  summary: string
  book?: string
  chapter?: number
  verse?: number
  reference?: string
}

export interface ReadingPlan {
  id: string
  slug?: string
  name: string
  description?: string
  duration?: number
  category?: string
}

export interface ReadingPlanDetail extends ReadingPlan {
  entries?: PlanEntry[]
}

export interface PlanEntry {
  day: number
  title?: string
  readings?: PlanReading[]
  reflection?: string
}

export interface PlanReading {
  book: string
  chapter: number
  verse?: number
  reference?: string
}

export interface LexiconEntry {
  strongs_number?: string
  original_word?: string
  transliteration?: string
  part_of_speech?: string
  gloss?: string
  definition?: string
  language?: string
  source?: string
}

export interface WordExplanation {
  explanation: string
  strongs_number?: string
}

export interface WordDomainResult {
  strongs_number: string
  word?: string
  original?: string
  transliteration?: string
  gloss?: string
  short_definition?: string
  domains?: SemanticDomain[]
  domain_code?: string
  domain_name?: string
}

export interface CollectionInfo {
  id: string
  name: string
  description?: string
  item_count?: number
}

export interface CollectionItem {
  book_id: string
  book_name?: string
  chapter: number
  verse: number
  text?: string
  note?: string
}

export interface ConceptDiscoveryResult {
  matched_concepts?: ConceptMatch[]
  matched_life_topics?: TopicSummary[]
  suggested_searches?: string[]
}

export interface ConceptMatch {
  name: string
  description?: string
  verse_count?: number
}

export interface ConceptGraph {
  center_concept: string
  nodes: GraphNode[]
  edges?: GraphEdge[]
  relationships: GraphRelationship[]
}

export interface GraphRelationship {
  source_concept: string
  target_concept: string
  relationship_type: string
  weight?: number
}

export interface GraphNode {
  id: string
  name: string
  label?: string
  type?: string
  is_center?: boolean
}

export interface GraphEdge {
  source: string
  target: string
  label?: string
  weight?: number
}

export interface Contribution {
  id: string
  contribution_type: string
  book_id?: string
  chapter?: number
  verse?: number
  content: string
  source?: string
  status: string
}

export interface ConceptProposal {
  id: string
  concept_name: string
  description?: string
  strongs_numbers?: string
  status: string
}

export interface PassageInfo {
  title: string
  start_verse: number
  end_verse: number
}

export interface GenreShift {
  chapter: number
  verse: number
  from_genre: string
  to_genre: string
  description?: string
}

export interface ChapterData {
  verses: Array<{
    number: number
    text: string
    words?: WordData[]
    richness_flags?: string[]
  }>
  genre?: string
  literary_structures?: LiteraryStructure[]
}

export interface FrequencyResult {
  strongs_number?: string
  strongs?: string
  word?: string
  original?: string
  gloss?: string
  short_definition?: string
  frequency?: number
  count?: number
}

export interface MorphologyResult {
  pattern?: string
  morphology_code?: string
  code?: string
  count?: number
  description?: string
  label?: string
  example?: string
}

export interface TranslationInfo {
  id: string
  name: string
  language: string
  english_name?: string | null
}
