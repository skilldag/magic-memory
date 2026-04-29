import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { Concept, ConceptEdge, ReviewRecord, UserAnnotation, ProcessChain, ProcessState, UserQuestion, CanvasHistoryItem } from '../types'
import { getMockGraphData } from '../data/mockGraphData'

interface KnowledgeGraphStore {
  concepts: Concept[]
  edges: ConceptEdge[]
  chains: ProcessChain[]
  selectedConcept: Concept | null
  reviewRecords: Map<string, ReviewRecord>
  annotations: UserAnnotation[]
  isLoading: boolean
  error: string | null
  viewMode: 'explore' | 'review'
  
  // 骨架填充
  questions: UserQuestion[]
  canvasHistory: CanvasHistoryItem[]
  skeletonCompleted: string[]
  conceptPanelMode: boolean
  
  loadGraph: () => Promise<void>
  selectConcept: (concept: Concept) => void
  getRelated: (conceptId: string) => Concept[]
  startReview: (conceptId: string, quality: number) => void
  addAnnotation: (annotation: Omit<UserAnnotation, 'id' | 'createdAt'>) => void
  setViewMode: (mode: 'explore' | 'review') => void
  addConcept: (concept: Omit<Concept, 'id' | 'lastModified'> & { id?: string }) => Concept
  addEdge: (sourceId: string, targetId: string, type: ConceptEdge['type'], label?: string) => void
  createConceptWithEdges: (source: Concept, input: {
    title: string
    problem?: string
    gap_anticipate?: string
    content?: string
    relationType: 'leads_to' | 'depends_on' | 'related'
    metadataStatus?: 'ai-generated' | 'draft'
  }) => Concept
  updateProcessState: (conceptId: string, state: Partial<ProcessState>) => void

  // 骨架填充 actions
  addQuestion: (q: Omit<UserQuestion, 'id' | 'createdAt'>) => void
  setConceptPanelMode: (mode: boolean) => void
  markSkeletonCompleted: (conceptId: string) => void
  pushHistory: (item: CanvasHistoryItem) => void
  popHistory: () => CanvasHistoryItem | undefined
}

export const useKnowledgeGraphStore = create<KnowledgeGraphStore>()(
  persist(
    (set, get) => ({
      concepts: [],
      edges: [],
      chains: [],
      selectedConcept: null,
      reviewRecords: new Map(),
      annotations: [],
      isLoading: false,
      error: null,
      viewMode: 'explore',
      questions: [],
      canvasHistory: [],
      skeletonCompleted: [],
      conceptPanelMode: true,
      
loadGraph: async () => {
    set({ isLoading: true, error: null })
    try {
      const data = getMockGraphData()
      await new Promise(resolve => setTimeout(resolve, 300))
      set({
        concepts: data.concepts,
        edges: data.edges,
        chains: data.chains ?? [],
        isLoading: false
      })
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : 'Unknown error',
        isLoading: false
      })
    }
  },
      
      selectConcept: (concept) => {
        set({ selectedConcept: concept })
      },
      
      getRelated: (conceptId) => {
        const { concepts, edges } = get()
        const relatedIds = edges
          .filter(e => e.source === conceptId || e.target === conceptId)
          .map(e => e.source === conceptId ? e.target : e.source)
        return concepts.filter(c => relatedIds.includes(c.id))
      },
      
      startReview: (conceptId, quality) => {
        const { reviewRecords, concepts } = get()
        const concept = concepts.find(c => c.id === conceptId)
        if (!concept) return
        
        const existing = reviewRecords.get(conceptId) || {
          concept_id: conceptId,
          last_reviewed: new Date(),
          next_review: new Date(),
          ease_factor: 2.5,
          interval: 0,
          review_count: 0,
          status: 'new' as const
        }
        
        let newInterval = existing.interval
        let newEaseFactor = existing.ease_factor
        
        if (quality < 3) {
          newInterval = 1
        } else if (existing.interval === 0) {
          newInterval = 1
        } else if (existing.interval === 1) {
          newInterval = 6
        } else {
          newInterval = Math.round(existing.interval * existing.ease_factor)
        }
        
        newEaseFactor = existing.ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        newEaseFactor = Math.max(1.3, newEaseFactor)
        
        const nextReview = new Date()
        nextReview.setDate(nextReview.getDate() + newInterval)
        
        const updated: ReviewRecord = {
          concept_id: conceptId,
          last_reviewed: new Date(),
          next_review: nextReview,
          ease_factor: newEaseFactor,
          interval: newInterval,
          review_count: existing.review_count + 1,
          status: newInterval > 21 ? 'mastered' : 'review'
        }
        
        const newRecords = new Map(reviewRecords)
        newRecords.set(conceptId, updated)
        set({ reviewRecords: newRecords })
      },
      
      addAnnotation: (annotation) => {
        const newAnnotation: UserAnnotation = {
          ...annotation,
          id: `ann_${Date.now()}`,
          createdAt: new Date()
        }
        set(state => ({ 
          annotations: [...state.annotations, newAnnotation] 
        }))
      },
      
      setViewMode: (mode) => {
        set({ viewMode: mode })
      },

      addConcept: (input) => {
        const id = input.id || `user_${Date.now()}`
        const concept: Concept = {
          ...input,
          id,
          lastModified: new Date(),
        }
        set(state => ({
          concepts: [...state.concepts, concept]
        }))
        return concept
      },

      addEdge: (sourceId, targetId, type, label) => {
        const edge: ConceptEdge = {
          id: `e_${sourceId}_${targetId}`,
          source: sourceId,
          target: targetId,
          type,
          label,
        }
        set(state => ({
          edges: [...state.edges, edge]
        }))
      },

      createConceptWithEdges: (source, input) => {
        const { concepts } = get()
        const id = `user_${Date.now()}`
        const concept: Concept = {
          id,
          title: input.title,
          level: source.level,
          category: source.category,
          problem: input.problem || `与「${source.title}」关联的概念`,
          gap_anticipate: input.gap_anticipate,
          depends_on: input.relationType === 'depends_on' ? [source.id] : [],
          leads_to: input.relationType === 'leads_to' ? [source.id] : [],
          related: input.relationType === 'related' ? [source.id] : [],
          content: input.content || `# ${input.title}\n\n## 问题\n${input.problem || `与「${source.title}」关联`}\n\n## 来源\n通过探索模式关联添加。`,
          path: `./docs/user/${input.title.toLowerCase().replace(/\s+/g, '-')}.md`,
          tags: [source.category.toLowerCase()],
          lastModified: new Date(),
          metadata: { status: input.metadataStatus || 'draft' },
        }
        set(state => ({
          concepts: [...state.concepts, concept]
        }))

        // 根据关系类型添加边
        const { addEdge: addEdgeFn } = get()
        if (input.relationType === 'leads_to') {
          addEdgeFn(source.id, concept.id, 'leads_to')
          addEdgeFn(concept.id, source.id, 'depends_on')
        } else if (input.relationType === 'depends_on') {
          addEdgeFn(concept.id, source.id, 'leads_to')
          addEdgeFn(source.id, concept.id, 'depends_on')
        } else {
          addEdgeFn(source.id, concept.id, 'related')
        }

        return concept
      },

      addQuestion: (q) => {
        const question: UserQuestion = {
          ...q,
          id: `q_${Date.now()}`,
          createdAt: new Date(),
        }
        set(state => ({ questions: [...state.questions, question] }))
      },

      setConceptPanelMode: (mode) => {
        set({ conceptPanelMode: mode })
      },

      markSkeletonCompleted: (conceptId) => {
        set(state => ({
          skeletonCompleted: [...state.skeletonCompleted, conceptId]
        }))
      },

      pushHistory: (item) => {
        set(state => ({ canvasHistory: [...state.canvasHistory, item] }))
      },

      popHistory: () => {
        const { canvasHistory } = get()
        if (canvasHistory.length === 0) return undefined
        const popped = canvasHistory[canvasHistory.length - 1]
        set({ canvasHistory: canvasHistory.slice(0, -1) })
        return popped
      },

      updateProcessState: (conceptId, state) => {
        const { reviewRecords } = get()
        const existing = reviewRecords.get(conceptId)
        const currentProcess = existing?.process_state ?? {
          user_flow: [],
          llm_flow: [],
          gaps: [],
          filled: false,
          compared: false,
        }
        const updated: ReviewRecord = {
          ...(existing ?? {
            concept_id: conceptId,
            last_reviewed: new Date(),
            next_review: new Date(),
            ease_factor: 2.5,
            interval: 0,
            review_count: 0,
            status: 'new' as const,
          }),
          process_state: { ...currentProcess, ...state },
        }
        if (!existing) {
          updated.last_reviewed = new Date()
          updated.next_review = new Date()
        }
        const newRecords = new Map(reviewRecords)
        newRecords.set(conceptId, updated)
        set({ reviewRecords: newRecords })
      }
    }),
    {
      name: 'knowledge-graph-storage',
      partialize: (state) => ({
        reviewRecords: Array.from(state.reviewRecords.entries()),
        annotations: state.annotations,
        questions: state.questions,
        skeletonCompleted: state.skeletonCompleted,
      }),
      merge: (persisted: any, current) => ({
        ...current,
        reviewRecords: new Map(persisted?.reviewRecords || [])
      })
    }
  )
)