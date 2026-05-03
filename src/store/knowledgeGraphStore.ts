import { create } from 'zustand'
import type { Concept, ConceptEdge, ReviewRecord, UserAnnotation, ProcessChain, ProcessState } from '../types'
import { useToastStore } from './toastStore'

const GS_URL = 'http://localhost:4321'

export interface ProjectInfo {
  id: string
  name: string
  sourceDir: string
  createdAt: string
  conceptCount: number
  edgeCount: number
}

interface KnowledgeGraphStore {
  concepts: Concept[]
  edges: ConceptEdge[]
  chains: ProcessChain[]
  selectedConcept: Concept | null
  reviewRecords: Map<string, ReviewRecord>
  annotations: UserAnnotation[]
  isLoading: boolean
  loadingProgress: number
  error: string | null
  viewMode: 'explore' | 'review'

  conceptPanelMode: boolean
  linkMode: boolean
  linkSource: string | null

  projects: ProjectInfo[]
  activeProjectId: string | null
  fetchProjects: () => Promise<void>
  loadProjectGraph: (projectId: string) => Promise<void>

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
    relationType: 'leads_to' | 'depends_on' | 'related'
    metadataStatus?: 'ai-generated' | 'draft'
  }) => Concept
  removeConcept: (conceptId: string) => void
  updateProcessState: (conceptId: string, state: Partial<ProcessState>) => void

  setConceptPanelMode: (mode: boolean) => void
  setLinkMode: (mode: boolean) => void
  setLinkSource: (source: string | null) => void
  updateConceptContent: (conceptId: string, content: string) => void
  persistToServer: () => Promise<void>
}

export const useKnowledgeGraphStore = create<KnowledgeGraphStore>()(
  (set, get) => ({
    concepts: [],
    edges: [],
    chains: [],
    selectedConcept: null,
    reviewRecords: new Map(),
    annotations: [],
    isLoading: false,
    loadingProgress: 0,
    error: null,
    viewMode: 'explore',
    conceptPanelMode: true,
    linkMode: false,
    linkSource: null,
    projects: [],
    activeProjectId: null,

    fetchProjects: async () => {
      try {
        const resp = await fetch(`${GS_URL}/api/projects`)
        if (resp.ok) {
          const data = await resp.json()
          set({ projects: data.projects || [] })
        }
      } catch {
        // GS not available
      }
    },

    loadProjectGraph: async (projectId: string) => {
      set({ isLoading: true, error: null, activeProjectId: projectId })
      set({ loadingProgress: 10 })
      try {
        const resp = await fetch(`${GS_URL}/api/projects/${projectId}/graph`)
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
        set({ loadingProgress: 60 })
        const data = await resp.json()
        set(state => {
          const currentSelectedId = state.selectedConcept?.id
          const currentContent = state.selectedConcept?.content
          const newConcepts = (data.concepts || []).map((c: any) => {
            if (c.id === currentSelectedId && currentContent) {
              return { ...c, content: currentContent }
            }
            return c
          })
          const preserved = currentSelectedId
            ? newConcepts.find((c: any) => c.id === currentSelectedId) ?? null
            : null
          return {
            concepts: newConcepts,
            edges: data.edges || [],
            selectedConcept: preserved,
            isLoading: false,
            loadingProgress: 100,
          }
        })
      } catch (e: any) {
        set({ error: e.message, isLoading: false, loadingProgress: 0 })
      }
    },

    loadGraph: async () => {
      const state = get()
      if (state.concepts.length > 0) return
      if (state.activeProjectId) {
        await state.loadProjectGraph(state.activeProjectId)
        // Restore selected concept after HMR reload
        try {
          const storedId = sessionStorage.getItem('magic-memory-selected-concept')
          if (storedId) {
            const concept = get().concepts.find(c => c.id === storedId)
            if (concept) {
              get().selectConcept(concept)
              const storedContent = sessionStorage.getItem('magic-memory-concept-content')
              if (storedContent) get().updateConceptContent(concept.id, storedContent)
            }
          }
        } catch {}
        return
      }
      await state.fetchProjects()
      const { projects } = get()
      if (projects.length > 0) {
        await get().loadProjectGraph(projects[0].id)
        try {
          const storedId = sessionStorage.getItem('magic-memory-selected-concept')
          if (storedId) {
            const concept = get().concepts.find(c => c.id === storedId)
            if (concept) {
              get().selectConcept(concept)
              const storedContent = sessionStorage.getItem('magic-memory-concept-content')
              if (storedContent) get().updateConceptContent(concept.id, storedContent)
            }
          }
        } catch {}
        return
      }
      set({ isLoading: false })
    },

    selectConcept: (concept) => {
      set({ selectedConcept: concept })
      try {
        if (concept) {
          sessionStorage.setItem('magic-memory-selected-concept', concept.id)
        } else {
          sessionStorage.removeItem('magic-memory-selected-concept')
          sessionStorage.removeItem('magic-memory-concept-content')
        }
      } catch {}
    },

    updateConceptContent: (conceptId, content) => {
      set(state => ({
        concepts: state.concepts.map(c =>
          c.id === conceptId ? { ...c, content } : c
        ),
        selectedConcept: state.selectedConcept?.id === conceptId
          ? { ...state.selectedConcept, content }
          : state.selectedConcept,
      }))
      try {
        if (content) sessionStorage.setItem('magic-memory-concept-content', content)
      } catch {}
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
        status: 'new' as const,
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
        status: newInterval > 21 ? 'mastered' : 'review',
      }

      const newRecords = new Map(reviewRecords)
      newRecords.set(conceptId, updated)
      set({ reviewRecords: newRecords })
    },

    addAnnotation: (annotation) => {
      const newAnnotation: UserAnnotation = {
        ...annotation,
        id: `ann_${Date.now()}`,
        createdAt: new Date(),
      }
      set(state => ({
        annotations: [...state.annotations, newAnnotation],
      }))
    },

    persistToServer: async () => {
      const { activeProjectId, concepts, edges } = get()
      if (!activeProjectId) return
      try {
        const resp = await fetch(`${GS_URL}/api/projects/${activeProjectId}/graph`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ concepts, edges }),
        })
        if (resp.status === 404) {
          // Project not yet registered on GS — register it with current data
          const regResp = await fetch(`${GS_URL}/api/projects`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              id: activeProjectId,
              name: activeProjectId,
              sourceDir: '',
              concepts,
              edges,
            }),
          })
          if (!regResp.ok) {
            useToastStore.getState().show('持久化失败：注册项目出错', 'error')
            return
          }
        } else if (!resp.ok) {
          useToastStore.getState().show('持久化失败：保存数据出错', 'error')
          return
        }
        useToastStore.getState().show('数据已保存', 'success')
      } catch (e) {
        console.warn('[KG] persistToServer failed:', e)
        useToastStore.getState().show('持久化失败：Graph Server 未运行', 'error')
      }
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
        concepts: [...state.concepts, concept],
      }))
      get().persistToServer()
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
        edges: [...state.edges, edge],
      }))
      get().persistToServer()
    },

    createConceptWithEdges: (source, input) => {
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
        path: `./docs/user/${Date.now()}-${input.title.toLowerCase().replace(/\s+/g, '-')}.md`,
        tags: [source.category.toLowerCase()],
        lastModified: new Date(),
        metadata: { status: input.metadataStatus || 'draft' },
      }
      set(state => ({
        concepts: [...state.concepts, concept],
      }))

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

      get().persistToServer()
      return concept
    },

    setConceptPanelMode: (mode) => {
      set({ conceptPanelMode: mode })
    },

    setLinkMode: (mode) => {
      set({ linkMode: mode })
    },

    setLinkSource: (source) => {
      set({ linkSource: source })
    },

    removeConcept: (conceptId) => {
      set(state => ({
        concepts: state.concepts.filter(c => c.id !== conceptId),
        edges: state.edges.filter(e => e.source !== conceptId && e.target !== conceptId),
        selectedConcept: state.selectedConcept?.id === conceptId ? null : state.selectedConcept,
      }))
      const newRecords = new Map(get().reviewRecords)
      newRecords.delete(conceptId)
      set({ reviewRecords: newRecords })
      get().persistToServer()
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
    },
  }),
)

// Debug: track selectedConcept changes
let prevScId: string | undefined | null = undefined
useKnowledgeGraphStore.subscribe((state: any) => {
  const id = state.selectedConcept?.id
  if (id !== prevScId) {
    console.log('[SC]', prevScId || 'null', '->', id || 'null')
    prevScId = id
  }
})
