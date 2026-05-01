import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { Annotation, AnnotationStats } from '../types'

interface AnnotationStore {
  annotations: Annotation[]
  selectedAnnotation: Annotation | null
  isLoading: boolean
  error: string | null

  loadAnnotations: (documentId: string) => Promise<void>
  addAnnotation: (annotation: Omit<Annotation, 'id' | 'createdAt' | 'updatedAt'>) => void
  updateAnnotation: (id: string, updates: Partial<Annotation>) => void
  deleteAnnotation: (id: string) => void
  selectAnnotation: (annotation: Annotation | null) => void
  getAnnotationsByDocument: (documentId: string) => Annotation[]
  getStats: (documentId: string) => AnnotationStats
  addReply: (annotationId: string, reply: Omit<AnnotationReply, 'id' | 'createdAt'>) => void
}

export const useAnnotationStore = create<AnnotationStore>()(
  persist(
    (set, get) => ({
      annotations: [],
      selectedAnnotation: null,
      isLoading: false,
      error: null,

      loadAnnotations: async (documentId) => {
        set({ isLoading: true, error: null })
        try {
          const response = await fetch(`/api/documents/${documentId}/annotations`)
          if (!response.ok) throw new Error('Failed to load annotations')
          const annotations = await response.json()
          set({ annotations, isLoading: false })
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : 'Unknown error',
            isLoading: false,
          })
        }
      },

      addAnnotation: (annotation) => {
        const newAnnotation: Annotation = {
          ...annotation,
          id: `ann-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
          createdAt: new Date(),
          updatedAt: new Date(),
        }
        set((state) => ({
          annotations: [...state.annotations, newAnnotation],
        }))
      },

      updateAnnotation: (id, updates) => {
        set((state) => ({
          annotations: state.annotations.map((ann) =>
            ann.id === id ? { ...ann, ...updates, updatedAt: new Date() } : ann
          ),
          selectedAnnotation:
            state.selectedAnnotation?.id === id
              ? { ...state.selectedAnnotation, ...updates, updatedAt: new Date() }
              : state.selectedAnnotation,
        }))
      },

      deleteAnnotation: (id) => {
        set((state) => ({
          annotations: state.annotations.filter((ann) => ann.id !== id),
          selectedAnnotation:
            state.selectedAnnotation?.id === id ? null : state.selectedAnnotation,
        }))
      },

      selectAnnotation: (annotation) => {
        set({ selectedAnnotation: annotation })
      },

      getAnnotationsByDocument: (documentId) => {
        const { annotations } = get()
        return annotations.filter((ann) => ann.documentId === documentId)
      },

      getStats: (documentId) => {
        const { annotations } = get()
        const docAnnotations = annotations.filter((ann) => ann.documentId === documentId)

        const byType: Record<string, number> = {}
        const byStatus: Record<string, number> = {}

        docAnnotations.forEach((ann) => {
          byType[ann.type] = (byType[ann.type] || 0) + 1
          byStatus[ann.status] = (byStatus[ann.status] || 0) + 1
        })

        const now = new Date()
        const oneWeekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000)
        const recent = docAnnotations.filter((ann) => ann.createdAt > oneWeekAgo).length

        return {
          total: docAnnotations.length,
          byType,
          byStatus,
          recent,
        }
      },

      addReply: (annotationId, reply) => {
        const newReply = {
          ...reply,
          id: `reply-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
          createdAt: new Date(),
        }
        set((state) => ({
          annotations: state.annotations.map((ann) =>
            ann.id === annotationId
              ? {
                  ...ann,
                  replies: [...(ann.replies || []), newReply],
                  updatedAt: new Date(),
                }
              : ann
          ),
        }))
      },
    }),
    {
      name: 'annotation-storage',
      partialize: (state) => ({
        annotations: state.annotations,
      }),
    }
  )
)
