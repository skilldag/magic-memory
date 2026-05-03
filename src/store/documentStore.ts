import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { Document } from '../types'

interface DocumentStore {
  documents: Document[]
  selectedDocument: Document | null
  isLoading: boolean
  error: string | null

  loadDocuments: () => Promise<void>
  selectDocument: (document: Document) => void
  addDocument: (document: Document) => void
  updateDocument: (id: string, updates: Partial<Document>) => void
  searchDocuments: (query: string) => Document[]
  filterByLevel: (level: number) => Document[]
  filterByCategory: (category: string) => Document[]
}

export const useDocumentStore = create<DocumentStore>()(
  persist(
    (set, get) => ({
      documents: [],
      selectedDocument: null,
      isLoading: false,
      error: null,

      loadDocuments: async () => {
        set({ isLoading: true, error: null })
        try {
          const response = await fetch('/api/documents')
          if (!response.ok) throw new Error('Failed to load documents')
          const documents = await response.json()
          set({ documents, isLoading: false })
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : 'Unknown error',
            isLoading: false,
          })
        }
      },

      selectDocument: (document) => {
        set({ selectedDocument: document })
      },

      addDocument: (document) => {
        set((state) => ({
          documents: state.documents.some(d => d.id === document.id)
            ? state.documents.map(d => d.id === document.id ? { ...d, ...document } : d)
            : [...state.documents, document],
        }))
      },

      updateDocument: (id, updates) => {
        set((state) => ({
          documents: state.documents.map((doc) =>
            doc.id === id ? { ...doc, ...updates } : doc
          ),
          selectedDocument:
            state.selectedDocument?.id === id
              ? { ...state.selectedDocument, ...updates }
              : state.selectedDocument,
        }))
      },

      searchDocuments: (query) => {
        const { documents } = get()
        const lowerQuery = query.toLowerCase()
        return documents.filter(
          (doc) =>
            doc.title.toLowerCase().includes(lowerQuery) ||
            doc.content.toLowerCase().includes(lowerQuery) ||
            doc.tags.some((tag) => tag.toLowerCase().includes(lowerQuery))
        )
      },

      filterByLevel: (level) => {
        const { documents } = get()
        return documents.filter((doc) => doc.level === level)
      },

      filterByCategory: (category) => {
        const { documents } = get()
        return documents.filter((doc) => doc.category === category)
      },
    }),
    {
      name: 'document-storage',
      partialize: (state) => ({
        documents: state.documents,
        selectedDocument: state.selectedDocument,
      }),
    }
  )
)
