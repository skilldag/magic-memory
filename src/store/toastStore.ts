import { create } from 'zustand'

type ToastType = 'success' | 'error'

interface ToastItem {
  id: number
  message: string
  type: ToastType
}

interface ToastStore {
  toasts: ToastItem[]
  show: (message: string, type: ToastType) => void
  dismiss: (id: number) => void
}

let nextId = 1

export const useToastStore = create<ToastStore>((set, get) => ({
  toasts: [],

  show: (message, type) => {
    const id = nextId++
    set(state => ({
      toasts: [...state.toasts, { id, message, type }],
    }))
    setTimeout(() => {
      get().dismiss(id)
    }, 3000)
  },

  dismiss: (id) => {
    set(state => ({
      toasts: state.toasts.filter(t => t.id !== id),
    }))
  },
}))
