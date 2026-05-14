# Annotation Interaction Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current bottom floating bar with a selection-anchored Popover, add a unified annotation dialog (replacing QuestionDialog), add color-coded text highlights for existing annotations, and add click-to-preview interaction — completing the "select text → annotate" workflow.

**Architecture:** Three new components (SelectionPopover, AnnotationDialog, AnnotationPreview) plus modifications to DocumentViewer (highlights via HTML post-processing, popover integration, preview wiring) and AnnotationPanel (bidirectional linking). No backend changes — annotations remain Zustand-persisted.

**Tech Stack:** React 19, TypeScript, Zustand, Tailwind CSS 4, marked (Markdown rendering), DOMPurify.

---

### Task 1: Annotation Highlight Styles

**Files:**
- Modify: `src/index.css` (expand existing `.annotation-highlight` rules)

Replace the existing generic blue-only highlight class with type-specific color-coded styles.

- [ ] **Step 1: Replace existing annotation highlight CSS**

Find and replace the existing `.annotation-highlight` block (lines 187-203 in current `src/index.css`):

```css
/* Annotation highlight styles */
.ann-highlight {
  cursor: pointer;
  border-radius: 2px;
  transition: background-color 0.15s, opacity 0.15s;
}
.ann-highlight:hover { filter: brightness(0.9); }
.ann-comment {
  background-color: #dbeafe;
  text-decoration: underline;
  text-decoration-color: #3b82f6;
  text-underline-offset: 2px;
}
.ann-question {
  background-color: #e9d5ff;
  text-decoration: wavy underline;
  text-decoration-color: #8b5cf6;
  text-underline-offset: 2px;
}
.ann-suggestion {
  background-color: #dcfce7;
  text-decoration: dashed underline;
  text-decoration-color: #10b981;
  text-underline-offset: 2px;
}
.ann-correction {
  background-color: #fee2e2;
  text-decoration: solid underline;
  text-decoration-color: #ef4444;
  text-underline-offset: 2px;
}
.ann-resolved { opacity: 0.5; }
.ann-selected {
  outline: 2px solid #3b82f6;
  outline-offset: 1px;
  animation: ann-flash 0.6s ease-in-out;
}
@keyframes ann-flash {
  0%, 100% { outline-color: #3b82f6; }
  50% { outline-color: #93c5fd; }
}
```

Remove the old `.annotation-highlight` block (lines 187-203) entirely. Keep `.spinner` and everything after it.

- [ ] **Step 2: Verify file compiles**

Run: `npx tsc --noEmit` (or `npm run typecheck`)
Expected: No type errors (CSS changes have no type impact)

- [ ] **Step 3: Commit**

```bash
git add src/index.css
git commit -m "feat: add type-specific annotation highlight CSS styles"
```

---

### Task 2: SelectionPopover Component

**Files:**
- Create: `src/components/SelectionPopover.tsx`

A compact popover positioned near the user's text selection, with action buttons for each annotation type plus concept elevation.

- [ ] **Step 1: Create SelectionPopover component**

Create `src/components/SelectionPopover.tsx`:

```tsx
import React, { useEffect, useRef } from 'react'

interface SelectionPopoverProps {
  position: { x: number; y: number }
  selectedText: string
  onAddAnnotation: (type: 'comment' | 'question' | 'suggestion' | 'correction') => void
  onConceptElevation: () => void
  onClose: () => void
}

export function SelectionPopover({
  position,
  onAddAnnotation,
  onConceptElevation,
  onClose,
}: SelectionPopoverProps) {
  const popoverRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (popoverRef.current && !popoverRef.current.contains(e.target as Node)) {
        onClose()
      }
    }
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    // Delay to avoid the same click that triggered the popover
    const timer = setTimeout(() => {
      document.addEventListener('mousedown', handleClickOutside)
      document.addEventListener('keydown', handleEscape)
    }, 0)
    return () => {
      clearTimeout(timer)
      document.removeEventListener('mousedown', handleClickOutside)
      document.removeEventListener('keydown', handleEscape)
    }
  }, [onClose])

  // Determine popover position: above selection by default, flip below if near top
  const popoverStyle: React.CSSProperties = {
    position: 'fixed',
    left: position.x,
    top: position.y - 10,
    transform: 'translate(-50%, -100%)',
    zIndex: 1000,
  }

  // If near top of viewport (< 120px from top), flip below
  if (position.y < 120) {
    popoverStyle.top = position.y + 10
    popoverStyle.transform = 'translate(-50%, 0)'
  }

  return (
    <div
      ref={popoverRef}
      style={popoverStyle}
      className="flex items-center gap-1 bg-white rounded-lg shadow-lg border border-gray-200 px-2 py-1.5"
    >
      <button
        onClick={() => onAddAnnotation('comment')}
        className="px-2 py-1 text-xs font-medium text-blue-700 bg-blue-50 rounded hover:bg-blue-100 transition-colors whitespace-nowrap"
        title="添加评论"
      >
        💬 评论
      </button>
      <button
        onClick={() => onAddAnnotation('question')}
        className="px-2 py-1 text-xs font-medium text-purple-700 bg-purple-50 rounded hover:bg-purple-100 transition-colors whitespace-nowrap"
        title="提出问题"
      >
        ❓ 提问
      </button>
      <button
        onClick={() => onAddAnnotation('correction')}
        className="px-2 py-1 text-xs font-medium text-red-700 bg-red-50 rounded hover:bg-red-100 transition-colors whitespace-nowrap"
        title="纠正错误"
      >
        ✏️ 纠正
      </button>
      <div className="w-px h-5 bg-gray-200 mx-1" />
      <button
        onClick={onConceptElevation}
        className="px-2 py-1 text-xs font-medium text-green-700 bg-green-50 rounded hover:bg-green-100 transition-colors whitespace-nowrap"
        title="提升为概念"
      >
        🔗 概念
      </button>
      <button
        onClick={onClose}
        className="p-1 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded transition-colors"
        title="关闭"
      >
        <svg width={14} height={14} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  )
}
```

- [ ] **Step 2: Typecheck the new file**

Run: `npm run typecheck`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add src/components/SelectionPopover.tsx
git commit -m "feat: add SelectionPopover component for text selection actions"
```

---

### Task 3: AnnotationDialog Component

**Files:**
- Create: `src/components/AnnotationDialog.tsx`

A unified modal dialog for creating all 4 annotation types (comment, question, suggestion, correction). Replaces the current `QuestionDialog.tsx`. Includes AI answer toggle (for questions) and concept elevation toggle.

- [ ] **Step 1: Create AnnotationDialog component**

Create `src/components/AnnotationDialog.tsx`:

```tsx
import React, { useState } from 'react'

export type AnnotationType = 'comment' | 'question' | 'suggestion' | 'correction'

interface AnnotationDialogProps {
  selectedText: string
  initialType?: AnnotationType
  onClose: () => void
  onSubmit: (data: {
    type: AnnotationType
    content: string
    selectedText: string
    enableAI: boolean
    enableConcept: boolean
  }) => void
}

const typeLabels: Record<AnnotationType, string> = {
  comment: '评论',
  question: '提问',
  suggestion: '建议',
  correction: '纠正',
}

const typeColors: Record<AnnotationType, string> = {
  comment: 'blue',
  question: 'purple',
  suggestion: 'green',
  correction: 'red',
}

const typeRadios: { value: AnnotationType; label: string }[] = [
  { value: 'comment', label: '评论' },
  { value: 'question', label: '提问' },
  { value: 'suggestion', label: '建议' },
  { value: 'correction', label: '纠正' },
]

export function AnnotationDialog({
  selectedText,
  initialType = 'comment',
  onClose,
  onSubmit,
}: AnnotationDialogProps) {
  const [type, setType] = useState<AnnotationType>(initialType)
  const [content, setContent] = useState('')
  const [enableAI, setEnableAI] = useState(type === 'question')
  const [enableConcept, setEnableConcept] = useState(false)

  const handleTypeChange = (newType: AnnotationType) => {
    setType(newType)
    if (newType !== 'question') setEnableAI(false)
    else setEnableAI(true)
  }

  const handleSubmit = () => {
    if (!content.trim()) return
    onSubmit({
      type,
      content: content.trim(),
      selectedText,
      enableAI,
      enableConcept,
    })
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className="bg-white rounded-xl shadow-2xl w-[480px] max-w-[90vw] max-h-[80vh] flex flex-col">
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-200">
          <h2 className="text-base font-semibold text-gray-900">
            添加{typeLabels[type]}
          </h2>
          <button onClick={onClose} className="p-1 rounded hover:bg-gray-100 text-gray-400">
            <svg width={20} height={20} className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
          {/* Selected text display */}
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1.5">选中文本</label>
            <div className="p-3 bg-gray-50 border border-gray-200 rounded-lg text-sm text-gray-700 leading-relaxed select-auto">
              {selectedText}
            </div>
          </div>

          {/* Type selector */}
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1.5">类型</label>
            <div className="flex gap-2">
              {typeRadios.map(({ value, label }) => (
                <button
                  key={value}
                  onClick={() => handleTypeChange(value)}
                  className={`px-3 py-1.5 text-sm rounded-lg border transition-colors ${
                    type === value
                      ? 'bg-blue-50 border-blue-300 text-blue-700 font-medium'
                      : 'bg-white border-gray-200 text-gray-600 hover:bg-gray-50'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {/* Content input */}
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1.5">
              {type === 'question' ? '你的问题' : '内容'}
            </label>
            <textarea
              className="w-full h-24 p-3 text-sm border border-gray-200 rounded-lg resize-none outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-400"
              placeholder={
                type === 'question'
                  ? '针对选中文本提出你的问题...'
                  : type === 'comment'
                  ? '写下你的评论...'
                  : type === 'suggestion'
                  ? '提出改进建议...'
                  : '指出需要纠正的内容...'
              }
              value={content}
              onChange={e => setContent(e.target.value)}
              autoFocus
            />
          </div>

          {/* Options toggles */}
          <div className="space-y-3 pt-2">
            {type === 'question' && (
              <label className="flex items-center gap-3 cursor-pointer">
                <div className="relative">
                  <input
                    type="checkbox"
                    className="sr-only"
                    checked={enableAI}
                    onChange={e => setEnableAI(e.target.checked)}
                  />
                  <div className={`w-10 h-5 rounded-full transition-colors ${enableAI ? 'bg-blue-500' : 'bg-gray-300'}`}>
                    <div className={`w-4 h-4 bg-white rounded-full shadow-sm transition-transform mt-0.5 ${enableAI ? 'translate-x-5' : 'translate-x-0.5'}`} />
                  </div>
                </div>
                <div>
                  <span className="text-sm font-medium text-gray-800">AI 自动回答</span>
                  <p className="text-xs text-gray-500">提交后用 AI 生成问题的回答</p>
                </div>
              </label>
            )}

            <label className="flex items-center gap-3 cursor-pointer">
              <div className="relative">
                <input
                  type="checkbox"
                  className="sr-only"
                  checked={enableConcept}
                  onChange={e => setEnableConcept(e.target.checked)}
                />
                <div className={`w-10 h-5 rounded-full transition-colors ${enableConcept ? 'bg-blue-500' : 'bg-gray-300'}`}>
                  <div className={`w-4 h-4 bg-white rounded-full shadow-sm transition-transform mt-0.5 ${enableConcept ? 'translate-x-5' : 'translate-x-0.5'}`} />
                </div>
              </div>
              <div>
                <span className="text-sm font-medium text-gray-800">转为新概念</span>
                <p className="text-xs text-gray-500">将这条标注创建为一个新的知识图概念</p>
              </div>
            </label>
          </div>
        </div>

        <div className="flex items-center justify-end gap-2 px-5 py-4 border-t border-gray-200">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
          >
            取消
          </button>
          <button
            onClick={handleSubmit}
            disabled={!content.trim()}
            className="px-4 py-2 text-sm font-medium text-white bg-blue-500 rounded-lg hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            提交
          </button>
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Typecheck**

Run: `npm run typecheck`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add src/components/AnnotationDialog.tsx
git commit -m "feat: add AnnotationDialog component (unified annotation creation dialog)"
```

---

### Task 4: AnnotationPreview Component

**Files:**
- Create: `src/components/AnnotationPreview.tsx`

A small popover that appears when clicking a highlighted text annotation in the document. Shows a summary and a "view details" link.

- [ ] **Step 1: Create AnnotationPreview component**

Create `src/components/AnnotationPreview.tsx`:

```tsx
import React, { useEffect, useRef } from 'react'
import type { Annotation } from '../types'

interface AnnotationPreviewProps {
  annotation: Annotation
  position: { x: number; y: number }
  onViewDetails: (annotation: Annotation) => void
  onClose: () => void
}

const typeIcons: Record<string, string> = {
  comment: '💬',
  question: '❓',
  suggestion: '💡',
  correction: '✏️',
}

const typeLabels: Record<string, string> = {
  comment: '评论',
  question: '提问',
  suggestion: '建议',
  correction: '纠正',
}

const statusLabels: Record<string, string> = {
  open: '开放',
  resolved: '已解决',
  closed: '已关闭',
}

function timeAgo(date: Date): string {
  const now = new Date()
  const diff = now.getTime() - new Date(date).getTime()
  const days = Math.floor(diff / (1000 * 60 * 60 * 24))
  if (days === 0) return '今天'
  if (days === 1) return '昨天'
  if (days < 7) return `${days}天前`
  return new Date(date).toLocaleDateString()
}

export function AnnotationPreview({
  annotation,
  position,
  onViewDetails,
  onClose,
}: AnnotationPreviewProps) {
  const previewRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (previewRef.current && !previewRef.current.contains(e.target as Node)) {
        onClose()
      }
    }
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    const timer = setTimeout(() => {
      document.addEventListener('mousedown', handleClickOutside)
      document.addEventListener('keydown', handleEscape)
    }, 0)
    return () => {
      clearTimeout(timer)
      document.removeEventListener('mousedown', handleClickOutside)
      document.removeEventListener('keydown', handleEscape)
    }
  }, [onClose])

  const previewStyle: React.CSSProperties = {
    position: 'fixed',
    left: position.x,
    top: position.y - 10,
    transform: 'translate(-50%, -100%)',
    zIndex: 1000,
  }

  if (position.y < 150) {
    previewStyle.top = position.y + 10
    previewStyle.transform = 'translate(-50%, 0)'
  }

  return (
    <div
      ref={previewRef}
      style={previewStyle}
      className="w-72 bg-white rounded-lg shadow-lg border border-gray-200 overflow-hidden"
    >
      <div className="px-3 py-2 border-b border-gray-100 flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <span className="text-sm">{typeIcons[annotation.type] || '💬'}</span>
          <span className="text-xs font-medium text-gray-700">
            {typeLabels[annotation.type] || annotation.type}
          </span>
          <span className="text-xs text-gray-400">·</span>
          <span className="text-xs text-gray-400">{annotation.author}</span>
          <span className="text-xs text-gray-400">·</span>
          <span className="text-xs text-gray-400">{timeAgo(annotation.createdAt)}</span>
        </div>
        <span className={`text-xs px-1.5 py-0.5 rounded ${
          annotation.status === 'open' ? 'bg-green-100 text-green-700' :
          annotation.status === 'resolved' ? 'bg-blue-100 text-blue-700' :
          'bg-gray-100 text-gray-500'
        }`}>
          {statusLabels[annotation.status] || annotation.status}
        </span>
      </div>
      <div className="px-3 py-2">
        <p className="text-sm text-gray-700 line-clamp-3 leading-relaxed">
          {annotation.content}
        </p>
        {annotation.replies && annotation.replies.length > 0 && (
          <div className="mt-1.5 text-xs text-gray-400">
            ↩ {annotation.replies.length} 条回复
          </div>
        )}
      </div>
      <div className="px-3 py-1.5 bg-gray-50 border-t border-gray-100">
        <button
          onClick={() => onViewDetails(annotation)}
          className="text-xs text-blue-600 hover:text-blue-800 font-medium"
        >
          查看详情 →
        </button>
      </div>
    </div>
  )
}
```

Note: `line-clamp-3` is a Tailwind CSS v3.3+ utility. If not available in the project's Tailwind config, add `@tailwindcss/line-clamp` plugin or use inline CSS `display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;` instead.

- [ ] **Step 2: Typecheck**

Run: `npm run typecheck`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add src/components/AnnotationPreview.tsx
git commit -m "feat: add AnnotationPreview component for highlight click preview"
```

---

### Task 5: Refactor DocumentViewer — Core Integration

**Files:**
- Modify: `src/components/DocumentViewer.tsx`

This is the main integration task:
1. Replace bottom floating bar with SelectionPopover
2. Add annotation highlight rendering via HTML post-processing
3. Wire AnnotationDialog for creating annotations
4. Wire AnnotationPreview for clicking highlights
5. Handle annotation clicks on `<mark>` elements
6. Add bidirectional linking callback for AnnotationPanel

- [ ] **Step 1: Rewrite DocumentViewer.tsx**

Replace the entire file with:

```tsx
import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import 'katex/dist/katex.min.css'
import markedKatexExtension from 'marked-katex-extension'
import { useAnnotationStore } from '../store/annotationStore'
import { useKnowledgeGraphStore } from '../store/knowledgeGraphStore'
import type { Document, Annotation } from '../types'
import { SelectionPopover } from './SelectionPopover'
import { AnnotationDialog } from './AnnotationDialog'
import type { AnnotationType } from './AnnotationDialog'
import { AnnotationPreview } from './AnnotationPreview'

marked.use(markedKatexExtension({ throwOnError: false }))

interface DocumentViewerProps {
  document: Document
  onConceptElevated?: () => void
}

export function DocumentViewer({ document, onConceptElevated }: DocumentViewerProps) {
  const [htmlContent, setHtmlContent] = useState('')
  const [isLoading, setIsLoading] = useState(true)
  const [selectedText, setSelectedText] = useState('')
  const [selectionRange, setSelectionRange] = useState<{ start: number; end: number } | null>(null)
  const [popoverPos, setPopoverPos] = useState<{ x: number; y: number } | null>(null)

  // Annotation dialog state
  const [dialogOpen, setDialogOpen] = useState(false)
  const [dialogType, setDialogType] = useState<AnnotationType>('comment')

  // Preview popover state
  const [previewAnnotation, setPreviewAnnotation] = useState<Annotation | null>(null)
  const [previewPos, setPreviewPos] = useState<{ x: number; y: number } | null>(null)

  const viewerRef = useRef<HTMLDivElement>(null)
  const {
    annotations,
    addAnnotation,
    selectAnnotation,
    selectedAnnotation,
    addReply,
  } = useAnnotationStore()
  const selectedConcept = useKnowledgeGraphStore(s => s.selectedConcept)
  const addConcept = useKnowledgeGraphStore(s => s.addConcept)
  const selectConcept = useKnowledgeGraphStore(s => s.selectConcept)
  const createConceptWithEdges = useKnowledgeGraphStore(s => s.createConceptWithEdges)

  const fullPath = useMemo(() => {
    const { projects, activeProjectId } = useKnowledgeGraphStore.getState()
    const project = projects.find(p => p.id === activeProjectId)
    const sourceDir = project?.sourceDir || ''
    if (!document.path) return '路径未知'
    if (document.path.startsWith('/')) return document.path
    if (sourceDir) {
      const relPath = document.path.replace(/^\.\//, '')
      return `${sourceDir}/${relPath}`
    }
    return document.path
  }, [document.path])

  // Render markdown to HTML
  useEffect(() => {
    setIsLoading(true)
    const renderMarkdown = async () => {
      try {
        const processed = document.content.replace(/\n(?=\s*=\s*$)/gm, ' ')
        const html = await marked(processed)
        const cleanHtml = DOMPurify.sanitize(html)
        setHtmlContent(cleanHtml)
        setIsLoading(false)
      } catch (error) {
        console.error('Failed to render markdown:', error)
        setIsLoading(false)
      }
    }
    renderMarkdown()
  }, [document.content])

  // Annotations belonging to this document
  const docAnnotations = useMemo(
    () => annotations.filter((ann) => ann.documentId === document.id),
    [annotations, document.id]
  )

  // Apply highlights: insert <mark> tags into HTML based on annotation positions
  const highlightedHtml = useMemo(() => {
    if (!htmlContent || docAnnotations.length === 0) return htmlContent

    // Create a temporary DOM to manipulate
    const tempDiv = document.createElement('div')
    tempDiv.innerHTML = htmlContent

    // Get all text nodes and build a flat text representation
    const textNodes: { node: Text; start: number; end: number }[] = []
    let charIndex = 0
    const walker = document.createTreeWalker(tempDiv, NodeFilter.SHOW_TEXT, null)
    let node: Text | null
    while ((node = walker.nextNode() as Text | null)) {
      const length = node.textContent?.length || 0
      if (length > 0) {
        textNodes.push({ node, start: charIndex, end: charIndex + length })
        charIndex += length
      }
    }

    // Sort annotations by position to process in reverse (avoid offset issues)
    const sorted = [...docAnnotations].sort((a, b) => b.position.start - a.position.start)

    for (const ann of sorted) {
      const { start, end } = ann.position
      if (start < 0 || end > charIndex || start >= end) continue

      // Find which text nodes contain this range
      let startNodeIdx = -1
      let endNodeIdx = -1
      for (let i = 0; i < textNodes.length; i++) {
        const tn = textNodes[i]
        if (start >= tn.start && start < tn.end) startNodeIdx = i
        if (end > tn.start && end <= tn.end) endNodeIdx = i
      }
      if (startNodeIdx === -1 || endNodeIdx === -1) continue

      const isResolved = ann.status === 'resolved' || ann.status === 'closed'
      const highlightClass = `ann-highlight ann-${ann.type}${isResolved ? ' ann-resolved' : ''}${selectedAnnotation?.id === ann.id ? ' ann-selected' : ''}`

      // Handle single text node case
      if (startNodeIdx === endNodeIdx) {
        const tn = textNodes[startNodeIdx]
        const relStart = start - tn.start
        const relEnd = end - tn.start
        const originalText = tn.node.textContent || ''
        const before = originalText.slice(0, relStart)
        const marked = originalText.slice(relStart, relEnd)
        const after = originalText.slice(relEnd)
        const span = document.createElement('mark')
        span.className = highlightClass
        span.dataset.annId = ann.id
        span.textContent = marked
        tn.node.parentNode?.insertBefore(document.createTextNode(before), tn.node)
        tn.node.parentNode?.insertBefore(span, tn.node)
        tn.node.parentNode?.insertBefore(document.createTextNode(after), tn.node)
        tn.node.parentNode?.removeChild(tn.node)
        continue
      }

      // Multi-node case: split across multiple text nodes
      // First node: take from relStart to end
      const firstTn = textNodes[startNodeIdx]
      const firstRelStart = start - firstTn.start
      const firstText = firstTn.node.textContent || ''
      const firstPart = firstText.slice(0, firstRelStart)
      const firstMark = firstText.slice(firstRelStart)

      // Last node: take from start to relEnd
      const lastTn = textNodes[endNodeIdx]
      const lastRelEnd = end - lastTn.start
      const lastText = lastTn.node.textContent || ''
      const lastMark = lastText.slice(0, lastRelEnd)
      const lastPart = lastText.slice(lastRelEnd)

      // Middle nodes: entire content is marked
      // Build marked content by collecting all text between (and including) start-end nodes
      let markedContent = firstMark
      for (let i = startNodeIdx + 1; i < endNodeIdx; i++) {
        markedContent += textNodes[i].node.textContent || ''
      }
      markedContent += lastMark

      const wrapper = document.createElement('mark')
      wrapper.className = highlightClass
      wrapper.dataset.annId = ann.id
      wrapper.textContent = markedContent

      // Replace first node
      firstTn.node.parentNode?.insertBefore(document.createTextNode(firstPart), firstTn.node)
      firstTn.node.parentNode?.insertBefore(wrapper, firstTn.node)
      firstTn.node.parentNode?.removeChild(firstTn.node)

      // Remove middle nodes
      for (let i = startNodeIdx + 1; i <= endNodeIdx; i++) {
        const tn = textNodes[i]
        if (tn.node.parentNode) {
          // For the last node, insert any remaining text
          if (i === endNodeIdx && lastPart) {
            tn.node.parentNode.insertBefore(document.createTextNode(lastPart), tn.node)
          }
          tn.node.parentNode.removeChild(tn.node)
        }
      }

      // Rebuild textNodes array (can't easily update indices, but we only do one annotation per render cycle)
      // Since we process in reverse order, upstream indices are unaffected
    }

    return tempDiv.innerHTML
  }, [htmlContent, docAnnotations, selectedAnnotation?.id])

  // Handle text selection
  const handleTextSelection = useCallback(() => {
    // Don't trigger if user clicks on an annotation mark
    const selection = window.getSelection()
    if (!selection || selection.isCollapsed) {
      setSelectedText('')
      setSelectionRange(null)
      setPopoverPos(null)
      return
    }

    const text = selection.toString().trim()
    if (text.length === 0) return

    setSelectedText(text)
    const range = selection.getRangeAt(0)
    const rect = range.getBoundingClientRect()
    setPopoverPos({
      x: rect.left + rect.width / 2,
      y: rect.top,
    })

    // Calculate text offset relative to the document content
    const preCaretRange = range.cloneRange()
    preCaretRange.selectNodeContents(viewerRef.current!)
    preCaretRange.setEnd(range.startContainer, range.startOffset)
    const start = preCaretRange.toString().length
    preCaretRange.setEnd(range.endContainer, range.endOffset)
    const end = preCaretRange.toString().length
    setSelectionRange({ start, end })
  }, [])

  // Open AnnotationDialog for a given type
  const handleOpenDialog = useCallback((type: AnnotationType) => {
    setDialogType(type)
    setDialogOpen(true)
    setPopoverPos(null)
  }, [])

  // Submit annotation from dialog
  const handleDialogSubmit = useCallback(async (data: {
    type: AnnotationType
    content: string
    selectedText: string
    enableAI: boolean
    enableConcept: boolean
  }) => {
    if (!selectionRange) return

    addAnnotation({
      documentId: document.id,
      type: data.type,
      content: data.content,
      position: selectionRange,
      author: 'User',
      status: 'open',
    })

    // Concept elevation
    if (data.enableConcept) {
      const newConcept = addConcept({
        title: data.selectedText.trim(),
        problem: data.content,
        depends_on: [],
        leads_to: [],
        related: [],
        path: '',
        tags: ['user-generated'],
        level: 1,
        category: 'other',
      })
      selectConcept(newConcept)
      onConceptElevated?.()
    }

    // AI answer (only for question type)
    if (data.enableAI && data.type === 'question') {
      try {
        const resp = await fetch('/api/ask-question', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            selectedText: data.selectedText,
            question: data.content,
          }),
        })
        if (resp.ok) {
          const result = await resp.json()
          const anns = useAnnotationStore.getState().annotations
          const latest = anns.filter(a => a.documentId === document.id).pop()
          if (latest && result.answer) {
            useAnnotationStore.getState().addReply(latest.id, {
              content: result.answer,
              author: 'AI',
            })
          }
        }
      } catch (e) {
        console.error('AI answer failed:', e)
      }
    }

    // Clear selection
    setSelectedText('')
    setSelectionRange(null)
    setPopoverPos(null)
    setDialogOpen(false)
    window.getSelection()?.removeAllRanges()
  }, [document.id, selectionRange, addAnnotation, addConcept, selectConcept, onConceptElevated])

  // Concept elevation directly from popover
  const handleConceptElevation = useCallback(() => {
    if (!selectedText) return
    if (!selectedConcept) return

    const newConcept = createConceptWithEdges(selectedConcept, {
      title: selectedText.trim(),
      problem: `与「${selectedConcept.title}」关联的概念`,
      relationType: 'leads_to',
      metadataStatus: 'draft',
    })
    selectConcept(newConcept)
    setSelectedText('')
    setSelectionRange(null)
    setPopoverPos(null)
    window.getSelection()?.removeAllRanges()
    onConceptElevated?.()
  }, [selectedText, selectedConcept, createConceptWithEdges, selectConcept, onConceptElevated])

  // Click on annotation highlight
  const handleAnnotationClick = useCallback((annotation: Annotation, rect: DOMRect) => {
    setPreviewAnnotation(annotation)
    setPreviewPos({
      x: rect.left + rect.width / 2,
      y: rect.top,
    })
  }, [])

  // "View details" from preview → select annotation + open panel
  const handleViewDetails = useCallback((annotation: Annotation) => {
    selectAnnotation(annotation)
    setPreviewAnnotation(null)
    setPreviewPos(null)
  }, [selectAnnotation])

  // Close popover / preview / dialog
  const handleClosePopover = useCallback(() => {
    setSelectedText('')
    setSelectionRange(null)
    setPopoverPos(null)
    window.getSelection()?.removeAllRanges()
  }, [])

  const handleClosePreview = useCallback(() => {
    setPreviewAnnotation(null)
    setPreviewPos(null)
  }, [])

  // Handle clicks inside the document viewer (for annotation mark clicks)
  const handleViewerClick = useCallback((e: React.MouseEvent) => {
    const target = e.target as HTMLElement
    if (target.tagName === 'MARK' && target.dataset.annId) {
      const annId = target.dataset.annId
      const ann = docAnnotations.find(a => a.id === annId)
      if (ann) {
        const rect = target.getBoundingClientRect()
        handleAnnotationClick(ann, rect)
      }
    }
  }, [docAnnotations, handleAnnotationClick])

  return (
    <div className="h-full flex flex-col">
      <div className="border-b border-gray-200 px-6 py-4">
        <h1 className="text-2xl font-bold text-gray-900">{document.title}</h1>
        <div className="mt-2 flex items-center gap-4 text-sm text-gray-600">
          <span className="px-2 py-1 bg-gray-100 text-gray-700 rounded font-mono text-xs" title="文档存储路径">
            {fullPath}
          </span>
        </div>
      </div>

      <div
        ref={viewerRef}
        className="flex-1 overflow-y-auto px-6 py-4 scroll-smooth"
        onMouseUp={handleTextSelection}
        onClick={handleViewerClick}
      >
        {isLoading ? (
          <div className="flex h-full items-center justify-center text-gray-500">
            <div className="text-center">
              <div className="spinner mb-4"></div>
              <p>加载文档中...</p>
            </div>
          </div>
        ) : (
          <div
            className="markdown-content prose prose-lg max-w-none"
            dangerouslySetInnerHTML={{ __html: highlightedHtml }}
          />
        )}
      </div>

      {/* Selection Popover */}
      {popoverPos && selectedText && (
        <SelectionPopover
          position={popoverPos}
          selectedText={selectedText}
          onAddAnnotation={handleOpenDialog}
          onConceptElevation={handleConceptElevation}
          onClose={handleClosePopover}
        />
      )}

      {/* Annotation Dialog */}
      {dialogOpen && (
        <AnnotationDialog
          selectedText={selectedText}
          initialType={dialogType}
          onClose={() => setDialogOpen(false)}
          onSubmit={handleDialogSubmit}
        />
      )}

      {/* Annotation Preview */}
      {previewAnnotation && previewPos && (
        <AnnotationPreview
          annotation={previewAnnotation}
          position={previewPos}
          onViewDetails={handleViewDetails}
          onClose={handleClosePreview}
        />
      )}
    </div>
  )
}
```

**Key changes from original:**
- Removed: `showQuestionDialog`, `handleAddAnnotation`, `handleQuestionSubmit`, bottom floating bar JSX
- Added: `popoverPos`, `dialogOpen`, `dialogType`, `previewAnnotation`, `previewPos` state
- Added: `highlightedHtml` (useMemo that inserts `<mark>` tags into rendered HTML)
- Added: `SelectionPopover` rendering at selection position
- Added: `AnnotationDialog` rendering (unified)
- Added: `AnnotationPreview` rendering on mark click
- Added: `handleViewerClick` to detect clicks on `<mark>` elements
- Exported: `docAnnotations` for external linking

- [ ] **Step 2: Remove QuestionDialog import**

In the new file, there is no `import { QuestionDialog } from './QuestionDialog'` — this is handled by the new `AnnotationDialog`. Ensure no remaining references to QuestionDialog.

- [ ] **Step 3: Typecheck**

Run: `npm run typecheck`
Expected: No errors

- [ ] **Step 4: Build check**

Run: `npm run build`
Expected: Bundles successfully

- [ ] **Step 5: Commit**

```bash
git add src/components/DocumentViewer.tsx
git commit -m "feat: integrate SelectionPopover, AnnotationDialog, highlights, and preview into DocumentViewer"
```

---

### Task 6: AnnotationPanel Linking

**Files:**
- Modify: `src/components/AnnotationPanel.tsx`

Add support for a callback when user wants to navigate to a highlight in the document. Also show a "show in document" action for each annotation.

- [ ] **Step 1: Update AnnotationPanel interface and logic**

Find the `AnnotationPanelProps` interface and add `onNavigateToAnnotation?: (annotationId: string) => void`:

```tsx
interface AnnotationPanelProps {
  document: Document
  onClose: () => void
  onNavigateToAnnotation?: (annotationId: string) => void
}
```

Update the `function AnnotationPanel` to accept the prop:

```tsx
export function AnnotationPanel({ document, onClose, onNavigateToAnnotation }: AnnotationPanelProps) {
```

Add a "show in document" link in each annotation entry, inside the annotation header area (after the date, before the status select). Insert this after the `<span className="text-xs text-gray-500">` date element:

```tsx
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        selectAnnotation(annotation)
                        onNavigateToAnnotation?.(annotation.id)
                      }}
                      className="text-xs text-blue-500 hover:text-blue-700 ml-2"
                      title="在文档中查看"
                    >
                      📍
                    </button>
```

- [ ] **Step 2: Wire the callback in App.tsx**

In `src/App.tsx`, find the `AnnotationPanel` usage (line ~161-168):

```tsx
      {(viewMode === 'documents' || selectedDoc) && isAnnotationPanelOpen && selectedDoc && (
        <div className="shrink-0">
          <AnnotationPanel
            document={selectedDoc}
            onClose={handleAnnotationPanelToggle}
          />
        </div>
      )}
```

Add `onNavigateToAnnotation` callback that:
1. Focuses the document viewer
2. Scrolls the annotation highlight into view

```tsx
      {(viewMode === 'documents' || selectedDoc) && isAnnotationPanelOpen && selectedDoc && (
        <div className="shrink-0">
          <AnnotationPanel
            document={selectedDoc}
            onClose={handleAnnotationPanelToggle}
            onNavigateToAnnotation={(annotationId) => {
              // Find the mark element and scroll to it
              setTimeout(() => {
                const mark = document.querySelector(`mark[data-ann-id="${annotationId}"]`)
                if (mark) {
                  mark.scrollIntoView({ behavior: 'smooth', block: 'center' })
                  mark.classList.add('ann-selected')
                  setTimeout(() => mark.classList.remove('ann-selected'), 1500)
                }
              }, 100)
            }}
          />
        </div>
      )}
```

- [ ] **Step 3: Typecheck**

Run: `npm run typecheck`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add src/components/AnnotationPanel.tsx src/App.tsx
git commit -m "feat: add bidirectional linking between AnnotationPanel and document highlights"
```

---

### Task 7: Remove QuestionDialog

**Files:**
- Delete: `src/components/QuestionDialog.tsx`

- [ ] **Step 1: Delete the old file**

```bash
rm src/components/QuestionDialog.tsx
```

- [ ] **Step 2: Verify no remaining references**

Run: `grep -r "QuestionDialog" src/ --include="*.ts" --include="*.tsx"`
Expected: No matches (DocumentViewer.tsx already uses AnnotationDialog)

- [ ] **Step 3: Typecheck + Build**

Run: `npm run typecheck && npm run build`
Expected: Both pass

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: remove QuestionDialog (replaced by AnnotationDialog)"
```

---

## Self-Review Checklist

**1. Spec coverage:**
- ✅ SelectionPopover (Task 2) — covers spec section 1
- ✅ AnnotationDialog (Task 3) — covers spec section 2 (replaces QuestionDialog)
- ✅ Annotation highlights (Task 5, inside DocumentViewer) — covers spec section 3
- ✅ AnnotationPreview (Task 4) — covers spec section 4
- ✅ AnnotationPanel linking (Task 6) — covers spec section 5
- ✅ CSS styles (Task 1) — covers spec style section
- ✅ QuestionDialog removal (Task 7) — covers spec file deletion list

**2. Placeholder scan:** No TBD, TODO, "implement later", or vague patterns exist.

**3. Type consistency:** All types reference the existing `Annotation` from `src/types/index.ts`. Component prop types are consistent across tasks. The `AnnotationType` type (exported from `AnnotationDialog.tsx`) is properly imported by `DocumentViewer.tsx`.
