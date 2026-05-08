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

const domDoc = window.document

export function DocumentViewer({ document, onConceptElevated }: DocumentViewerProps) {
  const [htmlContent, setHtmlContent] = useState('')
  const [isLoading, setIsLoading] = useState(true)
  const [selectedText, setSelectedText] = useState('')
  const [selectionRange, setSelectionRange] = useState<{ start: number; end: number } | null>(null)
  const [popoverPos, setPopoverPos] = useState<{ x: number; y: number } | null>(null)

  const [dialogOpen, setDialogOpen] = useState(false)
  const [dialogType, setDialogType] = useState<AnnotationType>('comment')

  const [previewAnnotation, setPreviewAnnotation] = useState<Annotation | null>(null)
  const [previewPos, setPreviewPos] = useState<{ x: number; y: number } | null>(null)

  const viewerRef = useRef<HTMLDivElement>(null)
  const {
    annotations,
    addAnnotation,
    selectAnnotation,
    selectedAnnotation,
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

  const docAnnotations = useMemo(
    () => annotations.filter((ann) => ann.documentId === document.id),
    [annotations, document.id]
  )

  const highlightedHtml = useMemo(() => {
    if (!htmlContent || docAnnotations.length === 0) return htmlContent

    const tempDiv = domDoc.createElement('div')
    tempDiv.innerHTML = htmlContent

    const textNodes: { node: Text; start: number; end: number }[] = []
    let charIndex = 0
    const walker = domDoc.createTreeWalker(tempDiv, NodeFilter.SHOW_TEXT, null)
    let node: Text | null
    while ((node = walker.nextNode() as Text | null)) {
      const length = node.textContent?.length || 0
      if (length > 0) {
        textNodes.push({ node, start: charIndex, end: charIndex + length })
        charIndex += length
      }
    }

    const totalDocLength = charIndex

    const sorted = [...docAnnotations].sort((a, b) => b.position.start - a.position.start)

    for (const ann of sorted) {
      const { start, end } = ann.position
      if (start < 0 || end > totalDocLength || start >= end) continue

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

      if (startNodeIdx === endNodeIdx) {
        const tn = textNodes[startNodeIdx]
        const relStart = start - tn.start
        const relEnd = end - tn.start
        const originalText = tn.node.textContent || ''
        const before = originalText.slice(0, relStart)
        const markedText = originalText.slice(relStart, relEnd)
        const after = originalText.slice(relEnd)
        const span = domDoc.createElement('mark')
        span.className = highlightClass
        span.dataset.annId = ann.id
        span.textContent = markedText
        tn.node.parentNode?.insertBefore(domDoc.createTextNode(before), tn.node)
        tn.node.parentNode?.insertBefore(span, tn.node)
        tn.node.parentNode?.insertBefore(domDoc.createTextNode(after), tn.node)
        tn.node.parentNode?.removeChild(tn.node)
        continue
      }

      const firstTn = textNodes[startNodeIdx]
      const firstRelStart = start - firstTn.start
      const firstText = firstTn.node.textContent || ''
      const firstPart = firstText.slice(0, firstRelStart)
      const firstMark = firstText.slice(firstRelStart)

      const lastTn = textNodes[endNodeIdx]
      const lastRelEnd = end - lastTn.start
      const lastText = lastTn.node.textContent || ''
      const lastMark = lastText.slice(0, lastRelEnd)
      const lastPart = lastText.slice(lastRelEnd)

      let markedContent = firstMark
      for (let i = startNodeIdx + 1; i < endNodeIdx; i++) {
        markedContent += textNodes[i].node.textContent || ''
      }
      markedContent += lastMark

      const wrapper = domDoc.createElement('mark')
      wrapper.className = highlightClass
      wrapper.dataset.annId = ann.id
      wrapper.textContent = markedContent

      firstTn.node.parentNode?.insertBefore(domDoc.createTextNode(firstPart), firstTn.node)
      firstTn.node.parentNode?.insertBefore(wrapper, firstTn.node)
      firstTn.node.parentNode?.removeChild(firstTn.node)

      for (let i = startNodeIdx + 1; i <= endNodeIdx; i++) {
        const tn = textNodes[i]
        if (tn.node.parentNode) {
          if (i === endNodeIdx && lastPart) {
            tn.node.parentNode.insertBefore(domDoc.createTextNode(lastPart), tn.node)
          }
          tn.node.parentNode.removeChild(tn.node)
        }
      }
    }

    return tempDiv.innerHTML
  }, [htmlContent, docAnnotations, selectedAnnotation?.id])

  const handleTextSelection = useCallback(() => {
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

    const preCaretRange = range.cloneRange()
    preCaretRange.selectNodeContents(viewerRef.current!)
    preCaretRange.setEnd(range.startContainer, range.startOffset)
    const start = preCaretRange.toString().length
    preCaretRange.setEnd(range.endContainer, range.endOffset)
    const end = preCaretRange.toString().length
    setSelectionRange({ start, end })
  }, [])

  const handleOpenDialog = useCallback((type: AnnotationType) => {
    setDialogType(type)
    setDialogOpen(true)
    setPopoverPos(null)
    setPreviewAnnotation(null)
    setPreviewPos(null)
  }, [])

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

    setSelectedText('')
    setSelectionRange(null)
    setPopoverPos(null)
    setDialogOpen(false)
    window.getSelection()?.removeAllRanges()
  }, [document.id, selectionRange, addAnnotation, addConcept, selectConcept, onConceptElevated])

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

  const handleAnnotationClick = useCallback((annotation: Annotation, rect: DOMRect) => {
    setPreviewAnnotation(annotation)
    setPreviewPos({
      x: rect.left + rect.width / 2,
      y: rect.top,
    })
  }, [])

  const handleViewDetails = useCallback((annotation: Annotation) => {
    selectAnnotation(annotation)
    setPreviewAnnotation(null)
    setPreviewPos(null)
  }, [selectAnnotation])

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

      {popoverPos && selectedText && (
        <SelectionPopover
          position={popoverPos}
          selectedText={selectedText}
          onAddAnnotation={handleOpenDialog}
          onConceptElevation={handleConceptElevation}
          onClose={handleClosePopover}
        />
      )}

      {dialogOpen && (
        <AnnotationDialog
          selectedText={selectedText}
          initialType={dialogType}
          onClose={() => setDialogOpen(false)}
          onSubmit={handleDialogSubmit}
        />
      )}

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
