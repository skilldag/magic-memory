import React, { useState, useEffect, useRef } from 'react'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import 'katex/dist/katex.min.css'
import markedKatexExtension from 'marked-katex-extension'
import { useAnnotationStore } from '../store/annotationStore'
import { useKnowledgeGraphStore } from '../store/knowledgeGraphStore'
import type { Document } from '../types'

// 配置 marked 支持 LaTeX 公式渲染
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
  const viewerRef = useRef<HTMLDivElement>(null)
  const { annotations, addAnnotation, selectAnnotation, selectedAnnotation } = useAnnotationStore()
  const selectedConcept = useKnowledgeGraphStore(s => s.selectedConcept)
  const createConceptWithEdges = useKnowledgeGraphStore(s => s.createConceptWithEdges)
  const selectConcept = useKnowledgeGraphStore(s => s.selectConcept)

  useEffect(() => {
    console.log('DocumentViewer received:', document.id, document.title, 'content length:', document.content?.length)
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

  const handleTextSelection = () => {
    const selection = window.getSelection()
    if (!selection || selection.isCollapsed) {
      setSelectedText('')
      setSelectionRange(null)
      return
    }

    const text = selection.toString().trim()
    if (text.length > 0) {
      setSelectedText(text)
      const range = selection.getRangeAt(0)
      const preCaretRange = range.cloneRange()
      preCaretRange.selectNodeContents(viewerRef.current!)
      preCaretRange.setEnd(range.startContainer, range.startOffset)
      const start = preCaretRange.toString().length

      preCaretRange.setEnd(range.endContainer, range.endOffset)
      const end = preCaretRange.toString().length

      setSelectionRange({ start, end })
    }
  }

  const handleAddAnnotation = (type: 'comment' | 'question' | 'suggestion' | 'correction') => {
    if (!selectionRange || !selectedText) return

    addAnnotation({
      documentId: document.id,
      type,
      content: '',
      position: selectionRange,
      author: 'User',
      status: 'open',
    })

    setSelectedText('')
    setSelectionRange(null)
    window.getSelection()?.removeAllRanges()
  }

  const handleConceptElevation = () => {
    if (!selectedText) {
      alert('请先选中要提升为概念的文字')
      return
    }
    if (!selectedConcept) {
      return
    }

    const newConcept = createConceptWithEdges(selectedConcept, {
      title: selectedText.trim(),
      problem: `与「${selectedConcept.title}」关联的概念`,
      relationType: 'leads_to',
      metadataStatus: 'draft',
    })

    selectConcept(newConcept)
    setSelectedText('')
    setSelectionRange(null)
    window.getSelection()?.removeAllRanges()
    console.log('[概念提升] 已创建概念:', newConcept.title, 'ID:', newConcept.id)
    onConceptElevated?.()
  }

  const handleAnnotationClick = (annotationId: string) => {
    const annotation = annotations.find((ann) => ann.id === annotationId)
    if (annotation) {
      selectAnnotation(annotation)
    }
  }

  return (
    <div className="h-full flex flex-col">
      <div className="border-b border-gray-200 px-6 py-4">
        <h1 className="text-2xl font-bold text-gray-900">{document.title}</h1>
        <div className="mt-2 flex items-center gap-4 text-sm text-gray-600">
          <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded">
            Level {document.level}
          </span>
          <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded">
            {document.category}
          </span>
          {document.tags.map((tag) => (
            <span key={tag} className="px-2 py-1 bg-gray-100 text-gray-700 rounded">
              {tag}
            </span>
          ))}
        </div>
      </div>

      <div
        ref={viewerRef}
        className="flex-1 overflow-y-auto px-6 py-4 scroll-smooth"
        onMouseUp={handleTextSelection}
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
            dangerouslySetInnerHTML={{ __html: htmlContent }}
          />
        )}
      </div>

      {selectedText && selectionRange && (
        <div className="fixed bottom-4 left-1/2 transform -translate-x-1/2 bg-white shadow-lg rounded-lg p-4 border border-gray-200">
          <div className="mb-3">
            <p className="text-sm text-gray-600 mb-2">选中的文本:</p>
            <p className="text-sm font-medium bg-gray-50 p-2 rounded">
              {selectedText}
            </p>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => handleAddAnnotation('comment')}
              className="px-3 py-1.5 bg-blue-500 text-white rounded hover:bg-blue-600 text-sm"
            >
              添加评论
            </button>
            <button
              onClick={() => handleAddAnnotation('question')}
              className="px-3 py-1.5 bg-purple-500 text-white rounded hover:bg-purple-600 text-sm"
            >
              提出问题
            </button>
<button
          onClick={handleConceptElevation}
          className="px-3 py-1.5 bg-green-500 text-white rounded hover:bg-green-600 text-sm"
        >
          概念提升
        </button>
            <button
              onClick={() => handleAddAnnotation('correction')}
              className="px-3 py-1.5 bg-red-500 text-white rounded hover:bg-red-600 text-sm"
            >
              纠正错误
            </button>
            <button
              onClick={() => {
                setSelectedText('')
                setSelectionRange(null)
                window.getSelection()?.removeAllRanges()
              }}
              className="px-3 py-1.5 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 text-sm"
            >
              取消
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
