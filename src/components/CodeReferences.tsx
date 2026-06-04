import { useState } from 'react'
import type { CodeReference } from '../types'

interface CodeReferencesProps {
  codeRefs: CodeReference[]
  sourceDir: string
}

export function CodeReferences({ codeRefs, sourceDir }: CodeReferencesProps) {
  const [selectedRef, setSelectedRef] = useState<CodeReference | null>(null)
  const [codeContent, setCodeContent] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const loadCode = async (ref: CodeReference) => {
    setSelectedRef(ref)
    setLoading(true)
    try {
      const resp = await fetch(`/api/read-doc?path=${encodeURIComponent(ref.file)}&baseDir=${encodeURIComponent(sourceDir)}`)
      if (resp.ok) {
        const data = await resp.json()
        setCodeContent(data.content || null)
      } else {
        setCodeContent(null)
      }
    } catch {
      setCodeContent(null)
    }
    setLoading(false)
  }

  const openInIDE = (ref: CodeReference) => {
    const fullPath = `${sourceDir}/${ref.file}`
    const lineArg = ref.lineStart ? `:${ref.lineStart}` : ''
    window.open(`vscode://file/${fullPath}${lineArg}`, '_blank')
  }

  if (!codeRefs || codeRefs.length === 0) return null

  return (
    <div className="px-5 py-4 space-y-3">
      {codeRefs.map((ref, i) => (
        <div key={i} className="border border-gray-200 rounded-lg overflow-hidden">
          <div
            className="flex items-center justify-between px-3 py-2 bg-gray-50 cursor-pointer hover:bg-gray-100 transition-colors"
            onClick={() => loadCode(ref)}
          >
            <div className="flex items-center gap-2 min-w-0">
              <svg width={14} height={14} fill="none" stroke="currentColor" viewBox="0 0 24 24" className="shrink-0 text-gray-400">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
              </svg>
              <span className="text-xs font-mono text-gray-700 truncate">{ref.file}</span>
              {ref.lineStart && (
                <span className="text-xs text-gray-400 shrink-0">L{ref.lineStart}{ref.lineEnd ? `-L${ref.lineEnd}` : ''}</span>
              )}
            </div>
            <button
              onClick={(e) => { e.stopPropagation(); openInIDE(ref); }}
              className="shrink-0 px-2 py-0.5 text-xs text-blue-600 hover:text-blue-800 hover:bg-blue-50 rounded transition-colors"
              title="在 IDE 中打开"
            >
              IDE
            </button>
          </div>
          {ref.description && (
            <div className="px-3 py-1.5 text-xs text-gray-500 border-t border-gray-100">
              {ref.description}
            </div>
          )}
          {selectedRef === ref && (
            <div className="border-t border-gray-200">
              {loading ? (
                <div className="px-3 py-4 text-xs text-gray-400 text-center">加载中...</div>
              ) : codeContent ? (
                <pre className="px-3 py-2 text-xs font-mono text-gray-800 bg-gray-50 overflow-x-auto max-h-60 overflow-y-auto">
                  {(ref.lineStart
                    ? codeContent.split('\n').slice(Math.max(0, (ref.lineStart || 1) - 1), ref.lineEnd || undefined)
                    : codeContent.split('\n').slice(0, 20)
                  ).map((line, idx) => (
                    <div key={idx} className="flex">
                      <span className="text-gray-400 w-8 text-right mr-2 shrink-0 select-none">
                        {(ref.lineStart || 1) + idx}
                      </span>
                      <span>{line || ' '}</span>
                    </div>
                  ))}
                </pre>
              ) : (
                <div className="px-3 py-4 text-xs text-gray-400 text-center">无法加载代码文件</div>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
