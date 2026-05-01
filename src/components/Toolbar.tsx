import React from 'react'
import { useKnowledgeGraphStore } from '../store/knowledgeGraphStore'

interface ToolbarProps {
  onSidebarToggle: () => void
  onAnnotationPanelToggle: () => void
  onExport: () => void
  isSidebarOpen: boolean
  isAnnotationPanelOpen: boolean
  hasDocument: boolean
viewMode?: 'documents' | 'knowledge-graph' | 'cluster'
  onViewModeChange?: (mode: 'documents' | 'knowledge-graph' | 'cluster') => void
}

export function Toolbar({
  onSidebarToggle,
  onAnnotationPanelToggle,
  onExport,
  isSidebarOpen,
  isAnnotationPanelOpen,
  hasDocument,
  viewMode = 'documents',
  onViewModeChange = () => {},
}: ToolbarProps) {
  const linkMode = useKnowledgeGraphStore(s => s.linkMode)
  const setLinkMode = useKnowledgeGraphStore(s => s.setLinkMode)
  const setLinkSource = useKnowledgeGraphStore(s => s.setLinkSource)

  const handleViewModeChange = (mode: 'documents' | 'knowledge-graph' | 'cluster') => {
    console.log('Toolbar: switching to', mode)
    onViewModeChange?.(mode)
  }

  const handleLinkModeToggle = () => {
    console.log('[Toolbar] handleLinkModeToggle called, current linkMode:', linkMode)
    if (linkMode) {
      window.dispatchEvent(new CustomEvent('exit-link-mode'))
    } else {
      window.dispatchEvent(new CustomEvent('toggle-link-mode'))
    }
  }

  return (
    <div className="h-14 border-b border-gray-200 bg-white flex items-center justify-between px-4">
      <div className="flex items-center gap-2">
        <button
          onClick={onSidebarToggle}
          className={`p-2 rounded hover:bg-gray-100 transition-colors ${
            isSidebarOpen ? 'bg-blue-50 text-blue-600' : 'text-gray-600'
          }`}
          aria-label="切换侧边栏"
        >
          <svg width={20} height={20} className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>

        <div className="h-6 w-px bg-gray-300 mx-2" />

        <h1 className="text-lg font-semibold text-gray-900">Magic Memory</h1>

        <div className="flex items-center gap-1 ml-4">
          <button
            onClick={() => handleViewModeChange('documents')}
            className={`px-3 py-1 rounded text-sm ${
              viewMode === 'documents' 
                ? 'bg-blue-500 text-white' 
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            文档
          </button>
          <button
            onClick={() => handleViewModeChange('knowledge-graph')}
            className={`px-3 py-1 rounded text-sm ${
              viewMode === 'knowledge-graph' 
                ? 'bg-blue-500 text-white' 
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            知识图
          </button>
          <button
            onClick={() => handleViewModeChange('cluster')}
            className={`px-3 py-1 rounded text-sm ${
              viewMode === 'cluster' 
                ? 'bg-blue-500 text-white' 
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            聚类
          </button>
        </div>
      </div>

      <div className="flex items-center gap-2">
        {viewMode === 'knowledge-graph' && (
          <button
            onClick={handleLinkModeToggle}
            style={{
              backgroundColor: linkMode ? '#2563eb' : '#f3f4f6',
              color: linkMode ? '#ffffff' : '#4b5563'
            }}
            className="p-1.5 rounded transition-colors hover:opacity-80"
            title={linkMode ? '退出连线' : '连线模式'}
          >
            <svg width={16} height={16} className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.979-1.101l1.101-1.102a4 4 0 005.657-5.656l-4-4z" />
            </svg>
          </button>
        )}
        
        <button
          onClick={onAnnotationPanelToggle}
          className={`p-2 rounded hover:bg-gray-100 transition-colors ${
            isAnnotationPanelOpen ? 'bg-blue-50 text-blue-600' : 'text-gray-600'
          }`}
          aria-label="切换注释面板"
        >
          <svg width={20} height={20} className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
          </svg>
        </button>

        <button
          onClick={onExport}
          disabled={!hasDocument}
          className={`p-2 rounded hover:bg-gray-100 transition-colors ${
            hasDocument ? 'text-gray-600' : 'text-gray-300 cursor-not-allowed'
          }`}
          aria-label="导出文档"
          title="导出文档"
        >
          <svg width={20} height={20} className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
        </button>

        <div className="h-6 w-px bg-gray-300 mx-2" />

        <button className="p-2 rounded hover:bg-gray-100 text-gray-600" aria-label="设置">
          <svg width={20} height={20} className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        </button>
      </div>
    </div>
  )
}
