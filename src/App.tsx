import React, { useState, useEffect } from 'react'
import { DocumentViewer } from './components/DocumentViewer'
import { ProjectList } from './components/ProjectList'
import { Sidebar } from './components/Sidebar'
import { AnnotationPanel } from './components/AnnotationPanel'
import { Toolbar } from './components/Toolbar'
import { ExportModal } from './components/ExportModal'
import { ImportModal } from './components/ImportModal'
import { Toast } from './components/Toast'
import { KnowledgeGraphView } from './components/KnowledgeGraphView'
import { ClusterView } from './components/ClusterView'
import { useDocumentStore } from './store/documentStore'
import { useAnnotationStore } from './store/annotationStore'
import { useProjectStore } from './store/projectStore'
import type { Document } from './types'

function App() {
  const [selectedDoc, setSelectedDoc] = useState<Document | null>(null)
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const [isAnnotationPanelOpen, setIsAnnotationPanelOpen] = useState(false)
  const [showExportModal, setShowExportModal] = useState(false)
  const [showImportModal, setShowImportModal] = useState(false)
  const [viewMode, setViewMode] = useState<'documents' | 'knowledge-graph' | 'cluster'>('knowledge-graph')

  const handleViewModeChange = (mode: 'documents' | 'knowledge-graph' | 'cluster') => {
    setViewMode(mode)
  }

  const documents = useDocumentStore(state => state.documents)
  const loadDocuments = useDocumentStore(state => state.loadDocuments)
  const { addAnnotation } = useAnnotationStore()

  useEffect(() => {
    loadDocuments()
  }, [loadDocuments])

  const handleDocumentSelect = (doc: Document) => {
    setSelectedDoc(doc)
  }

  const handleSidebarToggle = () => {
    setIsSidebarOpen(!isSidebarOpen)
  }

  const handleAnnotationPanelToggle = () => {
    setIsAnnotationPanelOpen(!isAnnotationPanelOpen)
  }

  const handleExport = () => {
    setShowExportModal(true)
  }

  const handleImport = () => {
    setShowImportModal(true)
  }

  const handleImportData = (data: { document: Document; annotations: any[] }) => {
    setSelectedDoc(data.document)
    data.annotations.forEach(ann => {
      addAnnotation({
        ...ann,
        documentId: data.document.id,
      })
    })
  }

  const handleConceptElevated = () => {
    setViewMode('knowledge-graph')
  }

  const handleAddProject = async () => {
    try {
      const handle = await (window as any).showDirectoryPicker()
      const name = handle.name
      const project = await useProjectStore.getState().createProject(name, handle)
      if (!project) {
        alert('创建项目失败，请重试')
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        console.error('Failed to create project:', err)
        alert('创建项目失败: ' + (err.message || err))
      }
    }
  }

  const handleProjectSwitch = () => {
    if (viewMode !== 'knowledge-graph') {
      setViewMode('knowledge-graph')
    }
  }

  const isGraphMode = viewMode === 'knowledge-graph'
  const isClusterMode = viewMode === 'cluster'

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-white">
      <div className="flex flex-1 flex-col overflow-hidden min-w-0">
        <Toolbar
          onSidebarToggle={handleSidebarToggle}
          onAnnotationPanelToggle={handleAnnotationPanelToggle}
          onExport={handleExport}
          isSidebarOpen={isSidebarOpen}
          isAnnotationPanelOpen={isAnnotationPanelOpen}
          hasDocument={!!selectedDoc}
          onViewModeChange={handleViewModeChange}
          viewMode={viewMode}
        />

        <div className="flex-1 overflow-hidden min-w-0 flex">
          <div className="flex-1 overflow-hidden min-w-0">
            {isClusterMode ? (
              <ClusterView />
            ) : isGraphMode ? (
              <KnowledgeGraphView />
            ) : selectedDoc ? (
              <DocumentViewer document={selectedDoc} onConceptElevated={handleConceptElevated} />
            ) : (
              <div className="flex h-full items-center justify-center text-gray-500">
                <div className="text-center">
                  <div className="mb-4 text-6xl">📚</div>
                  <h2 className="text-xl font-semibold mb-2">欢迎使用 Magic Memory</h2>
                  <p className="text-sm">从侧边栏选择一个文档开始阅读和标注</p>
                </div>
              </div>
            )}
          </div>
          {viewMode === 'documents' && (
            <div className="w-[560px] border-l border-gray-200 flex flex-col shrink-0 bg-white">
              <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200">
                <h2 className="text-sm font-semibold text-gray-900">项目</h2>
              </div>
              <div className="flex-1 overflow-y-auto">
                <ProjectList onSwitch={handleProjectSwitch} />
              </div>
              <div className="border-t border-gray-200">
                <button
                  onClick={handleAddProject}
                  className="w-full flex items-center justify-center gap-1.5 px-4 py-3 text-sm text-blue-600 hover:bg-blue-50 transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                  </svg>
                  添加项目
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {isSidebarOpen && viewMode !== 'documents' && (
        <div className="shrink-0">
          <Sidebar
            onClose={handleSidebarToggle}
            onProjectSwitch={() => setViewMode('knowledge-graph')}
          />
        </div>
      )}

      {(viewMode === 'documents' || selectedDoc) && isAnnotationPanelOpen && selectedDoc && (
        <div className="shrink-0">
          <AnnotationPanel
            document={selectedDoc}
            onClose={handleAnnotationPanelToggle}
            onNavigateToAnnotation={(annotationId) => {
              setTimeout(() => {
                const mark = window.document.querySelector(`mark[data-ann-id="${annotationId}"]`)
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

      {showExportModal && selectedDoc && (
        <ExportModal
          document={selectedDoc}
          isOpen={showExportModal}
          onClose={() => setShowExportModal(false)}
        />
      )}

      {showImportModal && (
        <ImportModal
          isOpen={showImportModal}
          onClose={() => setShowImportModal(false)}
          onImport={handleImportData}
        />
      )}

      <Toast />
    </div>
  )
}

export default App
