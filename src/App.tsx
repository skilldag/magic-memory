import React, { useState, useEffect } from 'react'
import { DocumentViewer } from './components/DocumentViewer'
import { Sidebar } from './components/Sidebar'
import { AnnotationPanel } from './components/AnnotationPanel'
import { Toolbar } from './components/Toolbar'
import { ExportModal } from './components/ExportModal'
import { ImportModal } from './components/ImportModal'
import { KnowledgeGraphView } from './components/KnowledgeGraphView'
import { ClusterView } from './components/ClusterView'
import { useDocumentStore } from './store/documentStore'
import { useAnnotationStore } from './store/annotationStore'
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
                <p className="text-sm">从左侧选择一个文档开始阅读和标注</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 侧边栏在右侧 */}
      {isSidebarOpen && (
        <div className="shrink-0" style={{ width: 288, minWidth: 288 }}>
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
    </div>
  )
}

export default App
