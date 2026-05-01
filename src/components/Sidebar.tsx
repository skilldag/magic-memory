import { useState } from 'react'
import { ProjectList } from './ProjectList'
import { useProjectStore } from '../store/projectStore'

interface SidebarProps {
  onClose: () => void
}

export function Sidebar({ onClose }: SidebarProps) {

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

  return (
    <div className="w-72 lg:w-80 border-r border-gray-200 flex flex-col bg-gray-50 shrink-0">
      <div className="border-b border-gray-200">
        <ProjectList onAddProject={handleAddProject} />
      </div>
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900">项目</h2>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-200 rounded"
            aria-label="关闭侧边栏"
          >
            <svg width={20} height={20} className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  )
}
