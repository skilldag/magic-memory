import { useState } from 'react'
import { ProjectList } from './ProjectList'
import { useProjectStore } from '../store/projectStore'

interface SidebarProps {
  onClose: () => void
  onProjectSwitch?: () => void
}

export function Sidebar({ onClose, onProjectSwitch }: SidebarProps) {

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
    <div className="w-72 lg:w-80 border-l border-gray-200 flex flex-col bg-white shrink-0 h-full">
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200">
        <h2 className="text-sm font-semibold text-gray-900">项目</h2>
        <button
          onClick={onClose}
          className="p-1 hover:bg-gray-100 rounded"
          aria-label="关闭侧边栏"
        >
          <svg width={16} height={16} className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
      <div className="flex-1 overflow-y-auto">
        <ProjectList onSwitch={onProjectSwitch} />
      </div>
      <div className="p-2 border-t border-gray-200">
        <button
          onClick={handleAddProject}
          className="w-full flex items-center justify-center gap-1 px-2 py-1.5 text-xs text-blue-600 hover:bg-blue-50 rounded transition-colors"
        >
          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          添加项目
        </button>
      </div>
    </div>
  )
}
