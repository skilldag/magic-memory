import { useEffect, useState } from 'react';
import { useProjectStore } from '../store/projectStore';

interface ProjectListProps {
  onSwitch?: () => void;
}

export function ProjectList({ onSwitch }: ProjectListProps) {
  const { projects, currentProjectId, switchProject, deleteProject, isLoading, error, loadProjects } = useProjectStore();
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);

  useEffect(() => {
    loadProjects();
  }, [loadProjects]);

  const handleSwitch = async (projectId: string) => {
    await switchProject(projectId);
    onSwitch?.();
  };

  const handleDelete = async (projectId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirmDelete === projectId) {
      await deleteProject(projectId);
      setConfirmDelete(null);
    } else {
      setConfirmDelete(projectId);
      setTimeout(() => setConfirmDelete(null), 3000);
    }
  };

  if (isLoading && projects.length === 0) {
    return <div className="p-3 text-sm text-gray-500 text-center">加载中...</div>;
  }

  if (error) {
    return <div className="p-3 text-sm text-red-500">{error}</div>;
  }

  if (projects.length === 0) {
    return <div className="p-3 text-sm text-gray-400 text-center">暂无项目</div>;
  }

  return (
    <div className="divide-y divide-gray-100">
      {projects.map((project) => (
        <div
          key={project.id}
          onClick={() => handleSwitch(project.id)}
          className={`group flex items-center justify-between px-4 py-2.5 cursor-pointer transition-colors ${
            currentProjectId === project.id
              ? 'bg-blue-50 text-blue-700'
              : 'hover:bg-gray-50 text-gray-700'
          }`}
        >
          <span className="text-sm font-medium truncate">{project.name}</span>
          <button
            onClick={(e) => handleDelete(project.id, e)}
            className={`shrink-0 p-1 rounded hover:bg-gray-200 transition-opacity ${
              confirmDelete === project.id ? 'opacity-100 bg-red-100 text-red-600' : 'opacity-0 group-hover:opacity-100'
            }`}
            title={confirmDelete === project.id ? '再次点击确认删除' : '删除项目'}
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      ))}
    </div>
  );
}
