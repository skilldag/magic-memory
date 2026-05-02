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

  const handleDeleteClick = (projectId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setConfirmDelete(projectId);
  };

  const handleConfirmDelete = async (projectId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    await deleteProject(projectId);
    setConfirmDelete(null);
  };

  const handleCancelDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    setConfirmDelete(null);
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
          className={`flex items-center justify-between px-4 py-2.5 cursor-pointer transition-colors ${
            currentProjectId === project.id
              ? 'bg-blue-50 text-blue-700'
              : 'hover:bg-gray-50 text-gray-700'
          }`}
        >
          <span className="text-sm font-medium truncate">{project.name}</span>
          {confirmDelete === project.id ? (
            <div className="flex items-center gap-1 shrink-0" onClick={e => e.stopPropagation()}>
              <span className="text-xs text-red-600 font-medium">确认删除?</span>
              <button
                onClick={(e) => handleConfirmDelete(project.id, e)}
                className="px-2 py-1 text-xs font-medium text-white bg-red-500 rounded hover:bg-red-600 transition-colors"
              >
                删除
              </button>
              <button
                onClick={handleCancelDelete}
                className="px-2 py-1 text-xs font-medium text-gray-600 bg-gray-100 rounded hover:bg-gray-200 transition-colors"
              >
                取消
              </button>
            </div>
          ) : (
            <button
              onClick={(e) => handleDeleteClick(project.id, e)}
              className="shrink-0 px-2 py-1 text-xs font-medium text-gray-400 hover:text-red-500 hover:bg-red-50 rounded transition-colors"
              title="删除项目"
            >
              删除
            </button>
          )}
        </div>
      ))}
    </div>
  );
}
