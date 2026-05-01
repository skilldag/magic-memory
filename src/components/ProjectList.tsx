import React, { useEffect, useState } from 'react';
import { useProjectStore } from '../store/projectStore';

interface ProjectListProps {
  onAddProject: () => void;
}

export function ProjectList({ onAddProject }: ProjectListProps) {
  const { projects, currentProjectId, switchProject, deleteProject, isLoading, error, loadProjects } = useProjectStore();
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);

  useEffect(() => {
    loadProjects();
  }, [loadProjects]);

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
    return <div className="p-2 text-sm text-gray-500">Loading...</div>;
  }

  if (error) {
    return <div className="p-2 text-sm text-red-500">{error}</div>;
  }

  return (
    <div className="flex flex-col h-full">
      <div className="text-xs font-semibold text-gray-500 px-2 py-1 uppercase tracking-wider">
        项目列表
      </div>
      <div className="flex-1 overflow-y-auto">
        {projects.map((project) => (
          <div
            key={project.id}
            onClick={() => switchProject(project.id)}
            className={`group flex items-center justify-between px-2 py-1.5 cursor-pointer rounded mx-1 ${
              currentProjectId === project.id
                ? 'bg-blue-50 text-blue-700'
                : 'hover:bg-gray-100 text-gray-700'
            }`}
          >
            <div className="flex items-center gap-2 min-w-0">
              <span className="text-sm truncate">
                {currentProjectId === project.id && '\u2713 '}
                {project.name}
              </span>
            </div>
            <button
              onClick={(e) => handleDelete(project.id, e)}
              className={`opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-gray-200 transition-opacity ${
                confirmDelete === project.id ? 'opacity-100 bg-red-100 text-red-600' : ''
              }`}
              title={confirmDelete === project.id ? '\u518d\u6b21\u70b9\u51fb\u786e\u8ba4\u5220\u9664' : '\u5220\u9664\u9879\u76ee'}
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        ))}
      </div>
      <button
        onClick={onAddProject}
        className="flex items-center gap-2 px-2 py-2 mx-1 mt-1 text-sm text-blue-600 hover:bg-blue-50 rounded border border-dashed border-blue-200"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
        </svg>
        添加项目文件夹
      </button>
    </div>
  );
}

export default ProjectList;
