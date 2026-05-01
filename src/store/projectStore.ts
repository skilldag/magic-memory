import { create } from 'zustand';
import type { Project } from '../types';
import { saveHandle, loadHandle, deleteHandle, ensurePermission } from '../utils/handleStorage';
import { readMdFiles } from '../utils/fileSystem';

interface ProjectStore {
  projects: Project[];
  currentProjectId: string | null;
  isLoading: boolean;
  isScanning: boolean;
  error: string | null;

  loadProjects: () => Promise<void>;
  createProject: (name: string, handle: FileSystemDirectoryHandle) => Promise<Project | null>;
  deleteProject: (projectId: string) => Promise<boolean>;
  switchProject: (projectId: string) => Promise<void>;
  clearError: () => void;
}

function generateId(): string {
  return `proj_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

export const useProjectStore = create<ProjectStore>((set, get) => ({
  projects: [],
  currentProjectId: null,
  isLoading: false,
  isScanning: false,
  error: null,

  loadProjects: async () => {
    set({ isLoading: true, error: null });
    try {
      const resp = await fetch('/api/projects');
      if (resp.ok) {
        const data = await resp.json();
        set({ projects: data.projects || [], isLoading: false });
        const { currentProjectId, projects } = get();
        if (!currentProjectId && projects.length > 0) {
          get().switchProject(projects[0].id);
        }
        return;
      }
    } catch {}
    // 后端不可用时，从 localStorage 恢复
    const stored = localStorage.getItem('magic-memory-projects');
    if (stored) {
      try {
        const projects: Project[] = JSON.parse(stored);
        set({ projects, isLoading: false });
        return;
      } catch {}
    }
    set({ isLoading: false });
  },

  createProject: async (name: string, handle: FileSystemDirectoryHandle) => {
    set({ isLoading: true, error: null, isScanning: true });
    try {
      const handleStoreId = generateId();
      await saveHandle(handleStoreId, handle);

      const project: Project = {
        id: generateId(),
        name,
        handleStoreId,
        createdAt: new Date().toISOString(),
        lastOpenedAt: new Date().toISOString(),
      };

      const ok = await ensurePermission(handle);
      if (!ok) {
        set({ error: '没有文件夹读取权限', isLoading: false, isScanning: false });
        return null;
      }

      // 读文件构建概念
      const files = await readMdFiles(handle);
      const concepts: any[] = [];
      for (const file of files) {
        if (!file.path.endsWith('.md')) continue;
        const id = file.path.replace('.md', '').replace(/\//g, '-');
        const title = file.path.replace('.md', '').split('/').pop() || file.path.replace('.md', '');
        concepts.push({
          id,
          title,
          path: file.path,
          level: 1,
          category: '',
          problem: '',
          gap_anticipate: '',
          depends_on: [],
          leads_to: [],
          related: [],
          tags: [],
          lastModified: new Date(),
        })
      }

      const { useKnowledgeGraphStore } = await import('./knowledgeGraphStore');
      useKnowledgeGraphStore.setState({ concepts, edges: [], isLoading: false });

      // 持久化项目列表
      const { projects } = get();
      const updated = [...projects, project];
      set({ projects: updated, currentProjectId: project.id, isLoading: false, isScanning: false });
      localStorage.setItem('magic-memory-projects', JSON.stringify(updated));

      try {
        await fetch('/api/projects', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name, handleStoreId }),
        });
      } catch {}

      return project;
    } catch (error) {
      set({ error: error instanceof Error ? error.message : '创建项目失败', isLoading: false, isScanning: false });
      return null;
    }
  },

  deleteProject: async (projectId: string) => {
    const { projects, currentProjectId } = get();
    const project = projects.find(p => p.id === projectId);
    if (project?.handleStoreId) {
      await deleteHandle(project.handleStoreId).catch(() => {});
    }
    const filtered = projects.filter(p => p.id !== projectId);
    set({
      projects: filtered,
      currentProjectId: currentProjectId === projectId ? null : currentProjectId,
    });
    localStorage.setItem('magic-memory-projects', JSON.stringify(filtered));
    try { await fetch(`/api/projects/${projectId}`, { method: 'DELETE' }); } catch {}
    return true;
  },

  switchProject: async (projectId: string) => {
    set({ isLoading: true, error: null });
    try {
      const { projects } = get();
      const project = projects.find(p => p.id === projectId);
      if (!project) throw new Error('项目不存在');

      if (project.handleStoreId) {
        const handle = await loadHandle(project.handleStoreId);
        if (!handle) throw new Error('项目文件夹句柄已丢失，请重新选择');

        const ok = await ensurePermission(handle);
        if (!ok) throw new Error('请授权文件夹读取权限');

      const files = await readMdFiles(handle);
      const concepts: any[] = [];
      for (const file of files) {
        if (!file.path.endsWith('.md')) continue;
        const id = file.path.replace('.md', '').replace(/\//g, '-');
        const title = file.path.replace('.md', '').split('/').pop() || file.path.replace('.md', '');
        concepts.push({
          id,
          title,
          path: file.path,
          level: 1,
          category: '',
          problem: '',
          gap_anticipate: '',
          depends_on: [],
          leads_to: [],
          related: [],
          tags: [],
          lastModified: new Date(),
        })
      }

      const { useKnowledgeGraphStore } = await import('./knowledgeGraphStore');
      useKnowledgeGraphStore.setState({ concepts, edges: [], isLoading: false });
      }

      set({
        currentProjectId: projectId,
        projects: projects.map(p => p.id === projectId ? { ...p, lastOpenedAt: new Date().toISOString() } : p),
        isLoading: false,
      });
    } catch (error) {
      set({ error: error instanceof Error ? error.message : '切换项目失败', isLoading: false });
    }
  },

  clearError: () => set({ error: null }),
}));
