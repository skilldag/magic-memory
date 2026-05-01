import { create } from 'zustand';
import type { Project } from '../types';
import { saveHandle, loadHandle, deleteHandle, ensurePermission } from '../utils/handleStorage';
import { readMdFiles } from '../utils/fileSystem';
import { parseFrontmatter, matchTitlesToIds } from '../utils/conceptParser';

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

      // 读文件解析概念
      const files = await readMdFiles(handle);
      const concepts: any[] = [];
      for (const file of files) {
        const parsed = parseFrontmatter(file.content);
        const hasFm = parsed.meta && Object.keys(parsed.meta).length > 0;
        if (hasFm) {
          const meta = parsed.meta as any;
          concepts.push({
            id: meta.id || file.path.replace('.md', '').replace(/\//g, '-'),
            title: meta.title || file.path.replace('.md', ''),
            path: file.path,
            level: meta.level ?? 1,
            category: meta.category || '',
            problem: meta.problem || '',
            gap_anticipate: meta.gap_anticipate || '',
            depends_on: meta.depends_on || [],
            leads_to: meta.leads_to || [],
            related: meta.related || [],
            alias: meta.alias,
            tags: meta.tags || [],
            lastModified: new Date(),
          })
        }
      }

      // 构建边
      const built = concepts.map((c: any) => ({
        ...c,
        depends_on: matchTitlesToIds(c.depends_on, concepts),
        leads_to: matchTitlesToIds(c.leads_to, concepts),
        related: matchTitlesToIds(c.related, concepts),
      }));
      const ids = new Set(built.map((c: any) => c.id));
      const edges: any[] = [];
      const edgeSet = new Set<string>();
      for (const c of built) {
        for (const t of c.leads_to) {
          if (ids.has(t)) { const eid = `${c.id}-leads-${t}`; if (!edgeSet.has(eid)) { edgeSet.add(eid); edges.push({ id: eid, source: c.id, target: t, type: 'leads_to' }); } }
        }
        for (const t of c.depends_on) {
          if (ids.has(t)) { const eid = `${c.id}-depends-${t}`; if (!edgeSet.has(eid)) { edgeSet.add(eid); edges.push({ id: eid, source: c.id, target: t, type: 'depends_on' }); } }
        }
        for (const t of c.related) {
          if (ids.has(t)) { const eid = `${c.id}-related-${t}`; if (!edgeSet.has(eid)) { edgeSet.add(eid); edges.push({ id: eid, source: c.id, target: t, type: 'related' }); } }
        }
      }

      // 更新 knowledgeGraphStore
      const { useKnowledgeGraphStore } = await import('./knowledgeGraphStore');
      useKnowledgeGraphStore.setState({ concepts: built, edges, isLoading: false });

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
          const parsed = parseFrontmatter(file.content);
          if (parsed.meta && Object.keys(parsed.meta).length > 0) {
            const meta = parsed.meta as any;
            concepts.push({
              id: meta.id || file.path.replace('.md', '').replace(/\//g, '-'),
              title: meta.title || file.path.replace('.md', ''),
              path: file.path,
              level: meta.level ?? 1,
              category: meta.category || '',
              problem: meta.problem || '',
              depends_on: meta.depends_on || [],
              leads_to: meta.leads_to || [],
              related: meta.related || [],
              tags: meta.tags || [],
              lastModified: new Date(),
            });
          }
        }

        const built = concepts.map((c: any) => ({
          ...c,
          depends_on: matchTitlesToIds(c.depends_on, concepts),
          leads_to: matchTitlesToIds(c.leads_to, concepts),
          related: matchTitlesToIds(c.related, concepts),
        }));
        const ids = new Set(built.map((c: any) => c.id));
        const edges: any[] = [];
        const edgeSet = new Set<string>();
        for (const c of built) {
          for (const t of c.leads_to) { if (ids.has(t)) { const eid = `${c.id}-leads-${t}`; if (!edgeSet.has(eid)) { edgeSet.add(eid); edges.push({ id: eid, source: c.id, target: t, type: 'leads_to' }); } } }
          for (const t of c.depends_on) { if (ids.has(t)) { const eid = `${c.id}-depends-${t}`; if (!edgeSet.has(eid)) { edgeSet.add(eid); edges.push({ id: eid, source: c.id, target: t, type: 'depends_on' }); } } }
          for (const t of c.related) { if (ids.has(t)) { const eid = `${c.id}-related-${t}`; if (!edgeSet.has(eid)) { edgeSet.add(eid); edges.push({ id: eid, source: c.id, target: t, type: 'related' }); } } }
        }

        const { useKnowledgeGraphStore } = await import('./knowledgeGraphStore');
        useKnowledgeGraphStore.setState({ concepts: built, edges, isLoading: false });
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
