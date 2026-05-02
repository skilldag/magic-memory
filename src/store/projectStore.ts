import { create } from 'zustand';
import type { Project, ConceptEdge, ReviewRecord, UserAnnotation, ProcessChain } from '../types';
import { saveHandle, loadHandle, deleteHandle, ensurePermission } from '../utils/handleStorage';

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

interface GraphSnapshot {
  edges: ConceptEdge[];
  reviewRecords: [string, ReviewRecord][];
  annotations: UserAnnotation[];
  chains: ProcessChain[];
}

function generateId(): string {
  return `proj_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

function mergeProjects(server: Project[], local: Project[]): Project[] {
  const seen = new Set<string>();
  const result: Project[] = [];

  for (const p of [...server, ...local]) {
    if (!seen.has(p.id)) {
      seen.add(p.id);
      result.push(p);
    }
  }
  return result;
}

function loadSnapshots(): Record<string, GraphSnapshot> {
  try {
    const raw = localStorage.getItem('magic-memory-graph-snapshots');
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function saveSnapshots(snapshots: Record<string, GraphSnapshot>) {
  localStorage.setItem('magic-memory-graph-snapshots', JSON.stringify(snapshots));
}

export const useProjectStore = create<ProjectStore>((set, get) => ({
  projects: [],
  currentProjectId: null,
  isLoading: false,
  isScanning: false,
  error: null,

  loadProjects: async () => {
    console.time('[perf] loadProjects total');
    set({ isLoading: true, error: null });

    let localProjects: Project[] = [];
    const stored = localStorage.getItem('magic-memory-projects');
    console.log('[loadProjects] stored projects:', stored);
    if (stored) {
      try {
        localProjects = JSON.parse(stored);
      } catch {}
    }

    try {
      console.time('[perf] fetch /api/projects');
      const resp = await fetch('/api/projects');
      console.timeEnd('[perf] fetch /api/projects');
      if (resp.ok) {
        const data = await resp.json();
        const serverProjects: Project[] = data.projects || [];
        const merged = mergeProjects(serverProjects, localProjects);
        console.log('[loadProjects] merged projects:', merged.length, merged.map(p => p.id));
        set({ projects: merged, isLoading: false });
        localStorage.setItem('magic-memory-projects', JSON.stringify(merged));

        const { currentProjectId, projects } = get();
        console.log('[loadProjects] currentProjectId:', currentProjectId, 'projects:', projects.length);
        if (!currentProjectId && projects.length > 0) {
          const { useKnowledgeGraphStore } = await import('./knowledgeGraphStore');
          const kg = useKnowledgeGraphStore.getState();
          console.log('[loadProjects] KG state before snapshot: edges=', kg.edges.length, 'concepts=', kg.concepts.length);
          const snapshots = loadSnapshots();
          snapshots[projects[0].id] = {
            edges: kg.edges,
            reviewRecords: Array.from(kg.reviewRecords.entries()),
            annotations: kg.annotations,
            chains: kg.chains,
          };
          saveSnapshots(snapshots);
          console.log('[loadProjects] saved snapshot for', projects[0].id, 'with', kg.edges.length, 'edges');
          get().switchProject(projects[0].id);
        }
        console.timeEnd('[perf] loadProjects total');
        return;
      }
    } catch {
      console.log('[loadProjects] server unavailable, using localStorage');
    }

    if (localProjects.length > 0) {
      console.log('[loadProjects] fallback to localStorage projects:', localProjects.length);
      set({ projects: localProjects, isLoading: false });
      console.timeEnd('[perf] loadProjects total');
      return;
    }

    set({ isLoading: false });
    console.timeEnd('[perf] loadProjects total');
  },

  createProject: async (name: string, handle: FileSystemDirectoryHandle) => {
    console.time('[perf] createProject total');
    set({ isLoading: true, error: null, isScanning: true });
    try {
      const handleStoreId = generateId();

      const { projects } = get();
      const existing = projects.find(p => p.name === name);
      if (existing) {
        set({ currentProjectId: existing.id, isLoading: false, isScanning: false });
        return existing;
      }

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

      // Batch read files (accumulate locally, single setState at end)
      const allConcepts: any[] = [];
      const allFiles: { path: string; content: string }[] = [];
      const { readMdFilesBatched } = await import('../utils/fileSystem');
      const kgStore = (await import('./knowledgeGraphStore')).useKnowledgeGraphStore;

      kgStore.setState({ isLoading: true });
      console.time('[perf] readMdFilesBatched');
      for await (const batch of readMdFilesBatched(handle)) {
        allFiles.push(...batch);
        for (const file of batch) {
          allConcepts.push({
            id: file.path.replace('.md', '').replace(/\//g, '-'),
            title: file.path.replace('.md', '').split('/').pop() || file.path.replace('.md', ''),
            path: file.path,
            level: 1, category: '', problem: '', gap_anticipate: '',
            depends_on: [], leads_to: [], related: [], tags: [],
            lastModified: new Date(),
          });
        }
      }
      console.timeEnd('[perf] readMdFilesBatched');

      const snapshot = loadSnapshots()[project.id];
      console.time('[perf] deriveEdgesInWorker');
      const { deriveEdgesInWorker } = await import('../workers/deriveEdges.worker');
      const derivedEdges = await deriveEdgesInWorker(allConcepts, allFiles);
      console.timeEnd('[perf] deriveEdgesInWorker');
      const edges = snapshot?.edges?.length ? snapshot.edges : derivedEdges;
      console.log('[createProject] edges: snapshot:', snapshot?.edges?.length, 'derived:', derivedEdges.length, 'final:', edges.length, 'project:', project.id);
      kgStore.setState({
        concepts: allConcepts,
        edges,
        reviewRecords: new Map(snapshot?.reviewRecords || []),
        annotations: snapshot?.annotations || [],
        chains: snapshot?.chains || [],
        isLoading: false,
      });

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

      console.timeEnd('[perf] createProject total');
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

    const snapshots = loadSnapshots();
    delete snapshots[projectId];
    saveSnapshots(snapshots);

    try { await fetch(`/api/projects/${projectId}`, { method: 'DELETE' }); } catch {}
    return true;
  },

  switchProject: async (projectId: string) => {
    console.time('[perf] switchProject total');
    set({ isLoading: true, error: null });
    try {
      const { projects, currentProjectId } = get();
      const project = projects.find(p => p.id === projectId);
      if (!project) throw new Error('项目不存在');

      console.log('[switchProject] from:', currentProjectId, 'to:', projectId);
      if (currentProjectId !== projectId) {
        const { useKnowledgeGraphStore } = await import('./knowledgeGraphStore');
        const kg = useKnowledgeGraphStore.getState();
        console.log('[switchProject] kg.edges before save:', kg.edges.length, 'sourceId:', currentProjectId || projectId);
        const snapshots = loadSnapshots();
        const sourceId = currentProjectId || projectId;
        snapshots[sourceId] = {
          edges: kg.edges,
          reviewRecords: Array.from(kg.reviewRecords.entries()),
          annotations: kg.annotations,
          chains: kg.chains,
        };
        saveSnapshots(snapshots);
      }

      if (project.handleStoreId) {
        const handle = await loadHandle(project.handleStoreId);
        if (!handle) throw new Error('项目文件夹句柄已丢失，请重新选择');
        const ok = await ensurePermission(handle);
        if (!ok) throw new Error('请授权文件夹读取权限');

        // Batch read files (accumulate locally, single setState at end)
        const allConcepts: any[] = [];
        const allFiles: { path: string; content: string }[] = [];
        const { readMdFilesBatched } = await import('../utils/fileSystem');
        const kgStore = (await import('./knowledgeGraphStore')).useKnowledgeGraphStore;

        kgStore.setState({ isLoading: true });
        console.time('[perf] readMdFilesBatched');
        for await (const batch of readMdFilesBatched(handle)) {
          allFiles.push(...batch);
          for (const file of batch) {
            allConcepts.push({
              id: file.path.replace('.md', '').replace(/\//g, '-'),
              title: file.path.replace('.md', '').split('/').pop() || file.path.replace('.md', ''),
              path: file.path,
              level: 1, category: '', problem: '', gap_anticipate: '',
              depends_on: [], leads_to: [], related: [], tags: [],
              lastModified: new Date(),
            });
          }
        }
        console.timeEnd('[perf] readMdFilesBatched');

        const snapshot = loadSnapshots()[projectId];
        console.time('[perf] deriveEdgesInWorker');
        const { deriveEdgesInWorker } = await import('../workers/deriveEdges.worker');
        const derivedEdges = await deriveEdgesInWorker(allConcepts, allFiles);
        console.timeEnd('[perf] deriveEdgesInWorker');
        const restoredEdges = snapshot?.edges?.length ? snapshot.edges : derivedEdges;
        console.log('[switchProject] edges: snapshot:', snapshot?.edges?.length, 'derived:', derivedEdges.length, 'final:', restoredEdges.length, 'for project:', projectId);

        kgStore.setState({
          concepts: allConcepts,
          edges: restoredEdges,
          reviewRecords: new Map(snapshot?.reviewRecords || []),
          annotations: snapshot?.annotations || [],
          chains: snapshot?.chains || [],
          isLoading: false,
        });
      }

      set({
        currentProjectId: projectId,
        projects: projects.map(p => p.id === projectId ? { ...p, lastOpenedAt: new Date().toISOString() } : p),
        isLoading: false,
      });
      console.timeEnd('[perf] switchProject total');
    } catch (error) {
      set({ error: error instanceof Error ? error.message : '切换项目失败', isLoading: false });
      console.timeEnd('[perf] switchProject total');
    }
  },

  clearError: () => set({ error: null }),
}));
