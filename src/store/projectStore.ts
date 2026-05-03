import { create } from 'zustand';
import type { Project, ConceptEdge, ReviewRecord, UserAnnotation, ProcessChain } from '../types';
import { saveHandle, loadHandle, deleteHandle, ensurePermission } from '../utils/handleStorage';

const GS_URL = 'http://localhost:4321';

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
    set({ isLoading: true, error: null });

    // Try Global Service first (CLI-registered projects)
    try {
      const resp = await fetch(`${GS_URL}/api/projects`);
      if (resp.ok) {
        const data = await resp.json();
        const gsProjects: Project[] = (data.projects || []).map((p: any) => ({
          id: p.id,
          name: p.name,
          folderPath: p.sourceDir,
          handleStoreId: null,
          createdAt: p.createdAt,
          lastOpenedAt: p.createdAt,
        }));
        set({ projects: gsProjects, isLoading: false });
        localStorage.setItem('magic-memory-projects', JSON.stringify(gsProjects));

        const { currentProjectId } = get();
        if (!currentProjectId && gsProjects.length > 0) {
          get().switchProject(gsProjects[0].id);
        }
        return;
      }
    } catch {
      // GS unavailable, fallback to old approach
    }

    // Fallback: localStorage projects (old approach)
    let localProjects: Project[] = [];
    const stored = localStorage.getItem('magic-memory-projects');
    if (stored) {
      try {
        localProjects = JSON.parse(stored);
      } catch {}
    }

    try {
      const resp = await fetch('/api/projects');
      if (resp.ok) {
        const data = await resp.json();
        const serverProjects: Project[] = data.projects || [];
        const merged = mergeProjects(serverProjects, localProjects);
        set({ projects: merged, isLoading: false });
        localStorage.setItem('magic-memory-projects', JSON.stringify(merged));

        const { currentProjectId, projects } = get();
        if (!currentProjectId && projects.length > 0) {
          const { useKnowledgeGraphStore } = await import('./knowledgeGraphStore');
          const kg = useKnowledgeGraphStore.getState();
          const snapshots = loadSnapshots();
          snapshots[projects[0].id] = {
            edges: kg.edges,
            reviewRecords: Array.from(kg.reviewRecords.entries()),
            annotations: kg.annotations,
            chains: kg.chains,
          };
          saveSnapshots(snapshots);
          get().switchProject(projects[0].id);
        }
        return;
      }
    } catch {}

    if (localProjects.length > 0) {
      set({ projects: localProjects, isLoading: false });
      const { currentProjectId } = get();
      if (!currentProjectId) {
        await get().switchProject(localProjects[0].id);
      }
      return;
    }

    set({ isLoading: false });
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

      // Phase 1: Quick metadata scan — enumerate file paths only
      const kgStore = (await import('./knowledgeGraphStore')).useKnowledgeGraphStore;
      const { scanMdPaths, readMdFilesBatched } = await import('../utils/fileSystem');

      kgStore.setState({ isLoading: false, loadingProgress: 0 });
      console.time('[perf] scanMdPaths');
      const pathDefs: { path: string; id: string; title: string }[] = [];
      for await (const batch of scanMdPaths(handle)) {
        for (const entryPath of batch) {
          const id = entryPath.replace('.md', '').replace(/\//g, '-');
          const title = entryPath.replace('.md', '').split('/').pop() || entryPath.replace('.md', '');
          pathDefs.push({ path: entryPath, id, title });
        }
      }
      console.timeEnd('[perf] scanMdPaths');

      // Build concepts from metadata and render immediately
      const concepts: any[] = pathDefs.map(({ path, id, title }) => ({
        id, title, path,
        level: 1, category: '', problem: '', gap_anticipate: '',
        depends_on: [], leads_to: [], related: [], tags: [],
        lastModified: new Date(),
      }));
      kgStore.setState({ concepts, edges: [], loadingProgress: 0, activeProjectId: project.id });

      // Phase 2: Background content scanning + edge derivation
      kgStore.setState({ loadingProgress: 5 });
      const allFiles: { path: string; content: string }[] = [];
      const totalPaths = pathDefs.length;
      let readCount = 0;

      console.time('[perf] readMdFilesBatched');
      for await (const batch of readMdFilesBatched(handle)) {
        allFiles.push(...batch);
        readCount += batch.length;
        const progress = Math.min(5 + Math.round((readCount / totalPaths) * 70), 75);
        kgStore.setState({ loadingProgress: progress });
      }
      console.timeEnd('[perf] readMdFilesBatched');

      kgStore.setState({ loadingProgress: 80 });
      console.time('[perf] deriveEdgesInWorker');
      const { deriveEdgesInWorker } = await import('../workers/deriveEdges.worker');
      const derivedEdges = await deriveEdgesInWorker(concepts, allFiles);
      console.timeEnd('[perf] deriveEdgesInWorker');

      const snapshot = loadSnapshots()[project.id];
      const edges = snapshot?.edges?.length ? snapshot.edges : derivedEdges;
      kgStore.setState({ edges, loadingProgress: 100 });

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
    try { await fetch(`${GS_URL}/api/projects/${projectId}`, { method: 'DELETE' }); } catch {}
    return true;
  },

  switchProject: async (projectId: string) => {
    set({ isLoading: true, error: null });
    try {
      const { projects, currentProjectId } = get();
      const project = projects.find(p => p.id === projectId);
      if (!project) throw new Error('项目不存在');

      if (currentProjectId !== projectId) {
        const { useKnowledgeGraphStore } = await import('./knowledgeGraphStore');
        const kg = useKnowledgeGraphStore.getState();
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

      // GS-registered project (no handle) → load from Global Service
      if (!project.handleStoreId) {
        const { useKnowledgeGraphStore } = await import('./knowledgeGraphStore');
        await useKnowledgeGraphStore.getState().loadProjectGraph(projectId);
        set({ currentProjectId: projectId, isLoading: false });
        return;
      }

      // Old browser-based project (has handle) → use File System API
      if (project.handleStoreId) {
        const handle = await loadHandle(project.handleStoreId);
        if (!handle) throw new Error('项目文件夹句柄已丢失，请重新选择');
        const ok = await ensurePermission(handle);
        if (!ok) throw new Error('请授权文件夹读取权限');

        const kgStore = (await import('./knowledgeGraphStore')).useKnowledgeGraphStore;
        const { scanMdPaths, readMdFilesBatched } = await import('../utils/fileSystem');
        const snapshot = loadSnapshots()[projectId];

        kgStore.setState({ isLoading: false, loadingProgress: 0 });
        const pathDefs: { path: string; id: string; title: string }[] = [];
        for await (const batch of scanMdPaths(handle)) {
          for (const entryPath of batch) {
            const id = entryPath.replace('.md', '').replace(/\//g, '-');
            const title = entryPath.replace('.md', '').split('/').pop() || entryPath.replace('.md', '');
            pathDefs.push({ path: entryPath, id, title });
          }
        }

        const concepts = pathDefs.map(({ path, id, title }) => ({
          id, title, path,
          level: 1, category: '', problem: '', gap_anticipate: '',
          depends_on: [], leads_to: [], related: [], tags: [],
          lastModified: new Date(),
        }));
        kgStore.setState({
          concepts,
          edges: snapshot?.edges || [],
          reviewRecords: new Map(snapshot?.reviewRecords || []),
          annotations: snapshot?.annotations || [],
          chains: snapshot?.chains || [],
          loadingProgress: 0,
          activeProjectId: projectId,
        });

        if (!snapshot?.edges?.length) {
          kgStore.setState({ loadingProgress: 5 });
          const allFiles: { path: string; content: string }[] = [];
          const totalPaths = pathDefs.length;
          let readCount = 0;

          for await (const batch of readMdFilesBatched(handle)) {
            allFiles.push(...batch);
            readCount += batch.length;
            const progress = Math.min(5 + Math.round((readCount / totalPaths) * 70), 75);
            kgStore.setState({ loadingProgress: progress });
          }

          kgStore.setState({ loadingProgress: 80 });
          const { deriveEdgesInWorker } = await import('../workers/deriveEdges.worker');
          const derivedEdges = await deriveEdgesInWorker(concepts, allFiles);
          kgStore.setState({ edges: derivedEdges, loadingProgress: 100 });
        } else {
          kgStore.setState({ loadingProgress: 100 });
        }
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
