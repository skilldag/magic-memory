import { create } from 'zustand';
import type { Project } from '../types';

interface ProjectStore {
  projects: Project[];
  currentProjectId: string | null;
  isLoading: boolean;
  error: string | null;

  // Actions
  loadProjects: () => Promise<void>;
  createProject: (name: string, folderPath: string) => Promise<Project | null>;
  deleteProject: (projectId: string) => Promise<boolean>;
  switchProject: (projectId: string) => Promise<void>;
  clearError: () => void;
}

export const useProjectStore = create<ProjectStore>((set, get) => ({
  projects: [],
  currentProjectId: null,
  isLoading: false,
  error: null,

  loadProjects: async () => {
    set({ isLoading: true, error: null });
    try {
      const resp = await fetch('/api/projects');
      if (!resp.ok) throw new Error('Failed to load projects');
      const data = await resp.json();
      set({ projects: data.projects || [], isLoading: false });

      const { currentProjectId, projects } = get();
      if (!currentProjectId && projects.length > 0) {
        get().switchProject(projects[0].id);
      }
    } catch (error) {
      set({ error: error instanceof Error ? error.message : 'Unknown error', isLoading: false });
    }
  },

  createProject: async (name: string, folderPath: string) => {
    set({ isLoading: true, error: null });
    try {
      const resp = await fetch('/api/projects', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, folderPath }),
      });
      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.error || 'Failed to create project');
      }
      const data = await resp.json();
      const { projects } = get();
      set({ projects: [...projects, data.project], isLoading: false });
      get().switchProject(data.project.id);
      return data.project;
    } catch (error) {
      set({ error: error instanceof Error ? error.message : 'Unknown error', isLoading: false });
      return null;
    }
  },

  deleteProject: async (projectId: string) => {
    set({ isLoading: true, error: null });
    try {
      const resp = await fetch(`/api/projects/${projectId}`, { method: 'DELETE' });
      if (!resp.ok) throw new Error('Failed to delete project');
      const { projects, currentProjectId } = get();
      set({
        projects: projects.filter(p => p.id !== projectId),
        currentProjectId: currentProjectId === projectId ? null : currentProjectId,
        isLoading: false,
      });
      return true;
    } catch (error) {
      set({ error: error instanceof Error ? error.message : 'Unknown error', isLoading: false });
      return false;
    }
  },

  switchProject: async (projectId: string) => {
    set({ isLoading: true, error: null });
    try {
      const resp = await fetch(`/api/projects/${projectId}`);
      if (!resp.ok) throw new Error('Failed to load project');
      const data = await resp.json();

      const { projects } = get();
      const updatedProjects = projects.map(p =>
        p.id === projectId ? { ...p, lastOpenedAt: data.project.lastOpenedAt } : p
      );
      set({ projects: updatedProjects, currentProjectId: projectId, isLoading: false });

      // Update knowledge graph store with project data
      const { useKnowledgeGraphStore } = await import('./knowledgeGraphStore');
      useKnowledgeGraphStore.setState({
        concepts: data.concepts || [],
        edges: data.edges || [],
      });
    } catch (error) {
      set({ error: error instanceof Error ? error.message : 'Unknown error', isLoading: false });
    }
  },

  clearError: () => set({ error: null }),
}));
