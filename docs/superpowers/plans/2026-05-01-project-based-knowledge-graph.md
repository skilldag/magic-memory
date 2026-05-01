# Project-Based Knowledge Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable users to create multiple projects, each with its own knowledge graph stored in `~/.magic-memory/projects/{project-id}/`, with UI to switch between projects.

**Architecture:** Server-side file-based storage with RESTful API. Frontend extends Zustand store to manage projects and project-specific graph data. UI adds project list sidebar in document view.

**Tech Stack:** Bun, TypeScript, React, Zustand, File System Access API

---

## File Structure

### New Files
- `src/types/project.ts` - Project-related type definitions
- `src/components/ProjectList.tsx` - Project list UI component
- `src/store/projectStore.ts` - Project management store (new file)

### Modified Files
- `server.ts` - Add project management API endpoints
- `src/store/knowledgeGraphStore.ts` - Add project state and methods
- `src/components/Sidebar.tsx` - Add project list panel
- `src/types/index.ts` - Add Project type export

---

## Task 1: Define Project Types

**Files:**
- Create: `src/types/project.ts`
- Test: N/A (type definition only)

- [ ] **Step 1: Create project types**

```typescript
// src/types/project.ts
export interface Project {
  id: string;
  name: string;
  folderPath: string;
  createdAt: string;
  lastOpenedAt: string;
}

export interface ProjectConfig {
  id: string;
  name: string;
  folderPath: string;
  createdAt: string;
  lastOpenedAt: string;
}

export interface ProjectGraphData {
  concepts: import('./index').Concept[];
  edges: import('./index').ConceptEdge[];
}
```

- [ ] **Step 2: Export types from index.ts**

Add to `src/types/index.ts`:
```typescript
export type { Project, ProjectConfig, ProjectGraphData } from './project';
```

- [ ] **Step 3: Commit**
```bash
git add src/types/project.ts src/types/index.ts
git commit -m "feat: add Project type definitions"
```

---

## Task 2: Server-Side Project Management

**Files:**
- Modify: `server.ts` (lines 1-20, add new constants)
- Modify: `server.ts` (lines 418-856, add new API endpoints)
- Test: Manual via curl

- [ ] **Step 1: Add project-related constants at top of server.ts**

After existing constants (around line 13), add:
```typescript
// Project storage
const MAGIC_MEMORY_DIR = join(os.homedir(), '.magic-memory');
const PROJECTS_DIR = join(MAGIC_MEMORY_DIR, 'projects');
const PROJECT_LIST_FILE = join(PROJECTS_DIR, 'project-list.json');

// Ensure directories exist
function ensureProjectDirs() {
  if (!existsSync(PROJECTS_DIR)) {
    mkdirSync(PROJECTS_DIR, { recursive: true });
  }
}
ensureProjectDirs();
```

- [ ] **Step 2: Add project helper functions** (before `const server = serve`)

```typescript
// Project list management
async function loadProjectList(): Promise<Project[]> {
  try {
    if (!existsSync(PROJECT_LIST_FILE)) return [];
    const content = await readFile(PROJECT_LIST_FILE, 'utf-8');
    return JSON.parse(content);
  } catch {
    return [];
  }
}

async function saveProjectList(projects: Project[]): Promise<void> {
  await writeFile(PROJECT_LIST_FILE, JSON.stringify(projects, null, 2));
}

async function createProjectDir(projectId: string): Promise<string> {
  const projectDir = join(PROJECTS_DIR, projectId);
  mkdirSync(projectDir, { recursive: true });
  return projectDir;
}

function generateProjectId(): string {
  return `proj_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}
```

- [ ] **Step 3: Add project API endpoints** (before the final `return new Response('Not found', status: 404)`)

Insert after line 850 (before the 404 response):

```typescript
// GET /api/projects — 获取项目列表
if (url.pathname === '/api/projects' && req.method === 'GET') {
  try {
    const projects = await loadProjectList();
    return new Response(JSON.stringify({ projects }), {
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (error) {
    return new Response(JSON.stringify({ error: String(error) }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}

// POST /api/projects — 创建新项目
if (url.pathname === '/api/projects' && req.method === 'POST') {
  try {
    const body = await req.json();
    const { name, folderPath } = body;

    if (!name || !folderPath) {
      return new Response(JSON.stringify({ error: 'name and folderPath required' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    // Check if folder exists
    if (!existsSync(folderPath)) {
      return new Response(JSON.stringify({ error: 'Folder does not exist' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    const projectId = generateProjectId();
    const projectDir = await createProjectDir(projectId);

    // Scan the folder for markdown files
    const result: { concepts: any[]; edges: any[] } = { concepts: [], edges: [] };
    await scanDirectoryForIndex(folderPath, result.concepts);

    // Build edges from concepts
    const ids = new Map(result.concepts.map(c => [c.id, c]));
    const edges: any[] = [];
    const edgeSet = new Set<string>();

    for (const c of result.concepts) {
      for (const t of c.leads_to || []) {
        if (ids.has(t)) {
          const eid = `${c.id}-leads-${t}`;
          if (!edgeSet.has(eid)) { edgeSet.add(eid); edges.push({ id: eid, source: c.id, target: t, type: 'leads_to' }); }
        }
      }
      for (const t of c.depends_on || []) {
        if (ids.has(t)) {
          const eid = `${c.id}-depends-${t}`;
          if (!edgeSet.has(eid)) { edgeSet.add(eid); edges.push({ id: eid, source: c.id, target: t, type: 'depends_on' }); }
        }
      }
      for (const t of c.related || []) {
        if (ids.has(t)) {
          const eid = `${c.id}-related-${t}`;
          if (!edgeSet.has(eid)) { edgeSet.add(eid); edges.push({ id: eid, source: c.id, target: t, type: 'related' }); }
        }
      }
    }
    result.edges = edges;

    // Save project config
    const now = new Date().toISOString();
    const project: Project = {
      id: projectId,
      name,
      folderPath,
      createdAt: now,
      lastOpenedAt: now,
    };
    await writeFile(join(projectDir, 'config.json'), JSON.stringify(project, null, 2));
    await writeFile(join(projectDir, 'concepts.json'), JSON.stringify(result.concepts, null, 2));
    await writeFile(join(projectDir, 'edges.json'), JSON.stringify(result.edges, null, 2));

    // Update project list
    const projects = await loadProjectList();
    projects.push(project);
    await saveProjectList(projects);

    return new Response(JSON.stringify({ project, concepts: result.concepts, edges: result.edges }), {
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (error) {
    return new Response(JSON.stringify({ error: String(error) }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}

// GET /api/projects/:id — 获取项目图谱数据
if (url.pathname.match(/^\/api\/projects\/([^/]+)$/) && req.method === 'GET') {
  const projectId = url.pathname.split('/')[3];
  const projectDir = join(PROJECTS_DIR, projectId);

  if (!existsSync(projectDir)) {
    return new Response(JSON.stringify({ error: 'Project not found' }), {
      status: 404,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  try {
    const configContent = await readFile(join(projectDir, 'config.json'), 'utf-8');
    const project = JSON.parse(configContent);

    let concepts: any[] = [];
    let edges: any[] = [];

    if (existsSync(join(projectDir, 'concepts.json'))) {
      concepts = JSON.parse(await readFile(join(projectDir, 'concepts.json'), 'utf-8'));
    }
    if (existsSync(join(projectDir, 'edges.json'))) {
      edges = JSON.parse(await readFile(join(projectDir, 'edges.json'), 'utf-8'));
    }

    // Update lastOpenedAt
    project.lastOpenedAt = new Date().toISOString();
    await writeFile(join(projectDir, 'config.json'), JSON.stringify(project, null, 2));

    // Update in project list
    const projects = await loadProjectList();
    const idx = projects.findIndex(p => p.id === projectId);
    if (idx >= 0) {
      projects[idx].lastOpenedAt = project.lastOpenedAt;
      await saveProjectList(projects);
    }

    return new Response(JSON.stringify({ project, concepts, edges }), {
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (error) {
    return new Response(JSON.stringify({ error: String(error) }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}

// DELETE /api/projects/:id — 删除项目
if (url.pathname.match(/^\/api\/projects\/([^/]+)$/) && req.method === 'DELETE') {
  const projectId = url.pathname.split('/')[3];
  const projectDir = join(PROJECTS_DIR, projectId);

  if (!existsSync(projectDir)) {
    return new Response(JSON.stringify({ error: 'Project not found' }), {
      status: 404,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  try {
    // Remove project directory
    await Bun.spawn(['rm', '-rf', projectDir]);

    // Update project list
    const projects = await loadProjectList();
    const filtered = projects.filter(p => p.id !== projectId);
    await saveProjectList(filtered);

    return new Response(JSON.stringify({ success: true }), {
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (error) {
    return new Response(JSON.stringify({ error: String(error) }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}
```

- [ ] **Step 4: Add os import at top of server.ts**

In the imports at the top of server.ts, add:
```typescript
import { homedir } from 'os';
```

And update the MAGIC_MEMORY_DIR line:
```typescript
const MAGIC_MEMORY_DIR = join(homedir(), '.magic-memory');
```

- [ ] **Step 5: Test API manually**

```bash
# Start server
bun run server.ts &
sleep 2

# Test project list (empty)
curl http://localhost:3001/api/projects

# Test creating a project (replace with real path)
curl -X POST http://localhost:3001/api/projects \
  -H "Content-Type: application/json" \
  -d '{"name":"test","folderPath":"/Users/meetai/docs"}'
```

- [ ] **Step 6: Commit**
```bash
git add server.ts
git commit -m "feat: add project management API endpoints"
```

---

## Task 3: Frontend Project Store

**Files:**
- Create: `src/store/projectStore.ts`
- Test: Manual via browser

- [ ] **Step 1: Create project store**

```typescript
// src/store/projectStore.ts
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

      // Auto-select first project if none selected
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
      // Auto-switch to new project
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

      // Update current project in list
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
```

- [ ] **Step 2: Commit**
```bash
git add src/store/projectStore.ts
git commit -m "feat: add project store for managing projects"
```

---

## Task 4: Project List UI Component

**Files:**
- Create: `src/components/ProjectList.tsx`
- Test: Manual via browser

- [ ] **Step 1: Create ProjectList component**

```typescript
// src/components/ProjectList.tsx
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
    return <div className="p-2 text-sm text-gray-500">加载中...</div>;
  }

  if (error) {
    return <div className="p-2 text-sm text-red-500">{error}</div>;
  }

  return (
    <div className="flex flex-col">
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
                {currentProjectId === project.id && '✓ '}
                {project.name}
              </span>
            </div>
            <button
              onClick={(e) => handleDelete(project.id, e)}
              className={`opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-gray-200 transition-opacity ${
                confirmDelete === project.id ? 'opacity-100 bg-red-100 text-red-600' : ''
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
```

- [ ] **Step 2: Commit**
```bash
git add src/components/ProjectList.tsx
git commit -m "feat: add ProjectList UI component"
```

---

## Task 5: Integrate ProjectList into Sidebar

**Files:**
- Modify: `src/components/Sidebar.tsx`
- Test: Manual via browser

- [ ] **Step 1: Read existing Sidebar.tsx to understand structure**

```bash
# Read first 50 lines to understand the structure
head -50 src/components/Sidebar.tsx
```

- [ ] **Step 2: Modify Sidebar.tsx**

Add import at top:
```typescript
import { ProjectList } from './ProjectList';
```

Find where to add the project panel. The Sidebar currently has document list. We'll add a two-column layout:

Replace the existing return statement with a split layout:

```typescript
// Inside Sidebar component, replace the return JSX with:
const handleAddProject = async () => {
  // Use File System Access API to select folder
  if ('showDirectoryPicker' in window) {
    try {
      // @ts-ignore - showDirectoryPicker is not in TypeScript types
      const handle = await window.showDirectoryPicker();
      const name = handle.name;
      // Get the folder path - note: we can't get full path from handle
      // We'll use the name and let user rename later, or store a reference
      // For now, we'll prompt for the path
      const folderPath = prompt('请输入文件夹完整路径:', `/${name}`);
      if (folderPath) {
        await useProjectStore.getState().createProject(name, folderPath);
      }
    } catch (err) {
      console.error('Failed to select folder:', err);
    }
  } else {
    alert('您的浏览器不支持文件夹选择器，请手动输入路径');
    const folderPath = prompt('请输入文件夹完整路径:');
    if (folderPath) {
      const name = folderPath.split('/').pop() || '新项目';
      await useProjectStore.getState().createProject(name, folderPath);
    }
  }
};

return (
  <div className="h-full flex flex-col bg-gray-50 border-r border-gray-200">
    <div className="flex-1 flex min-h-0">
      {/* Left: Project List */}
      <div className="w-1/2 border-r border-gray-200 flex flex-col">
        <ProjectList onAddProject={handleAddProject} />
      </div>

      {/* Right: Document List */}
      <div className="w-1/2 flex flex-col">
        <div className="text-xs font-semibold text-gray-500 px-2 py-1 uppercase tracking-wider">
          文档列表
        </div>
        {/* Existing document list content */}
        <div className="flex-1 overflow-y-auto">
          {documents.map((doc) => (
            <div
              key={doc.id}
              onClick={() => onDocumentSelect(doc)}
              className={`px-3 py-2 cursor-pointer hover:bg-gray-100 ${
                selectedDoc?.id === doc.id ? 'bg-blue-50 text-blue-700' : 'text-gray-700'
              }`}
            >
              <div className="text-sm font-medium truncate">{doc.title}</div>
              <div className="text-xs text-gray-500">
                {doc.level && <span className="mr-2">Level {doc.level}</span>}
                {doc.category && <span className="mr-2">{doc.category}</span>}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>

    <div className="p-2 border-t border-gray-200">
      <button
        onClick={onImport}
        className="w-full py-2 px-3 text-sm text-gray-600 hover:bg-gray-100 rounded flex items-center justify-center gap-2"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
        </svg>
        导入文档
      </button>
    </div>
  </div>
);
```

Note: You'll need to remove the existing JSX that was there before and adjust imports accordingly.

- [ ] **Step 3: Add import for useProjectStore**

```typescript
import { useProjectStore } from '../store/projectStore';
```

- [ ] **Step 4: Test manually**

- Refresh the page
- In document view, the sidebar should now show two columns
- Click "添加项目文件夹" to test project creation

- [ ] **Step 5: Commit**
```bash
git add src/components/Sidebar.tsx
git commit -m "feat: integrate ProjectList into Sidebar with two-column layout"
```

---

## Task 6: Update KnowledgeGraphView to Support Projects

**Files:**
- Modify: `src/components/KnowledgeGraphView.tsx`
- Test: Manual via browser

- [ ] **Step 1: Modify to load graph based on current project**

In KnowledgeGraphView.tsx, update the useEffect that calls loadGraph:

```typescript
// Replace the existing loadGraph call with project-aware version
const currentProjectId = useProjectStore(s => s.currentProjectId);
const loadProjects = useProjectStore(s => s.loadProjects);

// Load projects on mount
useEffect(() => {
  loadProjects();
}, [loadProjects]);

// When currentProjectId changes, reload the graph
useEffect(() => {
  if (currentProjectId) {
    useKnowledgeGraphStore.getState().loadGraph();
  }
}, [currentProjectId]);
```

Add the import:
```typescript
import { useProjectStore } from '../store/projectStore';
```

- [ ] **Step 2: Update folder selection to create projects**

Modify the handleBrowseFolder function to create a project instead of just scanning:

```typescript
const handleBrowseFolder = async () => {
  if (!('showDirectoryPicker' in window)) {
    alert('Your browser does not support folder selection');
    return;
  }

  try {
    // @ts-ignore
    const handle = await window.showDirectoryPicker();
    const folderPath = prompt('请输入选中的文件夹完整路径:');
    if (!folderPath) return;

    const name = handle.name;
    await useProjectStore.getState().createProject(name, folderPath);
  } catch (err) {
    if (err.name !== 'AbortError') {
      console.error('Error selecting folder:', err);
    }
  }
};
```

- [ ] **Step 3: Test manually**

- Select a project from the sidebar
- The knowledge graph should load that project's data

- [ ] **Step 4: Commit**
```bash
git add src/components/KnowledgeGraphView.tsx
git commit -m "feat: support project-based graph loading in KnowledgeGraphView"
```

---

## Task 7: Final Integration and Testing

**Files:**
- N/A
- Test: Manual end-to-end

- [ ] **Step 1: Test complete flow**

1. Start the server: `bun run server.ts`
2. Open browser to http://localhost:3000
3. Click "文档" to enter document view
4. Verify sidebar shows two columns (projects + documents)
5. Click "添加项目文件夹"
6. Enter a folder path containing markdown files
7. Verify project appears in list and is selected
8. Click "知识图" to view the graph
9. Verify the graph loads with the project's concepts
10. Add another project and switch between them
11. Verify data is isolated per project

- [ ] **Step 2: Test deletion**

1. In project list, click delete button twice on a project
2. Verify project is removed
3. Verify the project directory is deleted from `~/.magic-memory/projects/`

- [ ] **Step 3: Commit**
```bash
git add .
git commit -m "feat: complete project-based knowledge graph implementation"
```

---

## Summary

This plan implements project-based knowledge graph management with:

1. **Server-side storage**: Each project stored in `~/.magic-memory/projects/{project-id}/`
2. **RESTful API**: Full CRUD for projects at `/api/projects`
3. **Frontend store**: `projectStore.ts` manages project state
4. **UI integration**: Two-column sidebar with project list and document list
5. **Graph isolation**: Switching projects loads the corresponding graph data

All tasks follow TDD with failing test → minimal implementation → pass → commit cycle.