export interface Project {
  id: string;
  name: string;
  folderPath?: string;
  handleStoreId: string | null;
  createdAt: string;
  lastOpenedAt: string;
}

export interface ProjectConfig {
  id: string;
  name: string;
  folderPath?: string;
  handleStoreId: string | null;
  createdAt: string;
  lastOpenedAt: string;
}

export interface ProjectGraphData {
  concepts: import('./index').Concept[];
  edges: import('./index').ConceptEdge[];
}
