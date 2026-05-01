export interface Document {
  id: string
  title: string
  path: string
  content: string
  level: number
  category: string
  tags: string[]
  lastModified: Date
  metadata?: {
    author?: string
    version?: string
    status?: 'draft' | 'review' | 'approved'
  }
}

export interface Annotation {
  id: string
  documentId: string
  type: 'comment' | 'question' | 'suggestion' | 'correction'
  content: string
  position: {
    start: number
    end: number
    line?: number
  }
  author: string
  createdAt: Date
  updatedAt: Date
  status: 'open' | 'resolved' | 'closed'
  replies?: AnnotationReply[]
}

export interface AnnotationReply {
  id: string
  content: string
  author: string
  createdAt: Date
}

export interface DocumentNode {
  id: string
  title: string
  path: string
  level: number
  children: DocumentNode[]
  isExpanded: boolean
}

export interface AnnotationStats {
  total: number
  byType: Record<string, number>
  byStatus: Record<string, number>
  recent: number
}

export interface ViewMode {
  mode: 'read' | 'annotate' | 'review'
  showLineNumbers: boolean
  showAnnotations: boolean
  showMetadata: boolean
}

// ========== 知识图类型 ==========

export interface Concept {
  id: string
  title: string
  alias?: string[]           // 别名: [Attention, QKV]
  level: number             // 难度级别 1-3
  category: string
  
  // 核心字段 - 支持可推导学习
  problem?: string         // 问题: "如何让模型知道重点关注哪些词?"
  gap_anticipate?: string // 预判认知gap: "Q/K/V是什么？为什么需要三个?"
  
  // 关联关系
  depends_on: string[]     // 前置概念ID列表
  leads_to: string[]        // 引出概念ID列表
  related: string[]       // 相关概念
  
  // 过程推导
  process?: {
    chain_id: string
    step_index: number
    role: string
  }
  elements?: ConceptElement[]

  // 层级
  hierarchy?: {
    parentId: string | null
    level: number
    order: number
  }
  content?: string       // 文档正文（Markdown）
  path: string           // 文件路径
  
  // 元数据
  tags: string[]
  lastModified: Date
  metadata?: {
    author?: string
    version?: string
    status?: 'draft' | 'review' | 'approved'
  }
}

export interface ConceptEdge {
  id: string
  source: string          // 概念ID
  target: string         // 概念ID
  type: 'depends_on' | 'leads_to' | 'related'
  label?: string
}

export interface KnowledgeGraph {
  concepts: Concept[]
  edges: ConceptEdge[]
}

// ========== 过程推导类型 ==========

export interface ConceptElement {
  name: string
  description: string
  type: 'core_field' | 'design_pattern' | 'key_insight' | 'boundary' | 'relation'
  order: number
}

export interface ProcessStep {
  id: string
  label: string
  description: string
  question: string
  hint: string
  leads_to_type: 'element' | 'concept'
  leads_to_id?: string
  is_core: boolean
}

export interface ProcessChain {
  id: string
  name: string
  steps: ProcessStep[]
}

export interface ProcessState {
  user_flow: string[]
  llm_flow: string[]
  gaps: string[]
  filled: boolean
  compared: boolean
}

// ========== 骨架填充类型 ==========

// Skeleton-related types removed: BaseQuestion, UserQuestion, CanvasHistoryItem

export interface LearningPath {
  id: string
  title: string
  description?: string
  concept_ids: string[]   // 有序的概念ID列表
  estimated_minutes?: number
}

export interface ReviewRecord {
  concept_id: string
  last_reviewed: Date
  next_review: Date
  ease_factor: number
  interval: number
  review_count: number
  status: 'new' | 'learning' | 'review' | 'mastered'
  process_state?: ProcessState
}

export interface UserAnnotation {
  id: string
  conceptId: string
  type: 'question' | 'note' | 'gap_report' | 'correction'
  content: string
  createdAt: Date
  author: string
  status: 'open' | 'resolved'
}

export interface SuggestionItem {
  title: string
  problem: string
  relationType: 'leads_to' | 'depends_on' | 'related'
  checked: boolean
}
// Re-export Project-related types for downstream usage
export type { Project, ProjectConfig, ProjectGraphData } from './project';
