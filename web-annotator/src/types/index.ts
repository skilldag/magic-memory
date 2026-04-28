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
  
  // 内容
  content: string          // Markdown内容
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

export interface LearningPath {
  id: string
  title: string
  description?: string
  concept_ids: string[]   // 有序的概念ID列表
  estimated_minutes?: number
}

export interface ReviewRecord {
  concept_id: string
  last_reviewed: Date      // 上次复习时间
  next_review: Date       // 下次复习时间
  ease_factor: number      // 简易度因子 (初始2.5)
  interval: number        // 间隔天数
  review_count: number  // 复习次数
  status: 'new' | 'learning' | 'review' | 'mastered'
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
