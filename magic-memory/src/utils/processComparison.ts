import type { Concept, ProcessStep, ProcessChain } from '../types'

export type DiffStatus = 'match' | 'missing' | 'extra'

export interface DiffItem {
  stepId: string
  label: string
  description: string
  status: DiffStatus
  leads_to_type?: 'element' | 'concept'
  leads_to_id?: string
}

export function generateReferenceFlow(
  conceptId: string,
  concepts: Concept[],
  chains: ProcessChain[]
): ProcessStep[] {
  const concept = concepts.find(c => c.id === conceptId)
  if (!concept?.process) return []
  const chain = chains.find(ch => ch.id === concept.process.chain_id)
  if (!chain) return []
  return chain.steps
}

export function getUserDependentChain(
  conceptId: string,
  concepts: Concept[]
): ProcessChain | null {
  const concept = concepts.find(c => c.id === conceptId)
  if (!concept?.process) return null

  const chainId = concept.process.chain_id
  const chainConcepts = concepts
    .filter(c => c.process?.chain_id === chainId)
    .sort((a, b) => (a.process?.step_index ?? 0) - (b.process?.step_index ?? 0))

  const steps: ProcessStep[] = chainConcepts.map(c => ({
    id: c.id,
    label: c.title,
    description: c.process?.role ?? '',
    question: c.problem ?? '',
    hint: c.gap_anticipate ?? '',
    leads_to_type: 'concept' as const,
    leads_to_id: c.id,
    is_core: true,
  }))

  return { id: chainId, name: chainId, steps }
}

export function generateGenericChain(
  conceptId: string,
  concepts: Concept[]
): ProcessChain {
  const concept = concepts.find(c => c.id === conceptId)
  if (!concept) {
    return { id: 'generic', name: '概念推导', steps: [] }
  }

  const steps: ProcessStep[] = []

  if (concept.depends_on.length > 0) {
    const deps = concept.depends_on
      .map(id => concepts.find(c => c.id === id))
      .filter(Boolean) as Concept[]
    deps.forEach(d => {
      steps.push({
        id: `dep_${d.id}`,
        label: d.title,
        description: d.problem || '前置基础概念',
        question: '',
        hint: '',
        leads_to_type: 'concept',
        leads_to_id: d.id,
        is_core: true,
      })
    })
  }

  steps.push({
    id: `self_${concept.id}`,
    label: concept.title,
    description: concept.problem || '当前概念',
    question: concept.problem || '这个概念要解决什么问题？',
    hint: concept.gap_anticipate || '试着回忆它为什么存在',
    leads_to_type: 'concept',
    leads_to_id: concept.id,
    is_core: true,
  })

  if (concept.leads_to.length > 0) {
    const nexts = concept.leads_to
      .map(id => concepts.find(c => c.id === id))
      .filter(Boolean) as Concept[]
    nexts.forEach(n => {
      steps.push({
        id: `next_${n.id}`,
        label: n.title,
        description: n.problem || '后继推论概念',
        question: '',
        hint: '',
        leads_to_type: 'concept',
        leads_to_id: n.id,
        is_core: false,
      })
    })
  }

  return { id: 'generic', name: `理解 ${concept.title}`, steps }
}

export function diffFlows(
  userStepIds: string[],
  referenceSteps: ProcessStep[]
): DiffItem[] {
  const refIds = new Set(referenceSteps.map(s => s.id))
  const userIds = new Set(userStepIds)
  const userOrder = new Map(userStepIds.map((id, i) => [id, i]))

  const result: DiffItem[] = []

  for (const step of referenceSteps) {
    const inUser = userIds.has(step.id)
    result.push({
      stepId: step.id,
      label: step.label,
      description: step.description,
      status: inUser ? 'match' : 'missing',
      leads_to_type: inUser ? undefined : step.leads_to_type,
      leads_to_id: inUser ? undefined : step.leads_to_id,
    })
  }

  for (const id of userStepIds) {
    if (!refIds.has(id)) {
      result.push({
        stepId: id,
        label: id,
        description: '',
        status: 'extra',
      })
    }
  }

  result.sort((a, b) => {
    const oa = userOrder.get(a.stepId) ?? 999
    const ob = userOrder.get(b.stepId) ?? 999
    return oa - ob
  })

  return result
}

export function getGapConceptIds(diffs: DiffItem[]): string[] {
  return diffs
    .filter(d => d.status === 'missing' && d.leads_to_id)
    .map(d => d.leads_to_id!)
}

// ========== 骨架填充 ==========

export interface SkeletonNodeDef {
  id: string
  type: 'gap' | 'known' | 'current'
  label: string
  question: string
  correctConceptId?: string
  hint?: string
}

export function generateSkeletonNodes(
  concept: Concept,
  chain: ProcessChain | null,
  allConcepts: Concept[]
): SkeletonNodeDef[] {
  const nodes: SkeletonNodeDef[] = []
  const addedIds = new Set<string>()

  if (chain) {
    chain.steps.forEach((step) => {
      if (step.leads_to_id === concept.id) {
        nodes.push({
          id: `current_${concept.id}`,
          type: 'current',
          label: concept.title,
          question: step.question || '当前概念',
          correctConceptId: concept.id,
        })
        addedIds.add(`current_${concept.id}`)
      } else {
        const targetLabel = step.leads_to_id
          ? allConcepts.find(c => c.id === step.leads_to_id)?.title ?? step.label
          : step.label
        nodes.push({
          id: `gap_${step.id}`,
          type: 'gap',
          label: targetLabel,
          question: step.question || '这里应该填什么概念？',
          correctConceptId: step.leads_to_id,
          hint: step.hint || undefined,
        })
        if (step.leads_to_id) addedIds.add(`gap_${step.id}`)
      }
    })
  } else {
    const generic = generateGenericChain(concept.id, allConcepts)
    generic.steps.forEach((step) => {
      if (step.leads_to_id === concept.id) {
        nodes.push({
          id: `current_${concept.id}`,
          type: 'current',
          label: concept.title,
          question: step.question || '当前概念',
          correctConceptId: concept.id,
        })
      } else {
        nodes.push({
          id: `gap_${step.id}`,
          type: 'gap',
          label: step.label,
          question: step.question || '这里应该填什么概念？',
          correctConceptId: step.leads_to_id,
        })
      }
    })
  }

  return nodes
}
