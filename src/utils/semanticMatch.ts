import { pipeline, env } from '@xenova/transformers'
import type { FeatureExtractionPipeline } from '@xenova/transformers'

env.backends.onnx.wasm.wasmPaths = '/ort-wasm/'

let pipe: FeatureExtractionPipeline | null = null
let loading: Promise<FeatureExtractionPipeline> | null = null

async function getPipeline(): Promise<FeatureExtractionPipeline> {
  if (pipe) return pipe
  if (loading) return loading
  loading = pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
    quantized: true,
  }) as Promise<FeatureExtractionPipeline>
  pipe = await loading
  return pipe
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb)
  return denom === 0 ? 0 : dot / denom
}

export interface SemanticMatchResult {
  label: string
  score: number
  matched: boolean
}

export async function matchKeyConcepts(
  userText: string,
  keyConcepts: string[],
  threshold: number = 0.45
): Promise<SemanticMatchResult[]> {
  if (keyConcepts.length === 0) return []

  const model = await getPipeline()

  const texts = [userText, ...keyConcepts]
  const raw = await model(texts, { pooling: 'mean', normalize: true })
  const tensors = Array.isArray(raw) ? raw : [raw]
  const userEmb = Array.from(tensors[0].data as Float32Array)
  const results: SemanticMatchResult[] = []

  for (let i = 0; i < keyConcepts.length; i++) {
    const kcEmb = Array.from(tensors[i + 1].data as Float32Array)
    const score = cosineSimilarity(userEmb, kcEmb)
    results.push({
      label: keyConcepts[i],
      score: Math.round(score * 100),
      matched: score >= threshold,
    })
  }

  return results
}

export async function isModelReady(): Promise<boolean> {
  try {
    await getPipeline()
    return true
  } catch {
    return false
  }
}
