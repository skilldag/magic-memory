export const EDGE_COLORS: Record<string, string> = {
  depends_on: '#ef4444',
  leads_to: '#10b981',
  related: '#6b7280',
}

export const MASTERY_COLORS: Record<string, string> = {
  unaligned: '#d1d5db',  // gray - never aligned
  weak:      '#ef4444',   // red - score < 40
  partial:   '#f59e0b',   // amber - 40 ≤ score < 70
  good:      '#10b981',   // green - 70 ≤ score < 90
  mastered:  '#059669',   // deep green - score ≥ 90
}

export function getMasteryColor(score: number | undefined): string {
  if (score === undefined) return MASTERY_COLORS.unaligned
  if (score >= 90) return MASTERY_COLORS.mastered
  if (score >= 70) return MASTERY_COLORS.good
  if (score >= 40) return MASTERY_COLORS.partial
  return MASTERY_COLORS.weak
}
