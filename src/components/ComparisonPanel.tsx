import type { DiffItem } from '../utils/processComparison'

interface ComparisonPanelProps {
  diffs: DiffItem[]
  userStepCount: number
  referenceStepCount: number
  onNavigateGap: (conceptId: string) => void
}

export function ComparisonPanel({
  diffs,
  userStepCount,
  referenceStepCount,
  onNavigateGap,
}: ComparisonPanelProps) {
  const matchCount = diffs.filter(d => d.status === 'match').length
  const missingCount = diffs.filter(d => d.status === 'missing').length
  const extraCount = diffs.filter(d => d.status === 'extra').length
  const score = referenceStepCount > 0
    ? Math.round((matchCount / referenceStepCount) * 100)
    : 0

  return (
    <div className="px-5 py-4 space-y-4">
      <div>
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
          对照结果
        </h3>

        <div className="grid grid-cols-3 gap-2 mb-4">
          <div className="p-2.5 rounded-lg border border-emerald-200 bg-emerald-50 text-center">
            <div className="text-lg font-bold text-emerald-700">{matchCount}</div>
            <div className="text-xs text-emerald-600">匹配</div>
          </div>
          <div className="p-2.5 rounded-lg border border-amber-200 bg-amber-50 text-center">
            <div className="text-lg font-bold text-amber-700">{missingCount}</div>
            <div className="text-xs text-amber-600">遗漏</div>
          </div>
          <div className="p-2.5 rounded-lg border border-gray-200 bg-gray-50 text-center">
            <div className="text-lg font-bold text-gray-500">{extraCount}</div>
            <div className="text-xs text-gray-500">多余</div>
          </div>
        </div>

        <div className="mb-4">
          <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
            <span>推导覆盖率</span>
            <span>{score}%</span>
          </div>
          <div className="w-full h-2 bg-gray-100 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${
                score >= 80 ? 'bg-emerald-500' : score >= 50 ? 'bg-amber-500' : 'bg-red-400'
              }`}
              style={{ width: `${score}%` }}
            />
          </div>
        </div>
      </div>

      <div className="space-y-1.5">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
          逐项对比
        </h3>

        <div className="grid grid-cols-[1fr_auto_1fr] gap-2 text-xs text-gray-500 mb-1 px-1">
          <span>你的推导</span>
          <span />
          <span>参考流程</span>
        </div>

        {diffs.map((d, idx) => (
          <div key={`${d.stepId}-${idx}`} className={`rounded-lg border p-2.5 ${
            d.status === 'match'
              ? 'border-emerald-200 bg-emerald-50/50'
              : d.status === 'missing'
              ? 'border-amber-200 bg-amber-50'
              : 'border-gray-200 bg-gray-50'
          }`}>
            <div className="flex items-center gap-2">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5">
                  {d.status === 'match' && (
                    <span className="text-emerald-600 font-bold text-sm">✓</span>
                  )}
                  {d.status === 'missing' && (
                    <span className="text-amber-600 text-sm">⚡</span>
                  )}
                  {d.status === 'extra' && (
                    <span className="text-gray-400 text-sm">⟳</span>
                  )}
                  <span className={`text-sm font-medium ${
                    d.status === 'match' ? 'text-gray-800' :
                    d.status === 'missing' ? 'text-amber-900' :
                    'text-gray-500'
                  }`}>
                    {d.label}
                  </span>
                </div>
                {d.description && (
                  <p className="text-xs text-gray-500 mt-0.5">{d.description}</p>
                )}
              </div>
              {d.status === 'missing' && d.leads_to_id && (
                <button
                  onClick={() => onNavigateGap(d.leads_to_id!)}
                  className="shrink-0 px-2 py-1 text-xs font-medium text-amber-700 bg-amber-100 rounded-md hover:bg-amber-200 transition-colors"
                >
                  探索
                </button>
              )}
            </div>
          </div>
        ))}

        {diffs.length === 0 && (
          <p className="text-xs text-gray-400 text-center py-4">
            暂无对照数据，请先完成过程梳理
          </p>
        )}
      </div>
    </div>
  )
}
