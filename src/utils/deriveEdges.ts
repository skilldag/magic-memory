import type { Concept, ConceptEdge } from '../types';

/**
 * 从 .md 文件内容中自动推导概念之间的关联边。
 *
 * 策略：交叉引用 — 文件内容中提到其他概念的标题 → related 边
 */
export function deriveEdges(
  concepts: Concept[],
  files: { path: string; content: string }[],
): ConceptEdge[] {
  const seen = new Set<string>();
  const edges: ConceptEdge[] = [];

  function addEdge(source: string, target: string) {
    if (source === target) return;
    const pair = [source, target].sort().join('::');
    if (seen.has(pair)) return;
    seen.add(pair);
    edges.push({ id: `e_${pair}`, source, target, type: 'related' });
  }

  // 构建标题 → 概念ID 查找表
  const titleToId = new Map<string, string>();
  for (const c of concepts) {
    const plain = c.title.replace(/^\d+\s*/, '').toLowerCase();
    titleToId.set(c.title.toLowerCase(), c.id);
    if (plain !== c.title.toLowerCase()) titleToId.set(plain, c.id);
  }

  // 交叉引用：扫描每个文件的正文，找其他概念标题
  const contentById = new Map<string, string>(files.map(f => {
    const id = f.path.replace('.md', '').replace(/\//g, '-');
    return [id, f.content];
  }));

  for (const c of concepts) {
    const content = contentById.get(c.id) || '';
    if (!content) continue;
    for (const [title, otherId] of titleToId) {
      if (otherId === c.id) continue;
      if (title.length >= 3 && content.toLowerCase().includes(title)) {
        addEdge(c.id, otherId);
      }
    }
  }

  return edges;
}
