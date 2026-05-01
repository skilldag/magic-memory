#!/usr/bin/env bun
/**
 * 概念聚类管道 — TypeScript 全算法版
 *
 * 流程:
 *   1. 扫描目录，从文件名提取概念 ID
 *   2. 读取文档内容检测交叉引用 → 构建边
 *   3. 用目录结构 + 编号邻近性补充边
 *   4. Louvain 社区检测
 *   5. 输出 JSON 供前端渲染
 *
 * 用法:
 *   bun scripts/cluster.ts --dir ../docs
 *   bun scripts/cluster.ts --dir ../docs --files "00-vllm-config.md" "01-device.md"
 *   bun scripts/cluster.ts --dir ../docs --resolution 0.5
 */

import { readdirSync, readFileSync, existsSync } from "fs";
import { join, relative, parse } from "path";

// ─── Types ────────────────────────────────────────────────────────

export { };

export interface Concept {
  id: string;
  title: string;
  number: number | null;
  category: string;
  path: string;
}

export interface Edge {
  source: string;
  target: string;
  type: EdgeType;
  weight: number;
}

export type EdgeType = "references" | "co_directory" | "number_proximity" | "frontmatter";

export interface Community {
  name: string;
  size: number;
  numRange: string;
  cohesion: number;
  members: Concept[];
}

export interface ClusterResult {
  totalConcepts: number;
  totalEdges: number;
  totalCommunities: number;
  resolution: number;
  concepts: Concept[];
  edges: { source: string; target: string; type: EdgeType; weight: number }[];
  communities: Community[];
}

// ─── 概念名提取（纯算法） ──────────────────────────────────────────

/** 跳过非概念的文件名 */
const SKIP_FILES = new Set(["readme.md", "index.md", "readme"]);
const SKIP_PREFIXES = ["设计文档", "方法论", "框架", "推导", "自测", "模板"];

function filenameToConceptId(filename: string): string | null {
  const name = filename.replace(/\.md$/i, "").trim();
  if (!name || SKIP_FILES.has(name.toLowerCase())) return null;

  // 去掉前导数字 + 分隔符
  const stripped = name.replace(/^\d+[-_\s]+/, "");
  if (!stripped) return null;

  // 跳过非概念文档
  for (const prefix of SKIP_PREFIXES) {
    if (stripped.toLowerCase().startsWith(prefix.toLowerCase())) return null;
  }

  return stripped;
}

function conceptIdToTitle(cid: string): string {
  if (!/^[\x00-\x7F]+$/.test(cid)) return cid; // non-ASCII
  return cid
    .split(/[-_]/)
    .map((s) => s.charAt(0).toUpperCase() + s.slice(1))
    .join(" ");
}

function extractConceptNumber(filename: string): number | null {
  const m = filename.match(/^(\d+)/);
  return m ? parseInt(m[1], 10) : null;
}

function extractCategoryFromPath(
  filePath: string,
  basePath: string
): string {
  const rel = relative(basePath, filePath);
  const parts = rel.split("/");
  if (parts.length >= 2) {
    // 取第一个子目录
    return parts[0].replace(/^\d+-/, "");
  }
  return "其他";
}

// ─── 交叉引用检测 ──────────────────────────────────────────────────

function extractCrossReferences(
  content: string,
  knownConcepts: Map<string, Concept>
): string[] {
  const refs = new Set<string>();
  const knownIds = new Set(knownConcepts.keys());

  // 1. Wiki 链接 [[概念]]
  for (const m of content.matchAll(/\[\[([^\]]+)\]\]/g)) {
    const ref = m[1].trim().toLowerCase().replace(/\s+/g, "-");
    if (knownIds.has(ref)) refs.add(ref);
  }

  // 2. Markdown 链接 [text](path/to/some-concept.md)
  for (const m of content.matchAll(/\]\(([^)]+\.md)\)/g)) {
    const target = m[1];
    const fname = target.split("/").pop() || target;
    const cid = filenameToConceptId(fname);
    if (cid && knownIds.has(cid)) refs.add(cid);
  }

  return [...refs];
}

// ─── 文档扫描 ──────────────────────────────────────────────────────

function scanDocuments(docsDir: string): Concept[] {
  const conceptMap = new Map<string, Concept>();
  const fileMap = new Map<string, string>(); // cid → path

  function walkDir(dir: string) {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const fullPath = join(dir, entry.name);
      if (entry.isDirectory() && !entry.name.startsWith(".")) {
        walkDir(fullPath);
      } else if (entry.isFile() && entry.name.endsWith(".md")) {
        const cid = filenameToConceptId(entry.name);
        if (!cid) continue;

        // 去重：保留路径更短的
        const existing = fileMap.get(cid);
        if (existing) {
          if (fullPath.split("/").length < existing.split("/").length) {
            fileMap.set(cid, fullPath);
          }
          continue;
        }
        fileMap.set(cid, fullPath);
      }
    }
  }

  walkDir(docsDir);

  for (const [cid, filePath] of fileMap) {
    const num = extractConceptNumber(parse(filePath).name);
    const category = extractCategoryFromPath(filePath, docsDir);

    conceptMap.set(cid, {
      id: cid,
      title: conceptIdToTitle(cid),
      number: num,
      category,
      path: filePath,
    });
  }

  return [...conceptMap.values()];
}

// ─── 图构建 ────────────────────────────────────────────────────────

function buildGraph(
  concepts: Concept[],
  docsDir: string,
  contentMap: Map<string, string>
): { concepts: Concept[]; edges: Edge[] } {
  const knownIds = new Map(concepts.map((c) => [c.id, c]));
  const seenPairs = new Set<string>();
  const edges: Edge[] = [];

  function addEdge(src: string, tgt: string, type: EdgeType, weight: number) {
    const pair = [src, tgt].sort().join("::");
    if (seenPairs.has(pair)) return;
    seenPairs.add(pair);
    edges.push({ source: src, target: tgt, type, weight });
  }

  // 1. 交叉引用边（强）
  for (const c of concepts) {
    const content = contentMap.get(c.id) || "";
    const refs = extractCrossReferences(content, knownIds);
    for (const ref of refs) {
      if (ref !== c.id) addEdge(c.id, ref, "references", 0.9);
    }
  }

  // 2. 目录结构边（弱）
  const dirGroups = new Map<string, Concept[]>();
  for (const c of concepts) {
    const dir = relative(docsDir, c.path).split("/").slice(0, -1).join("/") || ".";
    const group = dirGroups.get(dir) || [];
    group.push(c);
    dirGroups.set(dir, group);
  }

  for (const [, members] of dirGroups) {
    if (members.length < 2) continue;
    for (let i = 0; i < members.length; i++) {
      for (let j = i + 1; j < members.length; j++) {
        addEdge(members[i].id, members[j].id, "co_directory", 0.2);
      }
    }
  }

  // 父-子目录边（中等）
  for (const [dir, members] of dirGroups) {
    if (dir === ".") continue;
    const parent = dir.split("/").slice(0, -1).join("/") || ".";
    const parentGroup = dirGroups.get(parent);
    if (!parentGroup) continue;
    for (const p of parentGroup) {
      for (const m of members) {
        addEdge(p.id, m.id, "co_directory", 0.4);
      }
    }
  }

  // 3. 编号邻近边
  const numbered = concepts
    .filter((c) => c.number !== null)
    .sort((a, b) => (a.number as number) - (b.number as number));

  for (let i = 0; i < numbered.length - 1; i++) {
    addEdge(numbered[i].id, numbered[i + 1].id, "number_proximity", 0.3);
  }

  return { concepts, edges };
}

// ─── Louvain 社区检测 ──────────────────────────────────────────────
// 实现标准的 Louvain 算法（Blondel et al., 2008）
// 不用 igraph/Leiden，纯 TypeScript 实现

function louvainCommunityDetection(
  concepts: Concept[],
  edges: Edge[]
): { communityOf: Map<string, number>; numCommunities: number } {
  const n = concepts.length;
  const nodeIds = concepts.map((c) => c.id);

  // 邻接表: nodeIdx → Map<neighborIdx, weight>
  const adj: Map<number, number>[] = nodeIds.map(() => new Map());
  let totalWeight = 0;

  for (const e of edges) {
    const si = nodeIds.indexOf(e.source);
    const ti = nodeIds.indexOf(e.target);
    if (si === -1 || ti === -1 || si === ti) continue;

    const existing = adj[si].get(ti) || 0;
    adj[si].set(ti, existing + e.weight);
    const existing2 = adj[ti].get(si) || 0;
    adj[ti].set(si, existing2 + e.weight);
    totalWeight += e.weight;
  }

  // 跳过无边的图
  if (totalWeight === 0) {
    const communityOf = new Map<string, number>();
    concepts.forEach((c, i) => communityOf.set(c.id, i));
    return { communityOf, numCommunities: n };
  }

  const m2 = 2 * totalWeight; // 2m

  // 节点度数 (sum of incident edge weights)
  const degrees: number[] = nodeIds.map((_, i) => {
    let d = 0;
    for (const w of adj[i].values()) d += w;
    return d;
  });

  // ── Phase 1: 局部优化 ──
  // 初始化：每个节点独立社区
  const communityIdx: number[] = nodeIds.map((_, i) => i);

  // community → sum of incident edge weights for its members
  const commTotals: number[] = degrees.slice();

  let improved = true;
  let pass = 0;
  const maxPasses = 20;

  while (improved && pass < maxPasses) {
    improved = false;
    pass++;

    // 随机化节点顺序有助于收敛
    const order = nodeIds.map((_, i) => i).sort(() => Math.random() - 0.5);

    for (const i of order) {
      const currentComm = communityIdx[i];
      const neighbors = adj[i];

      // 计算从 i 到各邻居社区的 k_i_in
      const commKI = new Map<number, number>();
      for (const [neighbor, weight] of neighbors) {
        const nc = communityIdx[neighbor];
        commKI.set(nc, (commKI.get(nc) || 0) + weight);
      }

      // 如果 i 是当前社区的唯一节点，先不移除
      let bestComm = currentComm;
      let bestGain = 0;
      const ki = degrees[i];

      // 移除 i 对当前社区的贡献
      const k_i_in_current = commKI.get(currentComm) || 0;
      const sigma_tot_current = commTotals[currentComm];
      // ΔQ_remove = -k_i_in/m + (sigma_tot - k_i) * k_i / (2m²)
      const removeGain =
        -k_i_in_current / totalWeight +
        ((sigma_tot_current - ki) * ki) / (m2 * totalWeight);

      for (const [nc, k_i_in] of commKI) {
        if (nc === currentComm) continue;
        const sigma_tot = commTotals[nc];
        // ΔQ_add = k_i_in/m - sigma_tot * k_i / (2m²)
        const addGain =
          k_i_in / totalWeight - (sigma_tot * ki) / (m2 * totalWeight);
        const totalGain = addGain + removeGain;

        if (totalGain > bestGain) {
          bestGain = totalGain;
          bestComm = nc;
        }
      }

      if (bestComm !== currentComm) {
        // 移动节点
        communityIdx[i] = bestComm;
        commTotals[currentComm] -= ki;
        commTotals[bestComm] += ki;
        improved = true;
      }
    }
  }

  // 压缩社区编号
  const uniqueComms = [...new Set(communityIdx)];
  const commMap = new Map<number, number>();
  uniqueComms.forEach((old, newId) => commMap.set(old, newId));

  const communityOf = new Map<string, number>();
  concepts.forEach((c, i) => {
    communityOf.set(c.id, commMap.get(communityIdx[i])!);
  });

  // ── Phase 2: 聚合（简单版） ──
  // 对于小型图 (< 200 节点)，一次 Louvain pass 就够
  // 如果需要更精确的结果，可以对聚合后的图递归

  return {
    communityOf,
    numCommunities: uniqueComms.length,
  };
}

// ─── 主管道 ────────────────────────────────────────────────────────

function formatNumRange(members: Concept[]): string {
  const nums = members
    .map((m) => m.number)
    .filter((n): n is number => n !== null)
    .sort((a, b) => a - b);
  if (nums.length === 0) return "";
  if (nums.length === 1) return `${nums[0]}`;
  return `${nums[0]}-${nums[nums.length - 1]}`;
}

export function clusterPipeline(
  docsDir: string,
  fileFilter: string[] | null,
  resolution: number
): ClusterResult {
  // 1. 扫描文档
  const concepts = scanDocuments(docsDir);

  // 2. 过滤
  let filtered = concepts;
  if (fileFilter && fileFilter.length > 0) {
    const targetIds = new Set<string>();
    for (const fname of fileFilter) {
      const cid = filenameToConceptId(fname);
      if (cid) targetIds.add(cid);
    }
    filtered = concepts.filter((c) => targetIds.has(c.id));
  }

  if (filtered.length === 0) {
    return {
      totalConcepts: 0,
      totalEdges: 0,
      totalCommunities: 0,
      resolution,
      concepts: [],
      edges: [],
      communities: [],
    };
  }

  // 3. 读取内容（用于交叉引用检测）
  const contentMap = new Map<string, string>();
  for (const c of filtered) {
    try {
      contentMap.set(c.id, readFileSync(c.path, "utf-8"));
    } catch {
      contentMap.set(c.id, "");
    }
  }

  // 4. 构建图
  const { concepts: finalConcepts, edges } = buildGraph(
    filtered,
    docsDir,
    contentMap
  );

  // 5. 社区检测
  const { communityOf, numCommunities } =
    louvainCommunityDetection(finalConcepts, edges);

  // 6. 组织社区
  const commGroups = new Map<number, Concept[]>();
  for (const c of finalConcepts) {
    const commId = communityOf.get(c.id) ?? 0;
    const group = commGroups.get(commId) || [];
    group.push(c);
    commGroups.set(commId, group);
  }

  const communities: Community[] = [];

  // 计算 cohesion
  const internalCount = new Map<number, number>();
  const externalCount = new Map<number, number>();
  for (const commId of commGroups.keys()) {
    internalCount.set(commId, 0);
    externalCount.set(commId, 0);
  }

  for (const e of edges) {
    const ci = communityOf.get(e.source);
    const cj = communityOf.get(e.target);
    if (ci !== undefined && cj !== undefined) {
      if (ci === cj) {
        internalCount.set(ci, (internalCount.get(ci) || 0) + 1);
      } else {
        externalCount.set(ci, (externalCount.get(ci) || 0) + 1);
        externalCount.set(cj, (externalCount.get(cj) || 0) + 1);
      }
    }
  }

  for (const [commId, members] of commGroups) {
    const cats = members.map((m) => m.category);
    const catCounts = new Map<string, number>();
    for (const cat of cats) {
      catCounts.set(cat, (catCounts.get(cat) || 0) + 1);
    }
    const topCat =
      [...catCounts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] || "未分类";

    const numRange = formatNumRange(members);
    const name = numRange ? `${topCat} (${numRange})` : topCat;

    const internal = internalCount.get(commId) || 0;
    const external = externalCount.get(commId) || 0;
    const total = internal + external;
    const cohesion = total > 0 ? Math.round((internal / total) * 10000) / 10000 : 0;

    // 按编号排序
    members.sort((a, b) => (a.number ?? 999) - (b.number ?? 999));

    communities.push({
      name,
      size: members.length,
      numRange,
      cohesion,
      members,
    });
  }

  // 按 size 降序
  communities.sort((a, b) => b.size - a.size);

  return {
    totalConcepts: finalConcepts.length,
    totalEdges: edges.length,
    totalCommunities: communities.length,
    resolution,
    concepts: finalConcepts,
    edges,
    communities,
  };
}

// ─── CLI 入口 ──────────────────────────────────────────────────────

function main() {
  const args = process.argv.slice(2);
  const dirIndex = args.indexOf("--dir");
  const filesIndex = args.indexOf("--files");
  const resIndex = args.indexOf("--resolution");
  const outIndex = args.indexOf("--output");

  const dir = dirIndex !== -1 ? args[dirIndex + 1] : null;
  const fileFilter =
    filesIndex !== -1
      ? args.slice(filesIndex + 1).filter((a) => !a.startsWith("--"))
      : null;
  const resolution = resIndex !== -1 ? parseFloat(args[resIndex + 1]) : 0.5;
  const outputFile = outIndex !== -1 ? args[outIndex + 1] : null;

  if (!dir) {
    console.error(JSON.stringify({ error: "请指定 --dir 文档目录路径" }));
    process.exit(1);
  }

  if (!existsSync(dir)) {
    console.error(JSON.stringify({ error: `目录不存在: ${dir}` }));
    process.exit(1);
  }

  const result = clusterPipeline(dir, fileFilter, resolution);

  const output = JSON.stringify(result, null, 2);
  if (outputFile) {
    import("fs").then((fs) => fs.writeFileSync(outputFile, output, "utf-8"));
    console.log(`已写入: ${outputFile}`);
  } else {
    console.log(output);
  }
}

if (import.meta.main) {
  main();
}
