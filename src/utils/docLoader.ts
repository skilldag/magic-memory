/**
 * 根据概念 path 加载真实文档内容
 * path 格式如 './docs/level-1/03-ear.md'
 * 对应 URL: /docs/level-1/03-ear.md
 */

const loadedCache = new Map<string, string>()

export function clearDocCache(path?: string) {
  if (path) {
    loadedCache.delete(path)
  } else {
    loadedCache.clear()
  }
}

export function getDocUrl(path: string): string {
  return '/' + path.replace(/^\.\//, '')
}

export async function loadDocContent(path: string): Promise<string | null> {
  if (loadedCache.has(path)) {
    return loadedCache.get(path)!
  }

  try {
    const url = getDocUrl(path)
    const resp = await fetch(url)
    if (!resp.ok) return null
    const text = await resp.text()
    loadedCache.set(path, text)
    return text
  } catch {
    return null
  }
}
