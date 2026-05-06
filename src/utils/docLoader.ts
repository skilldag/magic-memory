/**
 * 根据概念 path 加载真实文档内容
 * path 格式如 './docs/user/xxx.md'
 * 如果提供了 baseDir，则通过 /api/read-doc 加载
 * 否则通过 HTTP 从 Vite 静态服务加载
 */

const loadedCache = new Map<string, string>()

export function clearDocCache(path?: string, baseDir?: string) {
  if (path) {
    const cacheKey = baseDir ? `${baseDir}:${path}` : path
    loadedCache.delete(cacheKey)
  } else {
    loadedCache.clear()
  }
}

export async function loadDocContent(path: string, baseDir?: string): Promise<string | null> {
  const cacheKey = baseDir ? `${baseDir}:${path}` : path
  if (loadedCache.has(cacheKey)) {
    return loadedCache.get(cacheKey)!
  }

  // 如果有 baseDir 或是绝对路径，通过 /api/read-doc 加载
  if (baseDir || path.startsWith('/')) {
    const params = new URLSearchParams()
    params.set('path', path)
    if (baseDir) params.set('baseDir', baseDir)
    try {
      const resp = await fetch(`/api/read-doc?${params}`)
      if (!resp.ok) return null
      const data = await resp.json()
      if (data.content) {
        loadedCache.set(cacheKey, data.content)
        return data.content
      }
      return null
    } catch {
      return null
    }
  }

  // 相对路径无 baseDir，通过 Vite HTTP 加载
  const url = '/' + path.replace(/^\.\//, '')
  try {
    const resp = await fetch(url)
    if (!resp.ok) return null
    const text = await resp.text()
    loadedCache.set(cacheKey, text)
    return text
  } catch {
    return null
  }
}
