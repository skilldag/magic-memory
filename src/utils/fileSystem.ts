export async function readMdFiles(
  dirHandle: FileSystemDirectoryHandle,
  pathPrefix = ''
): Promise<{ path: string; content: string }[]> {
  const results: { path: string; content: string }[] = []
  for await (const [name, entry] of (dirHandle as any).entries()) {
    const entryPath = pathPrefix ? `${pathPrefix}/${name}` : name
    if (entry.kind === 'directory' && !name.startsWith('.')) {
      results.push(...await readMdFiles(entry, entryPath))
    } else if (entry.kind === 'file' && name.endsWith('.md')) {
      try {
        const file = await (entry as FileSystemFileHandle).getFile()
        const content = await file.text()
        if (content.trim()) {
          results.push({ path: entryPath, content })
        }
      } catch { }
    }
  }
  return results
}

export async function* readMdFilesBatched(
  dirHandle: FileSystemDirectoryHandle,
  pathPrefix = '',
  batchSize = 10,
): AsyncGenerator<{ path: string; content: string }[], void, undefined> {
  let batch: { path: string; content: string }[] = []
  for await (const [name, entry] of (dirHandle as any).entries()) {
    const entryPath = pathPrefix ? `${pathPrefix}/${name}` : name
    if (entry.kind === 'directory' && !name.startsWith('.')) {
      for await (const files of readMdFilesBatched(entry, entryPath, batchSize)) {
        batch.push(...files)
        if (batch.length >= batchSize) { yield batch; batch = [] }
      }
    } else if (entry.kind === 'file' && name.endsWith('.md')) {
      try {
        const file = await (entry as FileSystemFileHandle).getFile()
        const content = await file.text()
        if (content.trim()) {
          batch.push({ path: entryPath, content })
          if (batch.length >= batchSize) { yield batch; batch = [] }
        }
      } catch { }
    }
  }
  if (batch.length > 0) yield batch
}