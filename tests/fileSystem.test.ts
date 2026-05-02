/**
 * readMdFilesBatched 测试
 *
 * 运行: npx tsx tests/fileSystem.test.ts
 */

// Mock FileSystem API helpers
function createMockDirHandle(files: Record<string, string>): any {
  const entries = Object.entries(files).map(([name, content]) => [
    name,
    {
      kind: 'file' as const,
      name,
      getFile: async () => ({
        text: async () => content,
      }),
    },
  ])
  return {
    entries: async function* () { yield* entries },
  }
}

function createMockDirWithSubdirs(structure: Record<string, Record<string, string> | string>): any {
  const entries = Object.entries(structure).map(([name, value]) => {
    if (typeof value === 'string') {
      return [
        name,
        {
          kind: 'file' as const,
          name,
          getFile: async () => ({
            text: async () => value as string,
          }),
        },
      ]
    }
    // Recurse for subdirectories
    const subHandle = createMockDirWithSubdirs(value as Record<string, string>)
    return [
      name,
      { kind: 'directory' as const, name, entries: subHandle.entries },
    ]
  })
  return { entries: async function* () { yield* entries } }
}

async function main() {
  let passed = 0
  let failed = 0

  function assert(condition: boolean, msg: string) {
    if (condition) {
      console.log(`  ✅ ${msg}`)
      passed++
    } else {
      console.log(`  ❌ ${msg}`)
      failed++
    }
  }

  // Import the module
  const { readMdFilesBatched } = await import('../src/utils/fileSystem')

  // Test 1: Should yield files in batches of the specified size
  console.log('\nTest 1: Batch size')
  {
    const files: Record<string, string> = {}
    for (let i = 0; i < 25; i++) {
      files[`doc-${i}.md`] = `# Doc ${i}\ncontent`
    }
    const handle = createMockDirHandle(files)
    const batches: string[][] = []
    for await (const batch of readMdFilesBatched(handle, '', 10)) {
      batches.push(batch.map(f => f.path))
    }
    assert(batches.length === 3, `Expected 3 batches, got ${batches.length}`)
    assert(batches[0].length === 10, `First batch should have 10, got ${batches[0].length}`)
    assert(batches[1].length === 10, `Second batch should have 10, got ${batches[1].length}`)
    assert(batches[2].length === 5, `Third batch should have 5, got ${batches[2].length}`)
  }

  // Test 2: Should skip non-markdown files
  console.log('\nTest 2: Skip non-markdown files')
  {
    const files: Record<string, string> = {
      'readme.md': '# Readme',
      'notes.txt': 'plain text',
      'index.md': '# Index',
    }
    const handle = createMockDirHandle(files)
    let total = 0
    for await (const batch of readMdFilesBatched(handle, '', 10)) {
      total += batch.length
    }
    assert(total === 2, `Expected 2 files (.md only), got ${total}`)
  }

  // Test 3: Should skip empty files
  console.log('\nTest 3: Skip empty files')
  {
    const files: Record<string, string> = {
      'empty.md': '',
      'content.md': '# Has Content',
    }
    const handle = createMockDirHandle(files)
    let total = 0
    for await (const batch of readMdFilesBatched(handle, '', 10)) {
      total += batch.length
    }
    assert(total === 1, `Expected 1 non-empty file, got ${total}`)
  }

  // Test 4: Should recurse into subdirectories
  console.log('\nTest 4: Recurse into subdirectories')
  {
    const handle = createMockDirWithSubdirs({
      'root.md': '# Root',
      'sub': {
        'nested.md': '# Nested',
        'another.md': '# Another',
      },
      'empty-sub': {},
    })
    let total = 0
    for await (const batch of readMdFilesBatched(handle, '', 10)) {
      total += batch.length
    }
    assert(total === 3, `Expected 3 files (root + 2 nested), got ${total}`)
  }

  // Summary
  console.log('\n========================')
  console.log(`Results: ${passed} passed, ${failed} failed`)
  console.log('========================')
  process.exit(failed > 0 ? 1 : 0)
}

main().catch(err => {
  console.error('Test runner error:', err)
  process.exit(1)
})
