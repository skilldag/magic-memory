import * as vscode from 'vscode'

export async function extractEnclosingSymbolName(
  document: vscode.TextDocument,
  line: number
): Promise<string | null> {
  const symbols = await vscode.commands.executeCommand<vscode.DocumentSymbol[]>(
    'vscode.executeDocumentSymbolProvider',
    document.uri
  )
  if (!symbols) return null

  const validKinds = new Set([
    vscode.SymbolKind.Function,
    vscode.SymbolKind.Method,
    vscode.SymbolKind.Class,
    vscode.SymbolKind.Interface,
    vscode.SymbolKind.Struct,
    vscode.SymbolKind.Enum,
  ])

  function findDeepest(
    list: vscode.DocumentSymbol[],
    line: number
  ): vscode.DocumentSymbol | null {
    for (const s of list) {
      if (s.range.start.line <= line && s.range.end.line >= line) {
        const deeper = s.children.length > 0 ? findDeepest(s.children, line) : null
        return deeper || (validKinds.has(s.kind) ? s : null)
      }
    }
    return null
  }

  const deepest = findDeepest(symbols, line)
  return deepest?.name || null
}
