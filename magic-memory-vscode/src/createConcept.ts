import * as vscode from 'vscode'
import * as path from 'path'
import { execFileSync } from 'child_process'
import { extractEnclosingSymbolName } from './symbolExtractor'
import { findMatchingProject, listRepoProjects } from './projectFinder'

export async function createConcept() {
  const editor = vscode.window.activeTextEditor
  if (!editor) {
    vscode.window.showErrorMessage('没有打开的文件')
    return
  }

  const document = editor.document
  const selection = editor.selection
  const workspaceFolder = vscode.workspace.getWorkspaceFolder(document.uri)
  if (!workspaceFolder) {
    vscode.window.showErrorMessage('文件不在工作区内')
    return
  }
  const workspaceRoot = workspaceFolder.uri.fsPath

  let conceptName = await extractEnclosingSymbolName(document, selection.start.line)
  if (!conceptName) {
    conceptName = path.basename(document.fileName, path.extname(document.fileName))
  }

  let project = findMatchingProject(workspaceRoot)
  if (!project) {
    const repos = listRepoProjects()
    if (repos.length === 0) {
      const action = await vscode.window.showErrorMessage(
        '没有注册的仓库项目。先在终端中运行: memo init-repo <path>',
        { modal: true },
        '复制命令'
      )
      if (action === '复制命令') {
        vscode.env.clipboard.writeText('memo init-repo ' + workspaceRoot)
      }
      return
    }
    const picked = await vscode.window.showQuickPick(
      repos.map(p => ({
        label: p.name,
        description: p.sourceDir,
        detail: `${p.conceptCount} concepts`,
        project: p,
      })),
      { placeHolder: '选择要添加到的项目' }
    )
    if (!picked) return
    project = picked.project
  }

  const inputName = await vscode.window.showInputBox({
    prompt: '概念名称',
    value: conceptName,
    placeHolder: '输入概念名称',
  })
  if (!inputName) return

  const description = await vscode.window.showInputBox({
    prompt: '描述（可选，如"核心实现"）',
    placeHolder: '可选',
  })

  const filePath = path.relative(workspaceRoot, document.uri.fsPath)
  const lineStart = selection.start.line + 1
  const lineEnd = selection.end.line + 1

  const cliPath = vscode.workspace.getConfiguration('magicMemory').get<string>('cliPath', 'memo')
  const args = [
    'add-concept', project.id,
    '--name', inputName,
    '--file', filePath,
    '--lines', `${lineStart}-${lineEnd}`,
  ]
  if (description) {
    args.push('--desc', description)
  }

  try {
    execFileSync(cliPath, args, { cwd: workspaceRoot })
    vscode.window.showInformationMessage(`✓ 概念 "${inputName}" 已创建`)
  } catch (err: any) {
    const message = err.stderr?.toString() || err.message || '未知错误'
    vscode.window.showErrorMessage(`创建失败: ${message}`)
  }
}
