import * as vscode from 'vscode'
import { createConcept } from './createConcept'

export function activate(context: vscode.ExtensionContext) {
  const disposable = vscode.commands.registerCommand(
    'magicMemory.createConcept',
    createConcept
  )
  context.subscriptions.push(disposable)
}

export function deactivate() {}
