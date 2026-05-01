import { useEffect } from 'react'
import { useKnowledgeGraphStore } from '../store/knowledgeGraphStore'

export function useLinkModeEvents(linkMode: boolean) {
  const storeSetLinkMode = useKnowledgeGraphStore(s => s.setLinkMode)

  useEffect(() => {
    const toggleHandler = () => {
      console.log('[KnowledgeGraphView] toggle-link-mode received, setting linkMode to true')
    }
    const exitHandler = () => {
      console.log('[KnowledgeGraphView] exit-link-mode received, setting linkMode to false')
    }
    window.addEventListener('toggle-link-mode', toggleHandler)
    window.addEventListener('exit-link-mode', exitHandler)
    return () => {
      window.removeEventListener('toggle-link-mode', toggleHandler)
      window.removeEventListener('exit-link-mode', exitHandler)
    }
  }, [])

  useEffect(() => {
    console.log('[KnowledgeGraphView] linkMode state changed to:', linkMode)
  }, [linkMode])

  useEffect(() => {
    storeSetLinkMode(linkMode)
  }, [linkMode, storeSetLinkMode])
}