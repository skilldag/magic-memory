import { useState, useRef, useCallback } from 'react'

interface UsePanelResizingOptions {
  initialWidth?: number
  minWidth?: number
  maxWidth?: number
  maxWidthRatio?: number
}

export function usePanelResizing(options: UsePanelResizingOptions = {}) {
  const {
    initialWidth = 420,
    minWidth = 300,
    maxWidth = 720,
    maxWidthRatio = 0.6,
  } = options

  const [width, setWidth] = useState(initialWidth)
  const isResizing = useRef(false)
  const startXRef = useRef(0)
  const startWidthRef = useRef(0)
  const containerRef = useRef<HTMLDivElement | null>(null)

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    isResizing.current = true
    startXRef.current = e.clientX
    startWidthRef.current = width
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }, [width])

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isResizing.current || !containerRef.current) return
    const containerRect = containerRef.current.getBoundingClientRect()
    const maxAllowedWidth = Math.min(containerRect.width * maxWidthRatio, maxWidth)
    const newWidth = Math.max(minWidth, Math.min(maxAllowedWidth, startWidthRef.current + (startXRef.current - e.clientX)))
    setWidth(newWidth)
  }, [minWidth, maxWidth, maxWidthRatio])

  const handleMouseUp = useCallback(() => {
    isResizing.current = false
    document.body.style.cursor = ''
    document.body.style.userSelect = ''
  }, [])

  const setContainerRef = useCallback((el: HTMLDivElement | null) => {
    containerRef.current = el
  }, [])

  return {
    width,
    setWidth,
    isResizing,
    handleMouseDown,
    handleMouseMove,
    handleMouseUp,
    setContainerRef,
    containerRef: { current: containerRef.current },
  }
}