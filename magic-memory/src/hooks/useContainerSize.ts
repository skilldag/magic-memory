import { useState, useEffect, useRef } from 'react'

interface ContainerSize {
  width: number
  height: number
}

export function useContainerSize<T extends HTMLElement>() {
  const containerRef = useRef<T>(null)
  const [size, setSize] = useState<ContainerSize>({ width: 0, height: 0 })

  useEffect(() => {
    const el = containerRef.current
    if (!el) return

    const ro = new ResizeObserver(entries => {
      for (const entry of entries) {
        setSize({
          width: entry.contentRect.width,
          height: entry.contentRect.height,
        })
      }
    })

    ro.observe(el)
    setSize({ width: el.clientWidth, height: el.clientHeight })

    return () => ro.disconnect()
  }, [])

  return { containerRef, size }
}