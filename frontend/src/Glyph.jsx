/* Drawn state glyphs. Every state has its own SHAPE, so colour only ever
 * reinforces what the shape already says (PRODUCT.md: status never by
 * colour alone). One 12px grid, one 1.5px stroke. */

const S = 12

export function StateGlyph({ state, size = S, title }) {
  const common = { width: size, height: size, viewBox: '0 0 12 12', 'aria-hidden': title ? undefined : true, role: title ? 'img' : undefined, className: `glyph glyph-${state}` }
  switch (state) {
    case 'up':               // a lit link: solid dot
      return <svg {...common}>{title && <title>{title}</title>}<circle cx="6" cy="6" r="4.25" fill="currentColor" /></svg>
    case 'degraded':         // half lit: split dot
      return (
        <svg {...common}>{title && <title>{title}</title>}
          <circle cx="6" cy="6" r="4.25" fill="none" stroke="currentColor" strokeWidth="1.5" />
          <path d="M6 1.75 A4.25 4.25 0 0 1 6 10.25 Z" fill="currentColor" />
        </svg>
      )
    case 'refused':          // down: hollow square with a bar
      return (
        <svg {...common}>{title && <title>{title}</title>}
          <rect x="1.75" y="1.75" width="8.5" height="8.5" fill="none" stroke="currentColor" strokeWidth="1.5" />
          <path d="M3.5 8.5 L8.5 3.5" stroke="currentColor" strokeWidth="1.5" />
        </svg>
      )
    default:                 // idle / unknown: a dash
      return <svg {...common}>{title && <title>{title}</title>}<path d="M2.5 6 H9.5" stroke="currentColor" strokeWidth="1.5" /></svg>
  }
}

/* Change marks for the log: what kind of move, by shape. */
export function ChangeGlyph({ kind }) {
  const common = { width: 12, height: 12, viewBox: '0 0 12 12', 'aria-hidden': true, className: `glyph change-${kind}` }
  switch (kind) {
    case 'line':
    case 'price':
      return <svg {...common}><path d="M1.5 8.5 L4.5 5.5 L7 7.5 L10.5 3.5" fill="none" stroke="currentColor" strokeWidth="1.5" /></svg>
    case 'tier':
      return <svg {...common}><path d="M1.5 10 H4.5 V6.5 H7.5 V3 H10.5" fill="none" stroke="currentColor" strokeWidth="1.5" /></svg>
    case 'side':
      return <svg {...common}><path d="M2 4 H9 L7 2 M10 8 H3 L5 10" fill="none" stroke="currentColor" strokeWidth="1.5" /></svg>
    case 'refusal':
      return <svg {...common}><rect x="1.75" y="1.75" width="8.5" height="8.5" fill="none" stroke="currentColor" strokeWidth="1.5" /><path d="M3.5 8.5 L8.5 3.5" stroke="currentColor" strokeWidth="1.5" /></svg>
    default:
      return <svg {...common}><path d="M6 2 V10 M2 6 H10" stroke="currentColor" strokeWidth="1.5" /></svg>
  }
}

export function Chevron({ open }) {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true" className={`chevron${open ? ' open' : ''}`}>
      <path d="M4 2.5 L7.5 6 L4 9.5" fill="none" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  )
}
