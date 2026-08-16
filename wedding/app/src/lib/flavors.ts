import type { CSSProperties } from 'react'

/**
 * Everything sits on warm cream paper. Text keeps ONE unified rose
 * accent; the pastel variety lives in the floral illustrations.
 */
export interface Flavor {
  deep: string
  deeper: string
}

export const FLAVORS = {
  blossom: { deep: '#c73a72', deeper: '#a72d67' },
  sky: { deep: '#3e8ec4', deeper: '#2f6f9e' },
  mint: { deep: '#3a9a66', deeper: '#2e7a50' },
  lemon: { deep: '#a87f1c', deeper: '#8a6714' },
  peach: { deep: '#d1742f', deeper: '#b25b1d' },
  lavender: { deep: '#8a58bd', deeper: '#71449e' },
} as const satisfies Record<string, Flavor>

/** section-level style: cream stays, only the accent variables change */
export function flavorStyle(f: Flavor, extra?: CSSProperties): CSSProperties {
  return {
    '--champ': f.deep,
    '--champ2': f.deeper,
    ...extra,
  } as CSSProperties
}
