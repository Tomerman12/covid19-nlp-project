import { MACHINE_VERSION } from '@/lib/machineVersion'

/* The page is bundled to a single file, so anything imported as an asset gets
 * base64'd into it and every visitor pays for it on first load. That is right
 * for a 2 KB icon and wrong for the things in media/: the slot machine, and
 * the song — 395 KB that only a guest who presses the party button ever needs.
 *
 * So those live next to the page instead of inside it, and this builds their
 * URLs. The version stamp means a rebuild is a new URL, which is what lets
 * media/ be cached immutable for a year without ever serving a stale file.
 *
 * __WEDDING_MEDIA__ is the escape hatch for a build that really does want
 * everything inline (a single file to email, say); when it is absent, which
 * is the deployed case, the files are fetched normally. */
const inlined: Record<string, string> | undefined = (window as never as { __WEDDING_MEDIA__?: Record<string, string> })
  .__WEDDING_MEDIA__

export const media = (name: string) => inlined?.[name] ?? `media/${name}?v=${MACHINE_VERSION}`
