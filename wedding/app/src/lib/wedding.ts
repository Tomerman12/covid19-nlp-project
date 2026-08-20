export const COUPLE = { one: 'שחף', two: 'תומר', full: 'שחף טורטלטאוב · תומר מן' }
export const COUPLE_EN = { one: 'Shachaf', two: 'Tomer' }
export const VENUE = 'אולם אביגדור'
export const ADDR = 'בן אביגדור 24, תל אביב'
export const DATE_LABEL = '28.10.2026'
export const DAY_LABEL = 'יום רביעי'
export const TIME_LABEL = '19:30'
export const TARGET_TS = new Date('2026-10-28T19:30:00+02:00').getTime()

const TITLE = 'החתונה של שחף ותומר 💍'
const DESC = '19:30 קבלת פנים · 21:30 חופה · אחר־כך רוקדים עד הבוקר!'

export const wazeUrl =
  'https://waze.com/ul?q=' + encodeURIComponent(ADDR) + '&navigate=yes'

export const gmapsUrl =
  'https://www.google.com/maps/search/?api=1&query=' +
  encodeURIComponent(VENUE + ', ' + ADDR)

export const gcalUrl =
  'https://calendar.google.com/calendar/render?action=TEMPLATE' +
  '&text=' + encodeURIComponent(TITLE) +
  '&dates=20261028T193000/20261029T000000' +
  '&ctz=Asia/Jerusalem' +
  '&location=' + encodeURIComponent(VENUE + ', ' + ADDR) +
  '&details=' + encodeURIComponent(DESC)

const icsBody = [
  'BEGIN:VCALENDAR', 'VERSION:2.0', 'PRODID:-//Shachaf+Tomer//Wedding//HE', 'CALSCALE:GREGORIAN',
  'BEGIN:VEVENT',
  'UID:shachaf-tomer-20261028@wedding',
  'DTSTAMP:20260722T090000Z',
  'DTSTART:20261028T173000Z',
  'DTEND:20261028T220000Z',
  'SUMMARY:' + TITLE,
  'DESCRIPTION:' + DESC.split(',').join('\\,'),
  'LOCATION:' + (VENUE + ', ' + ADDR).split(',').join('\\,'),
  'END:VEVENT', 'END:VCALENDAR',
].join('\r\n')

export const icsUrl = 'data:text/calendar;charset=utf-8,' + encodeURIComponent(icsBody)

/* party-mode soundtrack: full song opens on YouTube; the chorus clip is embedded */
export const SONG_URL = 'https://www.youtube.com/watch?v=5sYxrCWNV20'
export const SONG_LABEL = 'אהבה בסוף הקיץ · צביקה פיק'

export const SCHEDULE = [
  {
    time: '19:30',
    range: '19:30–21:30',
    title: 'קבלת פנים',
    desc: 'אוכל טוב, דרינקים קרים והמון חיבוקים — תבואו רעבים.',
  },
  {
    time: '21:30',
    range: '21:30–22:00',
    title: 'חופה',
    desc: 'הרגע הגדול. מומלץ להצטייד בטישו.',
  },
  {
    time: '22:00',
    range: 'עד הבוקר',
    title: 'רחבה!',
    desc: 'ריקודים, קונפטי וכיף — עד שהרגליים יגידו די. או שלא.',
  },
] as const
