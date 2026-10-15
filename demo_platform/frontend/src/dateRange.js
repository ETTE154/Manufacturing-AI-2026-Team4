// Recorded calendar dates are labels, not browser-local instants. UTC is used
// only for integer day arithmetic, including across DST changes.
const DAY_MS = 86400000;
const pad = (value, width=2) => String(value).padStart(width, '0');

export function dateDay(value) {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return null;
  const [year, month, day] = value.split('-').map(Number);
  const date = new Date(0);
  date.setUTCHours(0, 0, 0, 0);
  date.setUTCFullYear(year, month - 1, day);
  if (date.getUTCFullYear() !== year || date.getUTCMonth() !== month - 1 || date.getUTCDate() !== day) return null;
  return date.getTime() / DAY_MS;
}

export function dayDate(value) {
  if (!Number.isInteger(value)) return '';
  const date = new Date(value * DAY_MS);
  const year = date.getUTCFullYear();
  return Number.isFinite(date.getTime()) && year >= 0 && year <= 9999
    ? `${pad(year, 4)}-${pad(date.getUTCMonth() + 1)}-${pad(date.getUTCDate())}` : '';
}

export function dateBounds(min, max) {
  const first = dateDay(min), last = dateDay(max);
  return first === null || last === null || first > last ? null : {min, max, first, last};
}

export function clampDate(value, min, max, fallback=min) {
  const bounds = dateBounds(min, max);
  if (!bounds) return '';
  const parsed = dateDay(value), alternate = dateDay(fallback);
  return dayDate(Math.max(bounds.first, Math.min(bounds.last, parsed ?? alternate ?? bounds.first)));
}

export function normalizeDateRange({start='', end='', min, max, maxSpanDays}) {
  const bounds = dateBounds(min, max);
  if (!bounds) return {start:'', end:''};
  const first = clampDate(start || end, min, max);
  let last = clampDate(end || first, min, max, first);
  if (last < first) last = first;
  const limit = Number.isInteger(maxSpanDays) && maxSpanDays > 0 ? maxSpanDays : null;
  if (limit) last = dayDate(Math.min(dateDay(last), dateDay(first) + limit - 1));
  return {start:first, end:last};
}

export function selectRangeDate(range, value, part, {min, max, maxSpanDays}) {
  const current = normalizeDateRange({...range, min, max, maxSpanDays});
  const chosen = clampDate(value, min, max);
  if (!chosen) return {start:'', end:''};
  if (maxSpanDays === 1) return {start:chosen, end:chosen};
  let start = part === 'end' ? (current.start < chosen ? current.start : chosen) : chosen;
  let end = part === 'end' ? chosen : (current.end > chosen ? current.end : chosen);
  if (Number.isInteger(maxSpanDays) && maxSpanDays > 0) {
    if (part === 'end') start = dayDate(Math.max(dateDay(start), dateDay(end) - maxSpanDays + 1));
    else end = dayDate(Math.min(dateDay(end), dateDay(start) + maxSpanDays - 1));
  }
  return {start, end};
}

export function validMonth(value) {
  return typeof value === 'string' && /^\d{4}-\d{2}$/.test(value) && dateDay(`${value}-01`) !== null;
}

export function clampMonth(value, min, max) {
  if (!dateBounds(min, max)) return '';
  const first = min.slice(0, 7), last = max.slice(0, 7);
  return validMonth(value) ? (value < first ? first : value > last ? last : value) : first;
}

export function availableYears(min, max) {
  if (!dateBounds(min, max)) return [];
  const first = Number(min.slice(0, 4)), last = Number(max.slice(0, 4));
  return Array.from({length:last-first+1}, (_, index) => first + index);
}

export function availableMonths(year, min, max) {
  if (!dateBounds(min, max)) return [];
  return Array.from({length:12}, (_, index) => `${pad(year, 4)}-${pad(index+1)}`)
    .filter(month => month >= min.slice(0, 7) && month <= max.slice(0, 7));
}

export function shiftMonth(value, delta, min, max) {
  const current = clampMonth(value, min, max);
  if (!current) return '';
  const [year, month] = current.split('-').map(Number);
  const indexOf = date => Number(date.slice(0,4))*12+Number(date.slice(5,7))-1;
  const index = Math.max(indexOf(min),Math.min(indexOf(max),year*12+month-1+delta));
  const changed = `${pad(Math.floor(index / 12), 4)}-${pad(((index % 12) + 12) % 12 + 1)}`;
  return clampMonth(changed, min, max);
}

function daysInMonth(year, month) {
  const date = new Date(0);
  date.setUTCFullYear(year, month, 0);
  return date.getUTCDate();
}

export function dateInMonth(month, preferred, min, max) {
  const target = clampMonth(month, min, max);
  if (!target) return '';
  const [year, number] = target.split('-').map(Number);
  const preferredDay = dateDay(preferred) === null ? 1 : Number(preferred.slice(-2));
  return clampDate(`${target}-${pad(Math.min(preferredDay, daysInMonth(year, number)))}`, min, max);
}

export function calendarMonth(month, min, max) {
  const current = clampMonth(month, min, max);
  if (!current) return {month:'', label:'', weeks:[]};
  const [year, number] = current.split('-').map(Number);
  const firstDay = new Date(dateDay(`${current}-01`) * DAY_MS).getUTCDay();
  const count = daysInMonth(year, number);
  const cells = Array.from({length:Math.ceil((firstDay+count)/7)*7}, (_, index) => {
    const day = index-firstDay+1, date = `${current}-${pad(day)}`;
    return day < 1 || day > count || date < min || date > max ? null : {date, day};
  });
  return {month:current, label:`${year}년 ${number}월`, weeks:Array.from({length:cells.length/7}, (_, i) => cells.slice(i*7,i*7+7))};
}

export function moveCalendarDate(value, key, {min, max, shiftKey=false}) {
  const current = clampDate(value, min, max);
  if (!current) return '';
  const day = dateDay(current), weekday = new Date(day*DAY_MS).getUTCDay();
  const shifts = {ArrowLeft:-1, ArrowRight:1, ArrowUp:-7, ArrowDown:7, Home:-weekday, End:6-weekday};
  if (Object.hasOwn(shifts, key)) {
    const bounds = dateBounds(min,max);
    return dayDate(Math.max(bounds.first,Math.min(bounds.last,day+shifts[key])));
  }
  if (key === 'PageUp' || key === 'PageDown') {
    const delta = (key === 'PageUp' ? -1 : 1) * (shiftKey ? 12 : 1);
    return dateInMonth(shiftMonth(current.slice(0,7), delta, min, max), current, min, max);
  }
  return current;
}

export const displayDate = value => dateDay(value) === null ? '날짜 선택' : value.replaceAll('-', '.');
export function dateLabel(value) {
  const day = dateDay(value);
  if (day === null) return '';
  const [year, month, date] = value.split('-').map(Number);
  return `${year}년 ${month}월 ${date}일 ${['일','월','화','수','목','금','토'][new Date(day*DAY_MS).getUTCDay()]}요일`;
}
