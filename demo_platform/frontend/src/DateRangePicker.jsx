import React, {useEffect, useId, useLayoutEffect, useMemo, useRef, useState} from 'react';
import {CalendarDays, ChevronDown, ChevronLeft, ChevronRight, X} from 'lucide-react';
import {availableMonths, availableYears, calendarMonth, clampDate, clampMonth, dateBounds, dateInMonth, dateLabel, displayDate, moveCalendarDate, normalizeDateRange, selectRangeDate, shiftMonth} from './dateRange';

const WEEKDAYS = ['일','월','화','수','목','금','토'];
const MOVE_KEYS = new Set(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End','PageUp','PageDown']);

// Controlled recorded-date selector. maxSpanDays=1 is the raw-data single-day
// mode. Only days and month/year options inside [min,max] are ever rendered.
export function DateRangePicker({min, max, start='', end='', onChange, disabled=false, maxSpanDays}) {
  const id = useId(), root = useRef(null), trigger = useRef(null), popup = useRef(null), days = useRef({}), focusGrid = useRef(false);
  const [open, setOpen] = useState(false), [part, setPart] = useState('start');
  const bounds = useMemo(() => dateBounds(min,max), [min,max]);
  const range = useMemo(() => normalizeDateRange({start,end,min,max,maxSpanDays}), [start,end,min,max,maxSpanDays]);
  const single = maxSpanDays === 1, hasSelection = !!(start || end), unavailable = disabled || !bounds;
  const [focused, setFocused] = useState(() => range.start), [month, setMonth] = useState(() => range.start.slice(0,7));
  const [placement, setPlacement] = useState({visibility:'hidden'});
  const view = calendarMonth(month,min,max), year = Number(view.month.slice(0,4));
  const previousMonth = shiftMonth(view.month,-1,min,max), nextMonth = shiftMonth(view.month,1,min,max);

  useEffect(() => {
    if (bounds && hasSelection && (range.start !== start || range.end !== end)) onChange?.(range);
  }, [bounds,hasSelection,range,start,end,onChange]);

  useEffect(() => {
    if (!bounds || disabled) { setOpen(false); return; }
    setFocused(value => clampDate(value || range.start,min,max));
    setMonth(value => clampMonth(value || range.start.slice(0,7),min,max));
  }, [min,max,disabled,bounds,range.start]);

  const close = (restoreFocus=true) => {
    focusGrid.current = false;
    setOpen(false);
    if (restoreFocus) trigger.current?.focus({preventScroll:true});
  };

  useEffect(() => {
    if (!open) return;
    const outside = event => { if (!root.current?.contains(event.target)) setOpen(false); };
    const escape = event => {
      if (event.key !== 'Escape') return;
      event.preventDefault(); event.stopPropagation(); setOpen(false); trigger.current?.focus({preventScroll:true});
    };
    document.addEventListener('pointerdown',outside);
    document.addEventListener('keydown',escape);
    return () => { document.removeEventListener('pointerdown',outside); document.removeEventListener('keydown',escape); };
  }, [open]);

  useLayoutEffect(() => {
    if (!open) return;
    const place = event => {
      if (event?.target instanceof Element && popup.current?.contains(event.target)) return;
      if (!popup.current || !trigger.current) return;
      const rect = trigger.current.getBoundingClientRect(), viewportWidth = document.documentElement.clientWidth;
      const width = Math.min(360,viewportWidth-24);
      if (rect.bottom < 0 || rect.top > window.innerHeight) { setOpen(false); return; }
      const height = Math.min(popup.current.scrollHeight,window.innerHeight-24);
      const left = Math.max(12,Math.min(rect.left,viewportWidth-width-12));
      const below = window.innerHeight-rect.bottom-12;
      const top = below < height && rect.top > below ? Math.max(12,rect.top-height-8) : Math.min(rect.bottom+8,Math.max(12,window.innerHeight-height-12));
      setPlacement({left,top,width,visibility:'visible'});
    };
    place();
    window.addEventListener('resize',place);
    window.addEventListener('scroll',place,true);
    return () => { window.removeEventListener('resize',place); window.removeEventListener('scroll',place,true); };
  }, [open,month,single]);

  useEffect(() => {
    if (open && focusGrid.current) {
      days.current[focused]?.focus({preventScroll:true});
      focusGrid.current = false;
    }
  }, [open,focused,month,part]);

  const show = () => {
    if (open) { close(); return; }
    const next = range.start || min;
    focusGrid.current = true;
    setPart('start'); setFocused(next); setMonth(next.slice(0,7)); setPlacement({visibility:'hidden'}); setOpen(true);
  };
  const switchPart = value => {
    focusGrid.current = true;
    setPart(value);
    const next = range[value] || range.start;
    setFocused(next); setMonth(next.slice(0,7));
  };
  const changeMonth = (value,moveFocus=true) => {
    focusGrid.current = moveFocus;
    const next = clampMonth(value,min,max);
    setMonth(next); setFocused(dateInMonth(next,focused,min,max));
  };
  const choose = date => {
    const next = selectRangeDate(range,date,part,{min,max,maxSpanDays});
    focusGrid.current = true;
    onChange?.(next);
    setFocused(date); setMonth(date.slice(0,7));
    if (single || part === 'end') close();
    else setPart('end');
  };
  const move = (event,date) => {
    if (!MOVE_KEYS.has(event.key)) return;
    event.preventDefault();
    focusGrid.current = true;
    const next = moveCalendarDate(date,event.key,{min,max,shiftKey:event.shiftKey});
    setFocused(next); setMonth(next.slice(0,7));
  };
  const selectedText = !hasSelection ? (single?'조회 날짜 선택':'조회 기간 선택') : single ? displayDate(range.start) : `${displayDate(range.start)} — ${displayDate(range.end)}`;

  return <div className="date-range-picker" ref={root} data-min={min} data-max={max} data-single-day={single}
    onBlur={event => { if (event.relatedTarget && !event.currentTarget.contains(event.relatedTarget)) setOpen(false); }}>
    <button className="date-range-trigger" type="button" ref={trigger} disabled={unavailable} aria-haspopup="dialog" aria-expanded={open} aria-controls={id} onClick={show}>
      <CalendarDays size={16} aria-hidden="true"/><span>{!bounds?'보유 기간 확인 중':selectedText}</span><ChevronDown size={14} aria-hidden="true"/>
    </button>
    {open && bounds && <section className="date-range-popover" role="dialog" aria-label={single?'원본 조회 날짜 선택':'집계 조회 기간 선택'} aria-describedby={`${id}-availability ${id}-help`} id={id} ref={popup} style={placement}>
      <div className="date-range-header"><strong>{single?'조회 날짜':'조회 기간'}</strong><button type="button" className="icon-button" aria-label="날짜 선택 닫기" onClick={() => close()}><X size={18}/></button></div>
      <p className="date-range-availability" id={`${id}-availability`}>선택 가능 <strong>{displayDate(min)} — {displayDate(max)}</strong></p>
      {!single && <div className="date-range-fields">{[['start','시작일'],['end','종료일']].map(([value,label]) => <button type="button" key={value} className={part===value?'active':''} aria-pressed={part===value} onClick={() => switchPart(value)}><span>{label}</span><strong>{hasSelection?displayDate(range[value]):'선택 전'}</strong></button>)}</div>}
      <div className="date-range-monthbar"><button type="button" className="icon-button" aria-label="이전 달" disabled={previousMonth===view.month} onClick={() => changeMonth(previousMonth)}><ChevronLeft size={18}/></button><div><select aria-label="달력 연도" value={year} onChange={event => changeMonth(`${event.target.value}-${view.month.slice(5)}`,false)}>{availableYears(min,max).map(value => <option value={value} key={value}>{value}년</option>)}</select><select aria-label="달력 월" value={view.month} onChange={event => changeMonth(event.target.value,false)}>{availableMonths(year,min,max).map(value => <option value={value} key={value}>{Number(value.slice(5))}월</option>)}</select></div><button type="button" className="icon-button" aria-label="다음 달" disabled={nextMonth===view.month} onClick={() => changeMonth(nextMonth)}><ChevronRight size={18}/></button></div>
      <div className="date-range-grid" role="grid" aria-label={view.label}>
        <div className="date-range-weekdays" role="row">{WEEKDAYS.map(value => <span key={value} role="columnheader" aria-label={`${value}요일`}>{value}</span>)}</div>
        {view.weeks.map((week,index) => <div className="date-range-week" role="row" key={`${view.month}-${index}`}>{week.map((cell,column) => {
          if (!cell) return <div className="date-range-cell blank" role="gridcell" aria-hidden="true" key={`blank-${column}`}/>;
          const isStart=hasSelection&&cell.date===range.start, isEnd=hasSelection&&cell.date===range.end, inRange=hasSelection&&cell.date>=range.start&&cell.date<=range.end;
          return <div role="gridcell" aria-selected={inRange} className={`date-range-cell${inRange?' in-range':''}${isStart?' range-start':''}${isEnd?' range-end':''}`} key={cell.date}><button type="button" ref={element => { if(element)days.current[cell.date]=element;else delete days.current[cell.date]; }} tabIndex={focused===cell.date?0:-1} aria-label={dateLabel(cell.date)} aria-pressed={inRange} data-date={cell.date} onFocus={() => setFocused(cell.date)} onKeyDown={event => move(event,cell.date)} onClick={() => choose(cell.date)}>{cell.day}</button></div>;
        })}</div>)}
      </div>
      <div className="date-range-footer"><p id={`${id}-help`}>{single?'원본 조회는 하루를 선택합니다.':'시작일을 선택한 뒤 종료일을 선택하세요.'}<span>방향키로 날짜 이동 · Page Up/Down으로 월 이동</span></p><button className="button small secondary" type="button" onClick={() => {onChange?.(range);close();}}>선택 완료</button></div>
    </section>}
  </div>;
}

export default DateRangePicker;
