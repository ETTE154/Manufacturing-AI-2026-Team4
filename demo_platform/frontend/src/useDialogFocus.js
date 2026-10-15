import {useEffect,useRef,useState} from 'react';

export function useNarrowScreen(){
  const [narrow,setNarrow]=useState(()=>window.matchMedia('(max-width:760px)').matches);
  useEffect(()=>{const media=window.matchMedia('(max-width:760px)'),update=()=>setNarrow(media.matches);media.addEventListener('change',update);return()=>media.removeEventListener('change',update);},[]);
  return narrow;
}
export function useDialogFocus(ref,active,onClose,fallbackSelector){
  const close=useRef(onClose);close.current=onClose;
  useEffect(()=>{
    if(!active)return;
    const previous=document.activeElement,overflow=document.body.style.overflow;
    const controls=()=>[...(ref.current?.querySelectorAll('button,a[href],input,select,textarea,summary,[tabindex]:not([tabindex="-1"])')||[])].filter(el=>!el.disabled&&el.getClientRects().length&&!el.closest('[inert]'));
    document.body.style.overflow='hidden';
    const frame=requestAnimationFrame(()=>{(controls()[0]||ref.current)?.focus({preventScroll:true});});
    const key=event=>{
      if(event.key==='Escape'){event.preventDefault();close.current();return;}
      if(event.key!=='Tab')return;
      const list=controls(),first=list[0],last=list.at(-1);
      if(!first){event.preventDefault();ref.current?.focus();return;}
      if(!ref.current?.contains(document.activeElement)||(event.shiftKey&&document.activeElement===first)){event.preventDefault();(event.shiftKey?last:first).focus();}
      else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first.focus();}
    };
    document.addEventListener('keydown',key);
    return()=>{
      cancelAnimationFrame(frame);document.body.style.overflow=overflow;document.removeEventListener('keydown',key);
      const available=el=>el?.isConnected&&el.getClientRects().length&&!el.closest('[inert]');
      const fallback=fallbackSelector?document.querySelector(fallbackSelector):null;
      const target=available(previous)?previous:available(fallback)?fallback:null;
      target?.focus({preventScroll:true});
    };
  },[active,ref,fallbackSelector]);
}
