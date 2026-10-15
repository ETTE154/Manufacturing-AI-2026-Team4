import { chromium } from 'file:///C:/Users/jeon/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
import fs from 'node:fs';import path from 'node:path';import {fileURLToPath,pathToFileURL} from 'node:url';
const root=path.dirname(fileURLToPath(import.meta.url));
const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
const page=await browser.newPage({viewport:{width:1280,height:900}});const errors=[];
page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
await page.goto(pathToFileURL(path.join(root,'data_report.html')).href);
await page.waitForFunction(()=>[...document.images].every(i=>i.complete&&i.naturalWidth>0));
const images=await page.locator('img').evaluateAll(xs=>xs.map(x=>({src:x.getAttribute('src'),complete:x.complete,naturalWidth:x.naturalWidth})));
const links=await page.locator('a').evaluateAll(xs=>xs.map(x=>x.getAttribute('href')));
const missing=links.filter(h=>h&&!h.startsWith('http')&&!h.startsWith('#')&&!fs.existsSync(path.resolve(root,h)));
const layouts=[];
for(const width of [1280,768,390]){await page.setViewportSize({width,height:900});layouts.push(await page.evaluate(()=>({width:innerWidth,scrollWidth:document.documentElement.scrollWidth,overflow:document.documentElement.scrollWidth>innerWidth})));}
const style=await page.evaluate(()=>({headerBackground:getComputedStyle(document.querySelector('header')).backgroundColor,bodyTextColor:getComputedStyle(document.body).color,sections:document.querySelectorAll('section').length}));
await page.setViewportSize({width:1280,height:900});await page.screenshot({path:path.join(root,'html_preview.png'),fullPage:true});
const result={pass:errors.length===0&&missing.length===0&&layouts.every(x=>!x.overflow)&&images.length===16&&images.every(i=>i.complete&&i.naturalWidth>0),images:images.length,layouts,errors,missing_local_links:missing,styles:style};
fs.writeFileSync(path.join(root,'browser_qa.json'),JSON.stringify(result,null,2));await browser.close();console.log(JSON.stringify(result));if(!result.pass)process.exitCode=1;
