import { chromium } from 'file:///C:/Users/jeon/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath,pathToFileURL } from 'node:url';
const root=path.dirname(fileURLToPath(import.meta.url));
const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
const page=await browser.newPage({viewport:{width:1280,height:900}});const errors=[];
page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
await page.goto(pathToFileURL(path.join(root,'resampling_report.html')).href);
await page.waitForFunction(()=>[...document.images].every(i=>i.complete&&i.naturalWidth>0));
const options=await page.locator('#model option').evaluateAll(os=>os.map(o=>o.value));
let checks=0;
for(const model of options){for(const split of ['may','test','validation']){
  await page.selectOption('#model',model);await page.selectOption('#split',split);
  await page.waitForFunction(()=>{const i=document.getElementById('chart');return i.complete&&i.naturalWidth>0;});checks++;
}}
const layouts=[];
for(const width of [1280,768,390]){
  await page.setViewportSize({width,height:900});
  layouts.push(await page.evaluate(()=>({width:innerWidth,scrollWidth:document.documentElement.scrollWidth,overflow:document.documentElement.scrollWidth>innerWidth})));
}
const links=await page.locator('a').evaluateAll(xs=>xs.map(x=>x.getAttribute('href')));
const missing=links.filter(h=>h&&!h.startsWith('http')&&!fs.existsSync(path.resolve(root,h)));
await page.setViewportSize({width:1280,height:900});await page.selectOption('#model','daily_chronos2');await page.selectOption('#split','may');
await page.screenshot({path:path.join(root,'report_preview.png'),fullPage:true});
const result={pass:errors.length===0&&missing.length===0&&layouts.every(x=>!x.overflow),model_count:options.length,interactive_image_checks:checks,layouts,errors,missing_local_links:missing};
fs.writeFileSync(path.join(root,'browser_qa.json'),JSON.stringify(result,null,2));
await browser.close();console.log(JSON.stringify(result));if(!result.pass)process.exitCode=1;
