import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {chromium} from 'file:///C:/Users/jeon/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';

const root=path.resolve('output/analysis');
const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe'});
try {
  const page=await browser.newPage({viewport:{width:1280,height:900},deviceScaleFactor:1});
  const errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(pathToFileURL(path.join(root,'project_analysis.html')).href);
  // Below-fold report images are lazy loaded in normal use; activate them for QA.
  await page.evaluate(()=>Array.from(document.images).forEach(x=>x.loading='eager'));
  await page.waitForFunction(()=>Array.from(document.images).every(x=>x.complete),null,{timeout:15000});
  const initial=await page.evaluate(()=>({
    title:document.title,
    headings:Array.from(document.querySelectorAll('h1,h2,h3')).map(x=>x.textContent),
    brokenImages:Array.from(document.images).filter(x=>!x.naturalWidth).length,
    imageCount:document.images.length,
    overflow:document.documentElement.scrollWidth>innerWidth,
    selects:Array.from(document.querySelectorAll('select')).map(x=>({id:x.id,options:x.options.length})),
    pending:document.body.innerText.includes('Pending'),
    replacementCharacters:document.body.innerText.includes('\uFFFD')
  }));
  await fs.mkdir(path.join(root,'tmp','report_qa'),{recursive:true});
  await page.screenshot({path:path.join(root,'tmp','report_qa','report_top.png')});
  let interaction=null;
  if(initial.selects.length){
    const select=page.locator('select').first();
    const before=await page.locator('svg').last().innerHTML();
    const next=await select.locator('option').nth(1).getAttribute('value');
    await select.selectOption(next);
    const after=await page.locator('svg').last().innerHTML();
    interaction={selected:next,svgUpdated:before!==after};
  }
  const finalSection=page.locator('h2').last();
  await finalSection.scrollIntoViewIfNeeded();
  await page.screenshot({path:path.join(root,'tmp','report_qa','report_end.png')});
  await page.setViewportSize({width:768,height:900});
  const compactOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
  const result={...initial,interaction,compactOverflow,errors};
  await fs.writeFile(path.join(root,'report_verification.json'),JSON.stringify(result,null,2),'utf8');
  console.log(JSON.stringify(result));
  if(errors.length||initial.brokenImages||initial.pending||initial.replacementCharacters||initial.overflow||compactOverflow||interaction&&!interaction.svgUpdated)process.exitCode=1;
} finally {await browser.close();}
