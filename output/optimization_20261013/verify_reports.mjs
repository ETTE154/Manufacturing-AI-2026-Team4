import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {chromium} from 'file:///C:/Users/jeon/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';

const root=path.resolve('output/optimization_20261013');
const targets=process.argv.slice(2);
const names=targets.length?targets:['optimization_report.html','anomaly/anomaly_report.html'];
const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe'});
const results={};
try {
  for(const name of names){
    const page=await browser.newPage({viewport:{width:1280,height:900}});
    const errors=[];page.on('pageerror',e=>errors.push(e.message));
    await page.goto(pathToFileURL(path.join(root,name)).href);
    await page.evaluate(()=>Array.from(document.images).forEach(x=>x.loading='eager'));
    await page.waitForFunction(()=>Array.from(document.images).every(x=>x.complete),null,{timeout:20000});
    const initial=await page.evaluate(()=>({title:document.title,imageCount:document.images.length,
      brokenImages:Array.from(document.images).filter(x=>!x.naturalWidth).length,
      overflow:document.documentElement.scrollWidth>innerWidth,
      replacementCharacters:document.body.innerText.includes('\uFFFD'),
      selects:Array.from(document.querySelectorAll('select')).map(x=>({id:x.id,options:x.options.length,values:Array.from(x.options).map(o=>o.value)}))}));
    const qaDir=path.join(root,'tmp','report_qa');await fs.mkdir(qaDir,{recursive:true});
    const stem=path.basename(name,'.html');
    await page.screenshot({path:path.join(qaDir,stem+'_top.png')});
    if(name==='optimization_report.html'){
      await page.getByRole('heading',{name:'최적화 결과와 남은 불확실성',exact:true}).scrollIntoViewIfNeeded();
      await page.screenshot({path:path.join(qaDir,stem+'_metrics.png')});
    }
    const interactions=[];
    for(const info of initial.selects){
      const locator=page.locator('select#'+info.id);
      const values=info.values.includes('ae_synthetic_denoising')?['ae_synthetic_denoising']:
        info.values.includes('candidate_2')?['candidate_2','normal_3']:[info.values[1]];
      for(const value of values){
        const before=await page.locator('body').innerHTML();
        await locator.selectOption(value);
        await page.waitForFunction(()=>Array.from(document.images).every(x=>x.complete&&x.naturalWidth>0),null,{timeout:20000});
        const after=await page.locator('body').innerHTML();
        interactions.push({select:info.id,value,updated:before!==after});
      }
    }
    await page.locator('h2').last().scrollIntoViewIfNeeded();
    await page.screenshot({path:path.join(qaDir,stem+'_end.png')});
    await page.setViewportSize({width:768,height:900});
    const compactOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
    results[name]={...initial,interactions,compactOverflow,errors};
    if(errors.length||initial.brokenImages||initial.overflow||compactOverflow||initial.replacementCharacters||interactions.some(x=>!x.updated))process.exitCode=1;
    await page.close();
  }
  await fs.writeFile(path.join(root,'report_verification.json'),JSON.stringify(results,null,2),'utf8');
  console.log(JSON.stringify(results));
} finally{await browser.close();}
