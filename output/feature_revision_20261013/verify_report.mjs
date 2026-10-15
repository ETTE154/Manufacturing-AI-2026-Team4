import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {chromium} from 'file:///C:/Users/jeon/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const root=path.resolve('output/feature_revision_20261013');
const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe'});
try{
 const page=await browser.newPage({viewport:{width:1280,height:900}});const errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto(pathToFileURL(path.join(root,'feature_revision_report.html')).href);
 await page.evaluate(()=>Array.from(document.images).forEach(x=>x.loading='eager'));
 await page.waitForFunction(()=>Array.from(document.images).every(x=>x.complete&&x.naturalWidth),null,{timeout:20000});
 const initial=await page.evaluate(()=>({title:document.title,imageCount:document.images.length,
  brokenImages:Array.from(document.images).filter(x=>!x.naturalWidth).length,
  overflow:document.documentElement.scrollWidth>innerWidth,replacementCharacters:document.body.innerText.includes('\uFFFD')}));
 const dir=path.join(root,'tmp','report_qa');await fs.mkdir(dir,{recursive:true});
 await page.screenshot({path:path.join(dir,'report_top.png')});
 await page.getByRole('heading',{name:'24시간 모델의 재검증 결과',exact:true}).scrollIntoViewIfNeeded();
 await page.screenshot({path:path.join(dir,'report_metrics.png')});
 const values=await page.locator('#prediction-model option').evaluateAll(xs=>xs.map(x=>x.value));const interactions=[];
 for(const value of values){
  await page.locator('#prediction-model').selectOption(value);
  await page.waitForFunction(()=>{const x=document.getElementById('prediction-figure');return x.complete&&x.naturalWidth>0;},null,{timeout:20000});
  interactions.push(await page.evaluate(v=>({value:v,caption:document.getElementById('prediction-caption').textContent,
    naturalWidth:document.getElementById('prediction-figure').naturalWidth,
    svgHref:document.getElementById('prediction-svg').getAttribute('href')}),value));
 }
 await page.getByRole('heading',{name:'모델별 실측과 예측',exact:true}).scrollIntoViewIfNeeded();
 await page.screenshot({path:path.join(dir,'report_gallery.png')});
 await page.setViewportSize({width:768,height:900});
 const compactOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
 const result={...initial,compactOverflow,errors,interactions,all14ModelGraphsLoaded:interactions.length===14&&interactions.every(x=>x.naturalWidth>0),
  screenshots:['tmp/report_qa/report_top.png','tmp/report_qa/report_metrics.png','tmp/report_qa/report_gallery.png']};
 await fs.writeFile(path.join(root,'report_verification.json'),JSON.stringify(result,null,2),'utf8');
 if(errors.length||initial.brokenImages||initial.overflow||compactOverflow||initial.replacementCharacters||!result.all14ModelGraphsLoaded)process.exitCode=1;
 console.log(JSON.stringify({...result,interactions:result.interactions.length}));
}finally{await browser.close();}
