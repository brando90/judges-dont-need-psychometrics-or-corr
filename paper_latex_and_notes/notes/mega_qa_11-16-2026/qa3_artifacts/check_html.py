"""Render the one-page source with experiment image routes; no publication."""
from pathlib import Path
from playwright.sync_api import sync_playwright
import json
root=Path.cwd();out=Path(__file__).resolve().parent
source=root/'paper_latex_and_notes/notes/correlation_vs_loss_judges_one_pager/judge-objectives-status.html'
images={p.name:p for pattern in ['experiments/06*/*.png','experiments/07*/*.png','experiments/08*/*.png','experiments/archive/13*/*.png'] for p in root.glob(pattern)}
with sync_playwright() as pw:
 browser=pw.chromium.launch(channel='chrome',headless=True)
 page=browser.new_page(viewport={'width':1100,'height':1000},device_scale_factor=1)
 errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 def route(req):
  url=req.request.url
  if url=='https://qa.local/':req.fulfill(status=200,content_type='text/html',body='<!doctype html><html><head><meta charset="utf-8"></head><body>'+source.read_text()+'</body></html>')
  elif url.rsplit('/',1)[-1] in images:req.fulfill(path=str(images[url.rsplit('/',1)[-1]]))
  else:req.abort()
 page.route('https://qa.local/**',route)
 page.goto('https://qa.local/',wait_until='networkidle',timeout=45000)
 report={'viewport': {'width':1100,'height':1000}, 'horizontal_overflow': page.evaluate('document.documentElement.scrollWidth > innerWidth'), 'page_errors':errors,'tex_count':page.locator('.tex').count(),'rendered_math':page.locator('.tex math').count(),'katex_errors':page.locator('.katex-error').all_text_contents(),'images':page.locator('img').evaluate_all('(xs)=>xs.map(x=>({src:x.getAttribute("src"),loaded:x.complete&&x.naturalWidth>0}))')}
 page.screenshot(path=str(out/'html_top.png'))
 for selector,name in [('#how-chosen','html_optimization.png'),('#pearson-appendix','html_pearson.png'),('#rps-appendix','html_rps.png')]:
  page.locator(selector).screenshot(path=str(out/name))
 page.set_viewport_size({'width':390,'height':844})
 page.evaluate('scrollTo(0,0)')
 page.screenshot(path=str(out/'html_mobile.png'))
 report['mobile_horizontal_overflow']=page.evaluate('document.documentElement.scrollWidth > innerWidth')
 report['mobile_scroll_width']=page.evaluate('document.documentElement.scrollWidth')
 report['mobile_overflow_candidates']=page.locator('body *').evaluate_all('els=>els.filter(e=>{let r=e.getBoundingClientRect(); if(r.right<=innerWidth) return false; for(let p=e.parentElement;p&&p.tagName!=="BODY";p=p.parentElement) if(["auto","scroll","hidden"].includes(getComputedStyle(p).overflowX)) return false; return true}).slice(0,30).map(e=>({tag:e.tagName,id:e.id,cls:typeof e.className==="string"?e.className:"",text:e.textContent.slice(0,150),right:e.getBoundingClientRect().right}))')
 (out/'html_check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
 assert not errors and report['tex_count']==report['rendered_math']==45
 assert not report['katex_errors'] and all(i['loaded'] for i in report['images'])
 assert len(report['images'])==8 and not report['horizontal_overflow'] and not report['mobile_horizontal_overflow']
 browser.close()
