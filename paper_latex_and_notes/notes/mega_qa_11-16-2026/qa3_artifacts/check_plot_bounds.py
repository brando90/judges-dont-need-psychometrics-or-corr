"""Inspect figure-level caption/legend bounds without rewriting generated artifacts."""
from pathlib import Path
import runpy,sys,json,io,builtins,contextlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.figure
root=Path.cwd();out=Path(__file__).resolve().parent
report={};real_open=builtins.open

def read_only_open(file,mode='r',*a,**kw):
 if 'w' in mode and str(file).endswith('results/tables.md'):return io.StringIO()
 return real_open(file,mode,*a,**kw)
def inspect(fig,path,*a,**kw):
 fig.canvas.draw();renderer=fig.canvas.get_renderer();width,height=fig.bbox.width,fig.bbox.height
 records=[]
 for t in [*fig.texts,*fig.legends]:
  b=t.get_window_extent(renderer);outside=b.x0<-.5 or b.y0<-.5 or b.x1>width+.5 or b.y1>height+.5
  records.append({'text':t.get_text() if hasattr(t,'get_text') else 'legend','bbox':list(b.bounds),'outside_canvas':bool(outside)})
 report[Path(path).name]={'canvas':[width,height],'figure_text':records}
matplotlib.figure.Figure.savefig=inspect
builtins.open=read_only_open
for n in ['06','07','08']:
 script=next((root/'experiments').glob(n+'_*'))/'scripts/make_report.py'
 sys.path.insert(0,str(script.parent))
 with contextlib.redirect_stdout(io.StringIO()):runpy.run_path(str(script),run_name='__main__')
builtins.open=real_open
(out/'plot_bounds.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:{'canvas':v['canvas'],'outside':[t for t in v['figure_text'] if t['outside_canvas']]} for k,v in report.items()},indent=2))
assert not any(t['outside_canvas'] for v in report.values() for t in v['figure_text'])
