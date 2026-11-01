"""Render paired, selectable layer maps and the expert-readiness detail."""
from html import escape


def flow_map(external):
    key = "external" if external else "owned"
    ink = "#72dfb2" if external else "#7abaff"
    parts = [f'<svg viewBox="0 0 440 470" role="group" aria-label="{key} layer dataflow">',
             f'<defs><marker id="arrow-{key}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 10 5 0 10" fill="{ink}"/></marker></defs>']
    def edge(path):
        parts.append(f'<path d="{path}" fill="none" stroke="{ink}" stroke-width="2" marker-end="url(#arrow-{key})"/>')
    def box(step, x, y, w, label):
        parts.append(f'<g class="station" data-step="{step}" role="button" tabindex="0" aria-label="Step {step+1}, {label}"><rect x="{x}" y="{y}" width="{w}" height="34" rx="6" fill="#172a39" stroke="{ink}"/><text x="{x+w/2}" y="{y+22}" text-anchor="middle">{label}</text></g>')
    parts.append('<path d="M46 30 V425 M46 170 H132 M46 425 H132" fill="none" stroke="#f5bd71" stroke-width="3"/><text x="15" y="278" transform="rotate(-90 15 278)" fill="#f5bd71">Residual stream · 8192</text>')
    rows = [32,89,146,208,267,326,408]
    for a,b in zip(rows,rows[1:]):
        if a in (146,208) and not external: continue
        edge(f'M245 {a+34} V{b-3}')
    box(0,132,32,226,'QKV projection')
    box(1,132,89,226,'Paged FP8 attention')
    box(2,132,146,226,'O projection → residual mix')
    if external:
        box(3,120,208,250,'Pooler + router projection')
    else:
        for x in (159,332):
            edge(f'M245 180 V191 H{x} V205')
            edge(f'M{x} 242 V253 H245 V264')
        box(3,99,208,120,'Pooler')
        box(3,269,208,126,'Router')
    box(4,132,267,226,'Route + expert input norm')
    box(5,105,326,280,'Expert ranges · per-expert flags' if external else 'Expert units · grid completion')
    box(6,110,408,270,'Unpool → final residual mix')
    parts.append('<text x="245" y="465" text-anchor="middle">Next layer · T × 8192</text></svg>')
    return ''.join(parts)


def walkthrough(ours, theirs):
    captions = ''.join(f'<article data-caption="{i}"'+(' hidden' if i else '')+f'><strong>{i+1} / 7 &nbsp; {escape(a[0])}</strong><div class="grid"><p><b class="ownlabel">Ours.</b> {escape(a[1])}</p><p><b class="extlabel">Keell0renz.</b> {escape(b[1])}</p></div></article>' for i,(a,b) in enumerate(zip(ours,theirs)))
    return '''
<style>
.mapfigure{margin:24px 0 36px;padding:clamp(10px,2vw,26px);background:#131f2a;border:1px solid var(--line);border-radius:12px}.mapfigure svg{display:block;width:100%;max-width:480px;margin:auto}.mapfigure svg text{font:15px system-ui;fill:#edf1f6}.mapfigure .station{cursor:pointer}.mapfigure .station.active rect,.mapfigure .station:focus rect{stroke:#ffcf82;stroke-width:3;fill:#354043}.mapfigure h3{text-align:center;margin:0}.mapfigure article p{margin:.4em 0}.mapfigure figcaption{border-top:1px solid var(--line);padding-top:16px}.mapcontrols{display:flex;align-items:center;gap:10px;margin-top:10px}.mapcontrols small{margin-left:auto}.mapfigure [hidden]{display:none}@media(max-width:760px){.mapfigure .grid{gap:16px}}
</style>
<figure class="mapfigure" id="layerwalk"><div class="grid"><div><h3 class="ownlabel">yelgabo · owned incumbent</h3>'''+flow_map(False)+'''</div><div><h3 class="extlabel">keell0renz · public comparator</h3>'''+flow_map(True)+'''</div></div><figcaption><div aria-live="polite">'''+captions+'''</div><div class="mapcontrols"><button type="button" id="stepback">← Previous</button><button type="button" id="stepnext">Next →</button><small>Choose a box or use ← → inside the figure.</small></div></figcaption></figure>
<script>
(()=>{const root=document.getElementById('layerwalk'), stations=[...root.querySelectorAll('[data-step]')], captions=[...root.querySelectorAll('[data-caption]')];let step=0;
function select(n){step=Math.max(0,Math.min(6,n));for(const el of stations){const active=Number(el.dataset.step)===step;el.classList.toggle('active',active);el.setAttribute('aria-pressed',String(active))}captions.forEach((el,i)=>el.hidden=i!==step);root.querySelector('#stepback').disabled=step===0;root.querySelector('#stepnext').disabled=step===6}
stations.forEach(el=>{el.onclick=()=>select(Number(el.dataset.step));el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();el.onclick()}}});root.querySelector('#stepback').onclick=()=>select(step-1);root.querySelector('#stepnext').onclick=()=>select(step+1);root.onkeydown=e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();select(step+(e.key==='ArrowLeft'?-1:1))}};select(0)})();
</script>'''


def completion_detail():
    panels=[]
    for shared in (True,False):
        color='#7abaff' if shared else '#72dfb2'
        paths=['M85 51 V80 H190 V97','M190 51 V97','M295 51 V80 H190'] if shared else ['M85 51 V80 H140 V97','M190 51 V80 H140','M295 51 V97']
        lines=''.join(f'<path d="{p}" fill="none" stroke="{color}" stroke-width="2"/>' for p in paths)
        def tile(x,y,w,label):
            return f'<rect x="{x}" y="{y}" width="{w}" height="32" rx="5" fill="#172a39" stroke="{color}"/><text x="{x+w/2}" y="{y+21}" text-anchor="middle">{label}</text>'
        tiles=''.join(tile(x,19,90,t) for x,t in [(40,'Expert A'),(145,'Expert B'),(250,'Expert C')])
        if shared:
            tiles+=tile(115,97,150,'Grid barrier')+tile(115,159,150,'Finish token') 
            lines+='<path d="M190 129 V159" stroke="#7abaff" stroke-width="2"/>'
        else:
            tiles+=tile(60,97,160,'A + B ready')+tile(240,97,112,'C ready')+tile(60,159,160,'Finish token')
            lines+='<path d="M140 129 V159" stroke="#72dfb2" stroke-width="2"/>'
        name='Owned · all blocks join' if shared else 'External · routed experts join'
        panels.append(f'<div><h3>{name}</h3><svg viewBox="0 0 380 205" role="img" aria-label="{name}">{lines}{tiles}</svg></div>')
    return '<figure class="mapfigure"><div class="grid">'+''.join(panels)+'</div><figcaption>Follow one token whose local routes use A and B. In the grid-barrier design, unfinished work for C can delay the final pass. With per-expert completion, the token can proceed once A and B publish readiness. Lines show dependencies; elapsed time and expert counts are illustrative.</figcaption></figure>'
