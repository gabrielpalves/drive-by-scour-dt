"""Draw the manuscript TTBI schematic as editable vector artwork.

No responses/results are synthesized. Sources: paper1/sections/numerical_model.tex;
scour_MATLAB/D01_DataProcessing.m; core/f25_experiment_contract.py. Geometries are
diagrammatic and share no length scale with the vehicle or layer illustrations.
Run from any directory with Python + NumPy + Matplotlib:
  python plotting/build_paper_model_figure.py               # F25 paper (paper1/)
  python plotting/build_paper_model_figure.py --regression  # four-block draft
"""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, Arc

ROOT = Path(__file__).resolve().parents[1]
NAVY = "#243b53"
TEAL = "#197c80"
ORANGE = "#b86022"
GRAY = "#627485"
PALE = "#eaf3f4"


def line(ax, xs, ys, color=NAVY, lw=1.1, **kwargs):
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", **kwargs)


def text(ax, x, y, value, size=8.2, color=NAVY, **kwargs):
    ax.text(x, y, value, fontsize=size, color=color, **kwargs)


def spring(ax, x, bottom, top, width=.7, color=NAVY, turns=4):
    lead=(top-bottom)*.18
    ys=np.linspace(bottom+lead, top-lead, 2*turns+1)
    xs=x+width*np.array([0]+[(-1)**i for i in range(2*turns-1)]+[0])
    line(ax,[x,x],[bottom,bottom+lead],color)
    line(ax,xs,ys,color)
    line(ax,[x,x],[top-lead,top],color)


def arrow(ax, x1, y1, x2, y2, color=NAVY, **kwargs):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",
                                mutation_scale=8,linewidth=1,color=color,**kwargs))


def curved_pitch(ax,x,y,scale=1,color=TEAL):
    ax.add_patch(Arc((x,y),3.4*scale,2.2*scale,theta1=20,theta2=155,color=color,lw=1.2))
    arrow(ax,x-1.55*scale,y+.48*scale,x-1.1*scale,y+.92*scale,color)


def support(ax, x, base, color):
    spring(ax,x,base-4.3,base,width=.78,color=color,turns=3)
    line(ax,[x-2.6,x+2.6],[base-4.4,base-4.4],GRAY,.8)
    for dx in [-2,-.5,1]:line(ax,[x+dx,x+dx+.9],[base-5.3,base-4.4],GRAY,.6)


def f25_bridge(ax):
    """Panel (c) of the F25 paper: one 2 x 19.95 m bridge and its damage sites."""
    text(ax,1,30.7,"(c) Bridge and damage locations",size=9.2,weight="bold")
    start,end,base,length=12,92,19.5,39.9
    scale=(end-start)/length
    line(ax,[start,end],[base,base],NAVY,2.2)
    for i,x in enumerate([start,start+19.95*scale,end]):
        support(ax,x,base,TEAL if i==1 else GRAY)
    for left,right in [(start,start+19.95*scale),(start+19.95*scale,end)]:
        text(ax,(left+right)/2,base+1.9,"19.95 m",size=8,ha="center")
        line(ax,[left,right],[base+1.2,base+1.2],"#d9e2e8",.6)
    ax.add_patch(Circle((start+19.95*scale,base),.7,facecolor=TEAL,edgecolor="white",lw=.5,zorder=4))
    text(ax,start+19.95*scale-1.6,base-3.1,"$k_v$: scour (5 or 10% loss)",size=7.6,color=TEAL,ha="right")
    # Entrance bearing (the train travels left to right, as in panel a).
    ax.add_patch(Arc((start,base),4.1,3.3,theta1=15,theta2=170,color=ORANGE,lw=1.1))
    text(ax,start-1,base+3.9,"$k_r$: bearing damage",size=7.6,color=ORANGE)
    crack=start+29.85*scale   # 29.70-30.00 m zone, widened for visibility
    ax.add_patch(Rectangle((crack-.75,base-.75),1.5,1.5,facecolor="#c0392b",edgecolor="none",zorder=5))
    line(ax,[crack,crack],[base-1,base-5.6],"#c0392b",.7)
    text(ax,crack,base-7.4,"crack zone, 29.70–30.00 m\n(14 or 22% $EI$ loss)",size=7.6,ha="center",va="top",color="#c0392b")
    arrow(ax,start+2,base+7.2,start+12,base+7.2,TEAL)
    text(ax,start+13,base+7.2,"direction of travel",size=7.4,va="center",color=GRAY)


def regression_bridges(ax):
    """Panel (c) of the four-block regression draft: F40 and L99 geometries."""
    text(ax,1,30.7,"(c) Prospective regression bridge geometries",size=9.2,weight="bold")
    for name,spans,base in [("F40",[20,20],23.9),("L99",[24.9]*4,12.6)]:
        start,end=25,94
        supports=np.linspace(start,end,len(spans)+1)
        text(ax,1,base+.2,name,size=9,weight="bold",va="center")
        line(ax,[start,end],[base,base],NAVY,2.2)
        for i,x in enumerate(supports):
            support(ax,x,base,TEAL if i not in [0,len(supports)-1] else GRAY)
            if i not in [0,len(supports)-1]:
                ax.add_patch(Circle((x,base),.7,facecolor=TEAL,edgecolor="white",lw=.5,zorder=4))
            text(ax,x,base-6.9,str(i+1),size=7.4,ha="center")
        for left,right,length in zip(supports[:-1],supports[1:],spans):
            text(ax,(left+right)/2,base+1.9,f"{length:g} m",size=8,ha="center")
            line(ax,[left,right],[base+1.2,base+1.2],"#d9e2e8",.6)
        # Orange curved symbols indicate M-block abutment rotation springs.
        for x in [start,end]:
            ax.add_patch(Arc((x,base),4.1,3.3,theta1=15,theta2=170,color=ORANGE,lw=1.1))
        text(ax,1,base-3.2,f"{sum(spans):g} m",size=7.8,color=GRAY)
    text(ax,1,2.9,"Teal: scour targets   |   Orange: abutment rotational springs in M blocks",size=7.5)
    text(ax,1,.5,"F25 is separate: 2 × 19.95 m = 39.9 m. All panels are schematic, not to scale.",size=7.4,color=GRAY)


def main(regression=False):
    out = ROOT / ("paper_regression" if regression else "paper1") / "figures"
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8.2,
                         "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})
    fig=plt.figure(figsize=(6.25,7.15),facecolor="white")
    ax=fig.add_axes([.02,.015,.96,.97])
    ax.set_xlim(0,100);ax.set_ylim(0,100);ax.axis("off")
    # The F25 paper names channels by response; the draft keeps payload codes.
    names = ({"CV":"S00","CP":"S05","RV":"S02","RP":"S07","FV":"S01","FP":"S06"} if regression
             else {key:key for key in ("CV","CP","RV","RP","FV","FP")})

    # (a) Consist, leading vehicle, and channel-to-DOF mapping.
    text(ax,1,98,"(a) Train and leading-vehicle response channels",size=9.2,weight="bold")
    for n,x in enumerate([6,23,40,57,74]):
        fill=PALE if n==4 else "#f4f6f8"
        ax.add_patch(Rectangle((x,91.5),14.5,3.6,facecolor=fill,edgecolor=TEAL if n==4 else GRAY,lw=1))
        for wx in [x+2.2,x+11.8]:
            ax.add_patch(Circle((wx,91),.62,facecolor="white",edgecolor=GRAY,lw=.7))
        text(ax,x+7.25,93.3,str(5-n),size=7,ha="center",va="center")
        if n<4:line(ax,[x+14.5,x+17],[93,93],GRAY,.7)
    arrow(ax,77,96.6,91,96.6,TEAL)
    text(ax,70,96.65,"$v$",size=8,va="center")
    text(ax,2,89,"Five planar vehicles; leading vehicle = 1",size=7.6,color=GRAY)
    # Enlarged leading vehicle. Front is on the right, consistent with v.
    ax.add_patch(Rectangle((17,76.6),65,5.6,facecolor=PALE,edgecolor=NAVY,lw=1.25))
    text(ax,49.5,79.3,"Carbody (vehicle 1)",size=9,ha="center",va="center")
    arrow(ax,43,83,43,87,TEAL)
    text(ax,40.5,85.5,names["CV"]+r": $\ddot z_c$",ha="right",size=8.4)
    curved_pitch(ax,54,84.5,1.55)
    text(ax,58,85.5,names["CP"]+r": $\dot\theta_c$",size=8.4)
    for x,vertical,pitch,name in [(31,names["RV"],names["RP"],"Rear bogie"),(68,names["FV"],names["FP"],"Front bogie")]:
        for dx in [-3,3]:spring(ax,x+dx,72.7,76.6,width=.58,color=GRAY,turns=3)
        ax.add_patch(Rectangle((x-10,69.6),20,3.1,facecolor="#f4f6f8",edgecolor=NAVY,lw=1))
        text(ax,x,71.2,name,size=8,ha="center",va="center")
        for dx in [-6,6]:
            spring(ax,x+dx,66.9,69.6,width=.45,color=GRAY,turns=2)
            ax.add_patch(Circle((x+dx,65.7),1.15,facecolor="white",edgecolor=NAVY,lw=1))
        text(ax,x,62.1,vertical+r": $\ddot z_b$   |   "+pitch+r": $\dot\theta_b$",size=8.3,ha="center",color=TEAL)
    line(ax,[13,86],[64.5,64.5],NAVY,1.4)
    text(ax,49.5,59.7,r"$\ddot z$: vertical acceleration   |   $\dot\theta$: pitch rate",size=8,ha="center")
    if regression:
        text(ax,49.5,57.4,"S03/S04: constrained-wheelset acceleration proxies; diagnostic only",size=7.4,ha="center",color=GRAY)
        text(ax,49.5,55.1,"Four constrained wheelsets per vehicle; no independent wheel-motion DOFs",size=7.4,ha="center",color=GRAY)
    else:
        text(ax,49.5,56.6,"Wheelsets follow the rail (no independent DOFs); their accelerations are not used",
             size=7.4,ha="center",color=GRAY)

    # (b) Inventory, not an invented track mass/spring topology.
    line(ax,[1,99],[52.7,52.7],"#d9e2e8",.8)
    text(ax,1,50.7,"(b) Retained track layers and coupling" if regression else "(b) Track model",size=9.2,weight="bold")
    layers=[("Rail beam",47.2), ("Pads",44.3), ("Sleepers",41.4),
            ("Ballast / sub-ballast",38.5), ("Bridge deck",35.6)]
    for i,(label,y) in enumerate(layers):
        ax.add_patch(Rectangle((5,y-1.05),36,2.1,facecolor=PALE if i in [0,4] else "#f4f6f8",edgecolor=GRAY,lw=.7))
        text(ax,23,y,label,size=7.8,ha="center",va="center")
        if i<4:line(ax,[23,23],[y-1.05,y-1.85],GRAY,.8)
    if regression:
        text(ax,47,46.9,"Layer inventory only",size=8.1,weight="bold")
        text(ax,47,43.7,"Inherited on-bridge ballast mass\nis condensed onto deck DOFs.",size=7.9,va="top",linespacing=1.45)
        text(ax,47,37.3,"Adjacent-ballast shear coupling\nis omitted in this model.",size=7.9,va="top",linespacing=1.45)
    else:
        text(ax,47,46.2,"Rail: beam on discrete supports,\none per 0.60 m sleeper bay.",size=7.9,va="top",linespacing=1.45)
        text(ax,47,40.4,"On the bridge, the ballast mass is lumped\nat the deck nodes; there is no shear\ncoupling between ballast blocks.",size=7.9,va="top",linespacing=1.45)

    # (c) Bridge geometry: the F25 bridge, or the two regression geometries.
    line(ax,[1,99],[32.8,32.8],"#d9e2e8",.8)
    if regression:
        regression_bridges(ax)
    else:
        f25_bridge(ax)
    for ext in ("pdf","png","svg"):
        fig.savefig(out/f"ttbi_model.{ext}",dpi=220,facecolor="white")
    plt.close(fig)


if __name__=="__main__":main(regression="--regression" in sys.argv[1:])
