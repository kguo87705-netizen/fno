"""Analytic ellipse area in node-centered control cells. No wave solver changes."""
import numpy as np


def node_cell_edges(length, intervals):
    nodes=np.linspace(0.,length,intervals+1,dtype=np.float64)
    return np.r_[0.,(nodes[:-1]+nodes[1:])/2,length]


def ellipse_rectangle_area(xl,xr,yl,yr,geometry):
    """Integrate vertical ellipse/rectangle overlap after all branch crossings."""
    xc,yc,a,b=[geometry[k] for k in ['x','y','a','b']]
    left=max(xl,xc-a); right=min(xr,xc+a)
    if left>=right or yr<=yc-b or yl>=yc+b: return 0.
    knots=[left,right]
    for height in [yl,yr]:
        v=(height-yc)/b
        if abs(v)<1:
            rad=a*np.sqrt(1-v*v)
            for t in [xc-rad,xc+rad]:
                if left<t<right: knots.append(t)
    knots=sorted(set(knots))
    def primitive(x):
        u=np.clip((x-xc)/a,-1.,1.)
        return a*.5*(u*np.sqrt(max(0.,1-u*u))+np.arcsin(u))
    total=0.
    for l,r in zip(knots[:-1],knots[1:]):
        mid=(l+r)/2; root=np.sqrt(max(0.,1-((mid-xc)/a)**2))
        top=yc+b*root; bottom=yc-b*root
        if min(yr,top)<=max(yl,bottom): continue
        uc,us=(yr,0.) if yr<top else (yc,b)
        lc,ls=(yl,0.) if yl>bottom else (yc,-b)
        total+=(uc-lc)*(r-l)+(us-ls)*(primitive(r)-primitive(l))
    return total


def ellipse_fraction(geometry,nx=255,ny=127,lx=.1,ly=.05):
    """Return float64 occupied fraction and control-cell measures.

    This is a geometry representation, not a validated effective acoustic medium.
    """
    ex=node_cell_edges(lx,nx); ey=node_cell_edges(ly,ny)
    xc,yc,a,b=[geometry[k] for k in ['x','y','a','b']]
    farx=np.maximum(np.abs(ex[:-1]-xc),np.abs(ex[1:]-xc))/a
    fary=np.maximum(np.abs(ey[:-1]-yc),np.abs(ey[1:]-yc))/b
    nearx=np.maximum(np.maximum(ex[:-1]-xc,xc-ex[1:]),0.)/a
    neary=np.maximum(np.maximum(ey[:-1]-yc,yc-ey[1:]),0.)/b
    inside=farx[:,None]**2+fary[None,:]**2<=1
    outside=nearx[:,None]**2+neary[None,:]**2>=1
    fraction=inside.astype(np.float64)
    cell_area=np.diff(ex)[:,None]*np.diff(ey)[None,:]
    for i,j in zip(*np.where(~inside & ~outside)):
        fraction[i,j]=ellipse_rectangle_area(ex[i],ex[i+1],ey[j],ey[j+1],geometry)/cell_area[i,j]
    if fraction.min() < -1e-9 or fraction.max()>1+1e-9:
        raise ArithmeticError('Area integration outside probability bounds')
    return np.clip(fraction,0,1),cell_area


def aggregate_fine_area(fine_fraction,fine_area,factor,nx,ny):
    """Odd refinement makes node-centered control-cell boundaries nested."""
    if factor%2!=1: raise ValueError('Use an odd refinement factor')
    startsx=np.maximum(0,factor*np.arange(nx+1)-factor//2)
    startsy=np.maximum(0,factor*np.arange(ny+1)-factor//2)
    return np.add.reduceat(np.add.reduceat(fine_fraction*fine_area,startsx,axis=0),startsy,axis=1)
