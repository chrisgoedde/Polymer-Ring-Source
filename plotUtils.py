import matplotlib.pyplot as plt
import os
import numpy as np

from matplotlib.patches import Wedge
from matplotlib.patches import Rectangle

import sys

if sys.platform == 'Darwin':
	import applescript

# Use AppleScript to write a spotlight comment so we can more easily
# search for pictures. Only used on MacOS.

def addFileComment(fileName, theComment):

    addComment = applescript.AppleScript('''

        on run {arg1, arg2}
            tell application "Finder" to set comment of (POSIX file arg1 as alias) to arg2 as Unicode text
            return
        end run

    ''')

    addComment.run(fileName, theComment)

def savePlot(pathName, fileName, type = 'pdf', dpiArg = None, finderComment = '', closeFigure = False):

    if not os.path.exists(pathName):
        os.makedirs(pathName)

    print('--> Saving ' + fileName + '.' + type + ' to ' + pathName)
    plt.savefig(pathName + '/' + fileName + '.' + type, transparent = True, dpi = dpiArg)

    if (sys.platform == 'Darwin') and (finderComment != ''):

        addFileComment(pathName + '/' + fileName + '.' + type, finderComment)

    if closeFigure:

        plt.close(plt.gcf())

def makeFigure(figsize = (8, 6), aspect = (), maxSize = (), polar = False, stacked = False, textSize = 10):

    # Note that the stacked keyword is ignored if we set the aspect ratio ...

    f = plt.figure(figsize = figsize)
    plt.rc('font', size=textSize)

    # We use maxSize to set the maximum size of the axes in the x and y directions

    if maxSize != ():

        maxX = maxSize[0]
        maxY = maxSize[1]

    else:

        maxX = 0.7
        maxY = 0.7

    # If no aspect ratio given, use the defaults, as determined by testing.

    if aspect == ():

        xLen = maxX

        if stacked:

            yLen = 0.25 + 0.025
            yOffset = 0.5 - 0.025

        else:

            yLen = maxY

        x0 = (1 - xLen) / 2
        y0 = (1 - yLen) / 2

    else:

        if aspect[0]/figsize[0] >= aspect[1]/figsize[1]:

            xLen = maxX
            yLen = aspect[1] * figsize[0] * xLen / (aspect[0] * figsize[1])

        else:

            yLen = maxY
            xLen = aspect[0] * figsize[1] * yLen / (aspect[1] * figsize[0])

        x0 = (1 - xLen)/2
        y0 = (1 - yLen)/2

    if stacked:

        a1 = f.add_axes([ x0, y0, xLen, yLen ], polar=polar)
        a2 = f.add_axes([ x0, y0 + yOffset, xLen, yLen ], polar=polar)

        ax = (a1, a2)

    else:

        ax = f.add_axes([ x0, y0, xLen, yLen ], polar=polar)

    return f, ax

def setAspectRatio(ratio, loglog = False, axis = []):
    '''Set a fixed aspect ratio on matplotlib plots regardless of axis units'''

    if axis == []:

        axis = plt.gca()

    xvals, yvals = axis.axes.get_xlim(), axis.axes.get_ylim()

    if type == 'loglog':

        xrange = np.log(xvals[1]) - np.log(xvals[0])
        yrange = np.log(yvals[1]) - np.log(yvals[0])

    else:

        xrange = xvals[1] - xvals[0]
        yrange = yvals[1] - yvals[0]

    axis.set_aspect(ratio*(xrange/yrange), adjustable='box')

def drawSquare(ax, xMin, xMax, yMin, yMax, theColor = (0.0, 0.5, 0.5, 0.5)):

    w = xMax - xMin
    h = yMax - yMin

    rect = Rectangle((xMin, yMin), w, h, color = theColor)
    ax.add_artist(rect)

##### Stackoverflow code for drawing a wedge in python #####

def drawWedge(ax, phiMin, phiMax, rMin, rMax, theColor = (1, 1, 1, 1), lined = False):

    # Translate the angle to our rotated coordinates

    thetaMax = 2*np.pi/3-phiMin
    thetaMin = 2*np.pi/3-phiMax

    ##compute the corner points of the wedge:
    axtmin = 0

    rs = np.array([rMin,  rMax,  rMin, rMax, rMin, rMax])
    ts = np.array([axtmin, axtmin, thetaMin, thetaMin, thetaMax, thetaMax])

    ##from https://matplotlib.org/users/transforms_tutorial.html
    trans = ax.transData + ax.transAxes.inverted()

    ##convert to figure cordinates, for a starter
    xax, yax = trans.transform([(t,r) for t,r in zip(ts, rs)]).T

    ##compute the angles of the wedge:
    thetaStart = np.rad2deg(angle(*np.array((xax[[0,1,2,3]],yax[[0,1,2,3]])).T))
    thetaEnd = np.rad2deg(angle(*np.array((xax[[0,1,4,5]],yax[[0,1,4,5]])).T))

    ##the center is where the two wedge sides cross (maybe outside the axes)
    center=seq_intersect(*np.array((xax[[2,3,4,5]],yax[[2,3,4,5]])).T)

    ##compute the inner and outer radii of the wedge:
    rInner = np.sqrt((xax[1]-center[0])**2+(yax[1]-center[1])**2)
    rOuter = np.max([np.sqrt((xax[2]-center[0])**2+(yax[2]-center[1])**2), 1e-6])

    if lined:

        wedge = Wedge(center, rOuter, thetaStart, thetaEnd,
                  width = rOuter-rInner,
                  #0.6,thetaStart,thetaEnd,0.3,
                  transform = ax.transAxes, linestyle='-', lw=2,
                  fc = theColor, ec = (0.5, 0, 0.5, 1))

    else:

        wedge = Wedge(center, rOuter, thetaStart, thetaEnd,
                  width = rOuter-rInner,
                  #0.6,thetaStart,thetaEnd,0.3,
                  transform = ax.transAxes, # linestyle='--', lw=3,
                  color = theColor)

    ax.add_artist(wedge)

    return wedge

def perp(a):
    ##from https://stackoverflow.com/a/3252222/2454357
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def seq_intersect(a1, a2, b1, b2):
    ##from https://stackoverflow.com/a/3252222/2454357
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1

def angle(a1, a2, b1, b2):
    ##from https://stackoverflow.com/a/16544330/2454357
    x1, y1 = a2-a1
    x2, y2 = b2-b1
    dot = x1*x2 + y1*y2      # dot product between [x1, y1] and [x2, y2]
    det = x1*y2 - y1*x2      # determinant
    return np.arctan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
