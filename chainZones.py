import math

import numpy as np
from math import floor

# Some constants used when subdividing the zones.
# LR -> lower right
# LL -> lower left
# UR -> upper right
# UL -> upper left
# These are strings, but we will also convert these to base-four integers
# when working with the zones.

LR = '3'
LL = '2'
UL = '1'
UR = '0'


# Create the zones. The zones can either be equal area have an equal number of data
# points.

def makeZones(x, y, shape='Wedge', style='Classic', maxLevel=7, minPoints=100):
    # x and y are the coordinate arrays. They are only used if we are making
    # variable-area zones that have an equal number of points.

    # We will return a dictionary whose key is the level of discretization. For equal-area
    # zones, this will run from 0 to maxLevel. For equal-occupancy zones, the maximum level
    # will be determined dynamically, based on the minPoints value.
    #
    # At each level, there are 4^l zones, numbered from 0 to 4^l - 1. These zone numbers
    # are the next keys in each dictionary. Each of these zones then itself has four
    # keys, 'xmin', 'xmax', 'ymin', 'ymax' that give the coordinates of the boundaries of
    # that zone.

    # For wedge plots, we assume that the angle coordinate, x, runs from 0° to 60°. The
    # radial coordinate, y, is assumed to run from 0 to 1.
    # For other plots, we assume that both the x- and y-coordinates run from 0 to 1.

    # Initialize the zone dictionary.

    zoneDict = {}

    # zoneDict has two top-level keys, to record the shape and style of the zones.

    zoneDict['shape'] = shape
    zoneDict['style'] = style

    # Fill in the values for level 0, which only has one zone, zone 0, which encompasses
    # either the entire region, or the part of the region that contains data.

    zoneDict[0] = {}
    zoneDict[0][0] = {}

    if shape == 'Wedge':

        if style == 'Classic':
            zoneDict[0][0]['ymin'] = 0
        elif style == 'noSector':
            zoneDict[0][0]['ymin'] = 0.25
        elif style == 'Dynamic':
            zoneDict[0][0]['ymin'] = findMinima(y)
        else:
            zoneDict[0][0]['ymin'] = 0

        zoneDict[0][0]['ymax'] = 1
        zoneDict[0][0]['xmin'] = 0
        zoneDict[0][0]['xmax'] = np.pi / 3

    else:

        zoneDict[0][0]['ymin'] = 0
        zoneDict[0][0]['ymax'] = 1
        zoneDict[0][0]['xmin'] = 0
        zoneDict[0][0]['xmax'] = 1

    divideZones(x, y, 0, zoneDict, shape=shape, style=style, maxLevel=maxLevel, \
                minPoints=minPoints)

    return zoneDict


def divideZones(x, y, level, zoneDict, shape='Wedge', style='Classic', maxLevel=7, \
                minPoints=100):
    # Note, typo found in the parameter default value for style (was "Classics" should be "Classic")

    newLevel = level + 1

    # We check that the maximum passed level has been reached
    if (style == 'Classic') or (style == 'noSector'):
        if newLevel > maxLevel:
            return
    # We check that, given minPoints, it is possible to partition at the newLevel
    else:
        if newLevel > compMaxLevel(x.shape[0], minPoints):
            return

    # Make the dictionary for the zones at the new level.
    if style == 'Classic' or style == 'noSector':
        zoneDict[newLevel] = {}
        for z in range(4 ** newLevel):
            zoneDict[newLevel][z] = {}

        # Go through the existing zones, dividing each one up into four pieces.

        for z in range(4 ** level):

            if (style == 'Classic') and (zoneDict[level][z]['ymin'] == 0):

                divideSector(zoneDict[level][z], intToKey(z, level), newLevel, zoneDict)

            elif (style == 'Classic') and (zoneDict[level][z]['xmin'] == 0) \
                    and (zoneDict[level][z]['xmax'] == np.pi / 3):

                divideAnnulus(zoneDict[level][z], intToKey(z, level), newLevel, zoneDict)

            else:

                if shape == 'Wedge':

                    yMid = np.sqrt((zoneDict[level][z]['ymin'] ** 2 + zoneDict[level][z]['ymax'] ** 2) / 2)

                else:

                    yMid = (zoneDict[level][z]['ymin'] + zoneDict[level][z]['ymax'] ** 2) / 2

                divideZoneEqualArea(zoneDict[level][z], yMid, intToKey(z, level), newLevel, zoneDict)
    else:
        zoneDict[newLevel] = {}
        pastZones = zoneDict[level]
        keys = list(pastZones.keys())
        keys.sort()
        # To  partition the past zones, we need iterate over them and use their bounds to
        # help define the new zones
        for z in keys:
            divideAnnulusDynamic(x, y, zoneDict[level][z], intToKey(z, level), newLevel, zoneDict, minPoints)

    divideZones(x, y, newLevel, zoneDict, shape=shape, style=style, maxLevel=maxLevel, minPoints=minPoints)


# Divide a sector up into four pieces. This is only used if shape == 'Wedge' and
# style == 'Classic'.


def divideAnnulusDynamic(x, y, bounds, key, newLevel, zoneDict, minPoints):
    """New partitioning method"""
    # First step is to assemble the data into an array for easier indexing
    plane = np.array([x, y])

    # Sort the plane using the angular data
    xS = plane[:, plane[0, :].argsort()]

    # Create an array "mask" that picks out the data points that are within the
    # current angular bounds
    xLess = xS[0, :] <= bounds['xmax']
    xGreat = xS[0, :] >= bounds['xmin']
    xMask = np.logical_and(xLess, xGreat)
    xPlane = xS[:, xMask]

    # Redo the "masking" process with respect to the radial data
    yS = xPlane[:, xPlane[1, :].argsort()]
    yLess = yS[1, :] <= bounds['ymax']
    yGreat = yS[1, :] >= bounds['ymin']
    yMask = np.logical_and(yLess, yGreat)
    plane = yS[:, yMask]

    # Redefine the data array with the picked out points, sorted
    # in terms of the angular data
    plane = plane[:, plane[0, :].argsort()]

    # Error Flag
    if (plane.shape[1] / 4) < minPoints:
        zoneDict[newLevel][keyToInt(key)] = bounds
        print('Should be Impossible! Partition minima encountered.')

    else:
        # We partition the angular dimension first

        # Check if the number of data points is even or not.
        # If not, we need to reward the "sparser" region the extra index.
        # Sparsity is simply defined by which area has a larger range of dimension values.
        if plane.shape[1] % 2 != 0:
            desc = compareRanges(plane[0, :])
            if desc == 0:
                splitIdx = (plane.shape[1] // 2) + 1
                xidx = plane[0, (plane.shape[1] // 2) + 1]
            else:
                splitIdx = (plane.shape[1] // 2) - 1
                xidx = plane[0, (plane.shape[1] // 2) - 1]
        else:
            # If we have an even number of points, just split the points
            # in half and use the value at the half-way mark as a boundary
            # value.

            splitIdx = plane.shape[1] // 2
            xidx = plane[0, plane.shape[1] // 2]

        # For splitting in the radial component, we need two subroutines, one fore each half of the angular split
        # Thus, we need to use two separate plane sets

        # Left side of the angular partition
        left = plane[:, 0:splitIdx]
        left = left[:, left[1, :].argsort()]
        # Right side of the angular partition
        right = plane[:, splitIdx:]
        right = right[:, right[1, :].argsort()]

        if left.shape[1] % 2 != 0:
            desc = compareRanges(left[1, :])
            if desc == 0:
                yidxL = left[1, (left.shape[1] // 2) + 1]
            else:
                yidxL = left[1, (left.shape[1] // 2) - 1]
        else:
            yidxL = left[1, left.shape[1] // 2]

        if right.shape[1] % 2 != 0:
            desc = compareRanges(right[1, :])
            if desc == 0:
                yidxR = right[1, (right.shape[1] // 2) + 1]
            else:
                yidxR = right[1, (right.shape[1] // 2) - 1]
        else:
            yidxR = right[1, right.shape[1] // 2]

        # The zone instantiation routine is borrowed from the other methods.
        # The only thing that is changed is that we now define our bounds
        # using the computed boundary values in the section above in some
        # cases boundaries are retained from the passed zone

        zNumber = keyToInt(key + LR)

        zoneDict[newLevel][zNumber] = {}
        zoneDict[newLevel][zNumber]['ymin'] = bounds['ymin']
        zoneDict[newLevel][zNumber]['ymax'] = yidxR
        zoneDict[newLevel][zNumber]['xmax'] = bounds['xmax']
        zoneDict[newLevel][zNumber]['xmin'] = xidx

        zNumber = keyToInt(key + LL)

        zoneDict[newLevel][zNumber] = {}
        zoneDict[newLevel][zNumber]['ymin'] = bounds['ymin']
        zoneDict[newLevel][zNumber]['ymax'] = yidxL
        zoneDict[newLevel][zNumber]['xmax'] = xidx
        zoneDict[newLevel][zNumber]['xmin'] = bounds['xmin']

        zNumber = keyToInt(key + UL)

        zoneDict[newLevel][zNumber] = {}
        zoneDict[newLevel][zNumber]['ymin'] = yidxL
        zoneDict[newLevel][zNumber]['ymax'] = bounds['ymax']
        zoneDict[newLevel][zNumber]['xmax'] = xidx
        zoneDict[newLevel][zNumber]['xmin'] = bounds['xmin']

        zNumber = keyToInt(key + UR)

        zoneDict[newLevel][zNumber] = {}
        zoneDict[newLevel][zNumber]['ymin'] = yidxR
        zoneDict[newLevel][zNumber]['ymax'] = bounds['ymax']
        zoneDict[newLevel][zNumber]['xmax'] = bounds['xmax']
        zoneDict[newLevel][zNumber]['xmin'] = xidx


def divideSector(bounds, key, newLevel, zoneDict):
    # We're going to divide the sector up into four zones.

    # One zone will be a new sector with half the radius of the original sector.

    zNumber = keyToInt(key + LR)

    zoneDict[newLevel][zNumber]['ymin'] = 0
    zoneDict[newLevel][zNumber]['ymax'] = bounds['ymax'] / 2
    zoneDict[newLevel][zNumber]['xmin'] = 0
    zoneDict[newLevel][zNumber]['xmax'] = np.pi / 3

    # One zone will be an annulus that sits just above the new sector and stretches
    # all the way across original sector.

    zNumber = keyToInt(key + LL)

    zoneDict[newLevel][zNumber]['ymin'] = bounds['ymax'] / 2
    zoneDict[newLevel][zNumber]['ymax'] = bounds['ymax'] / np.sqrt(2)
    zoneDict[newLevel][zNumber]['xmin'] = 0
    zoneDict[newLevel][zNumber]['xmax'] = np.pi / 3

    # The last two zones split the upper part of the original sector into left-right
    # halves.

    zNumber = keyToInt(key + UL)

    zoneDict[newLevel][zNumber]['ymin'] = bounds['ymax'] / np.sqrt(2)
    zoneDict[newLevel][zNumber]['ymax'] = bounds['ymax']
    zoneDict[newLevel][zNumber]['xmin'] = bounds['xmin']
    zoneDict[newLevel][zNumber]['xmax'] = bounds['xmax'] / 2

    zNumber = keyToInt(key + UR)

    zoneDict[newLevel][zNumber]['ymin'] = bounds['ymax'] / np.sqrt(2)
    zoneDict[newLevel][zNumber]['ymax'] = bounds['ymax']
    zoneDict[newLevel][zNumber]['xmin'] = bounds['xmax'] / 2
    zoneDict[newLevel][zNumber]['xmax'] = bounds['xmax']


# Divide an annulus up into four side-by-side pieces. This is only used if shape == 'Wedge'
# and style == 'Classic' and we are dividing the annulus directly above the bottom sector.

def divideAnnulus(bounds, key, newLevel, zoneDict):
    # We're going to divide the annulus up into four side-by-side zones, numbered right
    # to left. We don't change the radial values at all.

    zNumber = keyToInt(key + LL)

    zoneDict[newLevel][zNumber]['ymin'] = bounds['ymin']
    zoneDict[newLevel][zNumber]['ymax'] = bounds['ymax']
    zoneDict[newLevel][zNumber]['xmin'] = 0
    zoneDict[newLevel][zNumber]['xmax'] = np.pi / 12

    # One zone will be an annulus that sits just above the new sector and stretches
    # all the way across original sector.

    zNumber = keyToInt(key + LR)

    zoneDict[newLevel][zNumber]['ymin'] = bounds['ymin']
    zoneDict[newLevel][zNumber]['ymax'] = bounds['ymax']
    zoneDict[newLevel][zNumber]['xmin'] = np.pi / 12
    zoneDict[newLevel][zNumber]['xmax'] = np.pi / 6

    zNumber = keyToInt(key + UL)

    zoneDict[newLevel][zNumber]['ymin'] = bounds['ymin']
    zoneDict[newLevel][zNumber]['ymax'] = bounds['ymax']
    zoneDict[newLevel][zNumber]['xmin'] = np.pi / 6
    zoneDict[newLevel][zNumber]['xmax'] = np.pi / 4

    zNumber = keyToInt(key + UR)

    zoneDict[newLevel][zNumber]['ymin'] = bounds['ymin']
    zoneDict[newLevel][zNumber]['ymax'] = bounds['ymax']
    zoneDict[newLevel][zNumber]['xmin'] = np.pi / 4
    zoneDict[newLevel][zNumber]['xmax'] = np.pi / 3


# Divide a zone up into equal areas. This is only called for non-sectors (annuli or
# rectangles). We feed the function the value of yMid so account for the shape
# (annulus or rectangle).

def divideZoneEqualArea(bounds, yMid, key, newLevel, zoneDict):
    # Divide an annulus into four pieces of equal area. The dividing line in the
    # radial direction is at yMid = np.sqrt((yMin**2+yMax**2)/2). The dividing line
    # in the angle is simply the average angle for the old annulus.

    xMid = (bounds['xmin'] + bounds['xmax']) / 2
    yMid = np.sqrt((bounds['ymin'] ** 2 + bounds['ymax'] ** 2) / 2)

    # The two lower pieces run from yMin to yMid.

    zNumber = keyToInt(key + LR)

    zoneDict[newLevel][zNumber]['ymin'] = bounds['ymin']
    zoneDict[newLevel][zNumber]['ymax'] = yMid
    zoneDict[newLevel][zNumber]['xmin'] = xMid
    zoneDict[newLevel][zNumber]['xmax'] = bounds['xmax']

    zNumber = keyToInt(key + LL)

    zoneDict[newLevel][zNumber]['ymin'] = bounds['ymin']
    zoneDict[newLevel][zNumber]['ymax'] = yMid
    zoneDict[newLevel][zNumber]['xmin'] = bounds['xmin']
    zoneDict[newLevel][zNumber]['xmax'] = xMid

    # The two upper pieces from from yMid to yMax.

    zNumber = keyToInt(key + UL)

    zoneDict[newLevel][zNumber]['ymin'] = yMid
    zoneDict[newLevel][zNumber]['ymax'] = bounds['ymax']
    zoneDict[newLevel][zNumber]['xmin'] = bounds['xmin']
    zoneDict[newLevel][zNumber]['xmax'] = xMid

    zNumber = keyToInt(key + UR)

    zoneDict[newLevel][zNumber]['ymin'] = yMid
    zoneDict[newLevel][zNumber]['ymax'] = bounds['ymax']
    zoneDict[newLevel][zNumber]['xmin'] = xMid
    zoneDict[newLevel][zNumber]['xmax'] = bounds['xmax']


# Assign a zone to each data point, based on its x-y coordinate. We return a
# list with multiple zone assignments, from 4 zones at the minimum to 4**levels
# zones at a maximum.

def assignZones(x, y, zoneDict):
    # x and y are the coordinates for the zones. The maxLevel argument tells us the
    # maximum number of levels to use. zoneDict is a dictionary that holds the coordinates
    # of the zones at every level.  We will return a list that holds the zone
    # assignments for every discretization level from 1 to maxLevel.

    # The first entry of zoneList is an empty list. Each subsequent entry is an
    # time series array whose points have been assigned to zones at discreteness
    # level 4^l. So zoneList[1] is an array where each data point has been assigned
    # to 1 of 4 zones (numbered 0 to 3), zoneList[2] is an array where each data point
    # has been assigned to 1 of 16 zones (numbered 0 to 15), etc.

    zoneList = [[]]

    # Get the value of maxLevel from the zone dictionary.

    maxLevel = max([z for z in zoneDict.keys() if isinstance(z, int)])
    print('Max Level Is: ' + str(maxLevel))

    # Find the number of frames in the data.

    numFrames, = x.shape

    # zone is a temporary array to hold the zone assignments for each data point. It
    # has maxLevel rows and numFrames columns. The first row is the zone assignment for
    # each data point when discretization level == 1 (4 zones). The second row is the
    # zone assignment for each data point when the level == 2 (16 zones). Etc.

    zone = np.zeros((maxLevel, numFrames))

    # March through the frames and assign each data point to its zone.

    for t in range(numFrames):
        zone[:, t] = findZone(x[t], y[t], maxLevel, zoneDict)

    # Append the arrays of zone data to zoneList.

    for l in range(maxLevel):
        zoneList.append(np.copy(zone[l, :]))

    return zoneList


# Find the proper zone for a given location in the (x,y) plane.

def findZone(x, y, maxLevel, zoneDict):
    # To be safe, we don't make any assumptions about the ordering or positioning of the
    # zones. We just look at which zone holds the given (x,y) location, based on the
    # bounds stored in zoneDict.
    zone = np.zeros((maxLevel,))

    # Make a list of the base keys.

    keyList = [UL, UR, LR, LL]

    # Start with an empty key. We will add to this as we go through the levels.

    currentKey = ''

    for l in range(maxLevel):

        level = l + 1

        foundZone = False

        # Look at the four zones at this level, one at a time.

        for k in keyList:

            # Find the zone number that goes with the current key.

            z = keyToInt(currentKey + k)

            # If the data point is in the current zone, save the zone number
            # and break out of the for loop, going on to the next point.

            if (x >= zoneDict[level][z]['xmin']) \
                    and (x <= zoneDict[level][z]['xmax']) \
                    and (y >= zoneDict[level][z]['ymin']) \
                    and (y <= zoneDict[level][z]['ymax']):
                foundZone = True
                zone[l] = z
                break

        if not foundZone:
            print('Impossible! Could not find a zone for data point (x,y) = ' \
                  + '{:.2f}'.format(x) + ',' + '{:.2f}'.format(y) + ')')

        currentKey = currentKey + k

    return zone


def compareRanges(xy):
    """New Function for settling unequal partitions"""

    # What is the current splitting index?
    splitIdx = xy.shape[0] // 2

    # Create left and right partitions of a given data stream
    left = xy[0:splitIdx]
    right = xy[splitIdx:splitIdx * 2]

    # If the left side is more "sparse" (larger range), return 0.
    # Return 1 if the opposite case is true.
    if (left[-1] - left[0]) > (right[-1] - right[0]):
        return 0
    else:
        return 1


def compMaxLevel(numPoints, minPoints):
    """Compute the maximum dividing level for a dynamic partition"""

    return math.floor(math.log(numPoints / minPoints) / math.log(4))


def findMinima(xy):
    """New function for finding the minimum point value in a datastream"""

    minima = np.min(xy)
    return minima


# Convert a zone key (a string consisting of characters '0', '1', '2', and '3') to an
# integer by interpreting the key as a number in base 4.

def keyToInt(key):
    return int(key, 4)


# Convert an integer zone number to a zone key (a string consisting of characters '0',
# '1', '2', and '3') by converting the zone number to a base four number. The number of
# digits in the key is determined by the level argument.

def intToKey(index, level):
    t = np.base_repr(index, base=4)

    while len(t) < level:
        t = '0' + t

    return t


# Find the zones that neighbor (have touching boundaries) with any zone in a given group.

def findNeighboringZones(group, level, zoneDict):
    # Initialize the neighbor list.

    neighbors = []

    # Step through in incoming list of zones.

    for g in group:
        # Find the center of the current zone.

        xCenter = 0.5 * (zoneDict[level][g]['xmin'] + zoneDict[level][g]['xmax'])
        yCenter = 0.5 * (zoneDict[level][g]['ymin'] + zoneDict[level][g]['ymax'])

        xLength = zoneDict[level][g]['xmax'] - zoneDict[level][g]['xmin']
        yLength = zoneDict[level][g]['ymax'] - zoneDict[level][g]['ymin']

        # Now find the four neighbors, being careful not to leave the region.

        if (xCenter + xLength) <= zoneDict[0][0]['xmax']:

            ng = findZone(xCenter + xLength, yCenter, level, zoneDict)[-1]
            if (ng not in group) and (ng not in neighbors):
                neighbors.append(ng)

        if (xCenter - xLength) >= zoneDict[0][0]['xmin']:

            ng = findZone(xCenter - xLength, yCenter, level, zoneDict)[-1]
            if (ng not in group) and (ng not in neighbors):
                neighbors.append(ng)

        if (yCenter + yLength) <= zoneDict[0][0]['ymax']:

            ng = findZone(xCenter, yCenter + yLength, level, zoneDict)[-1]
            if (ng not in group) and (ng not in neighbors):
                neighbors.append(ng)

        if (xCenter - xLength) >= zoneDict[0][0]['ymin']:

            ng = findZone(xCenter, yCenter - yLength, level, zoneDict)[-1]
            if (ng not in group) and (ng not in neighbors):
                neighbors.append(ng)

    return neighbors
