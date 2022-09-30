import numpy as np
import time as tP
import copy
import chainZones as cZ


# Assign the zones to groups.

def groupZones(zoneList, zoneDict, time, theLevel, method='Trans', \
               probCutoff=0.98, theShape='Wedge', minGroupSize=20):
    # zoneList is a list that holds the numpy arrays for the zone
    #	  of each data point. The keys are the run numbers for each of
    #	  the data sets in the dictionary.
    # zoneDict is a dictionary that holds the zones themselves, including the
    #     bounds of each zone. The first key is the discreteness level,
    #     the second key is the zone number, and the third key is one of
    #     'xmin', 'xmax', 'ymin', 'ymax'.
    # time is a numpy array holding the time of each data point.
    # theLevel is the number of levels we are using for the grouping;
    #	  there are 4^(theLevel) zones total.
    # method determines which transition matrix to use for the grouping.
    # probCutoff determines when the algorithm stops.

    # We're going to keep track of how long the algorithm runs so we can
    # print updates on the progress.

    start = tP.time()

    # We're going to store everything a dictionary; this is our return object.

    topDict = {}

    # We store the input time and zone arrays as elements in the dictionary.

    topDict['time'] = time
    topDict['zones'] = zoneList

    # Everything else is a dictionary, with the key being the number of groups.
    # By this we mean a grouping into N groups can be found in topDict['string'][N].

    # This first dictionary lists the group assigned to each data point.

    topDict['groups'] = {}

    # These are the matrices used in finding the groups.
    # The elements of jumpMatrix are the number of jumps from group A to group B.
    # The elements of transMatrix are the probabilities of jumping from A to B.
    # The elements of balMatrix are the flux from group A to group B.
    # The elements of netFluxMatrix combine the probability of jumps from A to B and B to A.

    topDict['jumpMatrix'] = {}
    topDict['transMatrix'] = {}
    topDict['balMatrix'] = {}
    topDict['netFluxMatrix'] = {}
    topDict['netFlux'] = {}

    # The colors assigned to each group.

    topDict['colors'] = {}

    # The 'groupList' dictionary holds a list of groups, where each element
    # of the list is a flat list of zones. It will start with one zone per group.
    # The 'groupHierarchy' dictionary holds the same information, but in a
    # hierarchical manner, preserving the history of the groupings. The
    # 'groupSizeList' dictionary holds the number of of data points in each group.

    topDict['groupList'] = {}
    topDict['groupHierarchy'] = {}
    topDict['groupSizeList'] = {}

    # The list of currently active groups.

    topDict['currentGroups'] = {}

    # We start with an empty list of colors for the groups.

    colors = []

    # Create a list of all the zones with at least one data point.
    # We start with each zone being its own group, so groupList is
    # a list of lists, each with a single element. groupSizeList
    # has the same number of elements, each of which is the number
    # of data points in each group.

    # groupMap is a dictionary whose keys are the zone numbers and
    #	  whose values are the group that the zone belongs to.
    # currentGroups is a list of currently active groups.

    (groupList, groupHierarchy, groupSizeList, groupMap, currentGroups) \
        = newGroupList(zoneList)

    # The initial number of groups, equal to the number of zones
    # with at least one data point.

    print('Starting grouping with ' + str(len(groupList)) + ' groups')

    # We make the jump matrix, which tracks how the data points move
    # from zone to zone (and therefore from group to group).
    # The transition matrix is simply the jump matrix normalized
    # by the size of each group to yield a transition probability
    # from group to group. The balance matrix is the flux from group to
    # group; the name comes from "detailed balance". Finally, the netFluxMatrix
    # is the square root of the transition matrix times its transpose.

    (jumpMatrix, transMatrix, balMatrix, netFluxMatrix) \
        = newMatrices(groupList, groupSizeList, groupMap, zoneList)

    print('Found the matrices after ' + '{:.1f}'.format((tP.time() - start) / 60) \
          + ' minutes')

    if method == 'Bal':

        print('Using balance matrix for combining groups')

    elif method == 'Con':

        print('Using connectivity matrix for combining groups')

    else:

        print('Using transition matrix for combining groups')

    # Print out some statistics about the initial groups.

    printGroupStatistics(groupSizeList, currentGroups, transMatrix)

    # Combine as many of the groups that are smaller than minGroupSize as we can.
    # We only combine groups that are contiguous.

    eliminateSmallGroups(minGroupSize, groupList, groupHierarchy, \
                         groupSizeList, groupMap, currentGroups, \
                         jumpMatrix, transMatrix, balMatrix, netFluxMatrix, \
                         theLevel, zoneDict)

    # We are going to keep combining groups until the smallest diagonal
    # element of the transition matrix is less than probCutoff.

    diagProb = np.diagonal(transMatrix)[currentGroups]
    minSelfProb = np.min(diagProb)

    # I have substituted, considered for currentGroups. The instantiation
    # and updating routines for currentGroups do not account for isolated
    # groups.
    considered = copy.copy(currentGroups)

    # We'll stop combining groups when the smallest self-transition probability
    # is below the cutoff.

    while minSelfProb < probCutoff:

        # We combine two groups based on the transition probabilities.
        # We will also modify the jump matrix to take our new combined
        # groups into account, so its size is reduced by one row and
        # one column.

        if method == 'Bal':

            # Use the detailed balance matrix for combining zones.

            g1, g2 = getSimilarGroups(balMatrix, transMatrix, currentGroups, \
                                      probCutoff=probCutoff)

        else:

            # Use the transition probabilities.

            g1, g2 = getSimilarGroups(transMatrix, transMatrix, currentGroups, \
                                      probCutoff=probCutoff)
        # If we were able to find two distinct grouping candidates
        if g1 != g2:
            # Combine the candidates
            combineGroups(g1, g2, groupList, groupHierarchy, groupSizeList, \
                          groupMap, currentGroups, jumpMatrix, transMatrix, \
                          balMatrix, netFluxMatrix)
            # Recompute the diagonal transition probabilities
            diagProb = np.diagonal(transMatrix)
            # Remove g1 from the considered index mask
            considered.pop(considered.index(g1))
            # Recompute the minimum self transition probability
            minSelfProb = np.min(diagProb[considered])

        else:
            # Remove g1 from consideration, it does not have any
            # symmetric transition qualities with the other groups. It is isolated.
            considered.pop(considered.index(g1))
            # Recompute the diagonal transition probabilities
            diagProb = np.diagonal(transMatrix)
            # Recompute the minimum self transition probability
            minSelfProb = np.min(diagProb[considered])

        end = tP.time()
        if (end - start) > 180:
            print('Still working with ' + str(len(currentGroups)) \
                  + ' groups after ' + '{:.1f}'.format((end - start) / 60) \
                  + ' minutes')

            start = end

        # Once the number of groups becomes small enough, or if all the
        # groups have a self-transition probability above the cutoff,
        # we will save them in the group dictionary.

        if len(currentGroups) < 10 or minSelfProb >= probCutoff:

            # Assign colors to the groups. The returned list has
            # len(groupList) items, of which only len(currentGroups)
            # are relevant.

            if colors == []:
                colors = assignColorstoGroups(groupList, currentGroups, theLevel, zoneDict)

            addGrouptoDict(topDict, zoneList, time, groupList, groupHierarchy, \
                           groupSizeList, currentGroups, jumpMatrix, \
                           transMatrix, balMatrix, netFluxMatrix, \
                           theLevel, colors)

    print('Finished with ' + str(len(currentGroups)) + ' groups and minProb ' \
          + str(minSelfProb))

    # Return the dictionary holding the data.

    return topDict


# Add the current set of groups to the group dictionary, using the
# number of groups as the key.

def addGrouptoDict(topDict, zoneList, time, groupList, groupHierarchy, \
                   groupSizeList, currentGroups, jumpMatrix, transMatrix, \
                   balMatrix, netFluxMatrix, levels, colors, reorder=False):
    # Find the number of active groups.

    numCurrent = len(currentGroups)

    print('Adding ' + str(numCurrent) + ' groups to dictionary')

    newTransMatrix = np.zeros((numCurrent, numCurrent))
    newJumpMatrix = np.zeros((numCurrent, numCurrent))
    newBalMatrix = np.zeros((numCurrent, numCurrent))
    newConMatrix = np.zeros((numCurrent, numCurrent))
    netFlux = np.zeros((1, numCurrent))

    # This array will hold the group number for each point in the trajectory.

    groups = np.zeros(zoneList.shape)

    # Go through the currently active groups.

    for i in range(numCurrent):

        c = currentGroups[i]

        # Go through the zones in each group

        for g in groupList[c]:
            # Assign a group number to the appropriate points in the trajectory.

            groups[zoneList == g] = i

        # Create the matrices to be saved.

        for j in range(numCurrent):
            d = currentGroups[j]

            newTransMatrix[i, j] = transMatrix[c, d]
            newBalMatrix[i, j] = balMatrix[c, d]
            newJumpMatrix[i, j] = jumpMatrix[c, d]
            newConMatrix[i, j] = netFluxMatrix[c, d]

    netFlux = np.sum(newBalMatrix, axis=0) - np.sum(newBalMatrix, axis=1)
    netFlux = netFlux * time[-1]

    topDict['groups'][numCurrent] = groups
    topDict['transMatrix'][numCurrent] = newTransMatrix
    topDict['balMatrix'][numCurrent] = newBalMatrix
    topDict['jumpMatrix'][numCurrent] = newJumpMatrix
    topDict['netFluxMatrix'][numCurrent] = newConMatrix
    topDict['groupList'][numCurrent] = copy.deepcopy([groupList[i] for i in currentGroups])
    topDict['groupHierarchy'][numCurrent] \
        = copy.deepcopy([groupHierarchy[i] for i in currentGroups])
    topDict['groupSizeList'][numCurrent] \
        = copy.deepcopy([groupSizeList[i] for i in currentGroups])
    topDict['netFlux'][numCurrent] = netFlux

    if colors != []:
        topDict['colors'][numCurrent] = copy.deepcopy([colors[i] for i in currentGroups])


# Create new jump, transition, and balance matrices for a given zone array,
# based on the groups in groupList.

def newMatrices(groupList, groupSizeList, groupMap, zoneList):
    # groupList is the list of groups.
    # groupSizeList is the list of group sizes.
    # groupMap is a map of groups to zones.
    # zoneList is the data set of zones for the trajectory being analyzed.

    # The size of the jump matrix depends on the number of groups.

    numGroups = len(groupList)

    # We need to know the total number of data points for the detailed balance matrix.

    totalPoints = sum(groupSizeList)

    # Initialize the jump matrix as all zeros. When finished, this matrix holds
    # the number of jumps from group A to group B.

    jumpMatrix = np.zeros((numGroups, numGroups))

    # Go through the zone arrays in zoneDict one-by-one, adding each trajectory
    # to the jump matrix. This means that the jump matrix combines the data from
    # all the runs contained in zoneDict.

    # Find the group of the first data point. sG is the starting group, eG is the ending
    # group. The matrices are column stochastic, so jumpMatrix[eG, sG] is the number of
    # jumps from starting group sG to ending group eG.

    start = 0
    sG = groupMap[zoneList[start]]

    # Go through the data point-by-point.

    for end in range(1, len(zoneList)):
        # Find the group of the next point and update the jump matrix.

        eG = groupMap[zoneList[end]]

        jumpMatrix[eG, sG] = jumpMatrix[eG, sG] + 1

        # Move on to the next point.

        sG = eG

    # Calculate the transition and balance matrices from the jump matrix.

    transMatrix = np.zeros(jumpMatrix.shape)
    balMatrix = np.zeros(jumpMatrix.shape)
    netFluxMatrix = np.zeros(jumpMatrix.shape)

    # transMatrix[eG, sG] is the probability of jumping from starting group sG to ending
    # group eG. To calculate this, we divide the elements of each column of jumpMatrix
    # by the sum of that column.

    # balMatrix[eG, sG] is the flux from starting group sG to ending group eG. It is the
    # transition rate from sG to eG times the probability of being in group sG.

    for sG in range(numGroups):
        transMatrix[:, sG] = jumpMatrix[:, sG] / np.sum(jumpMatrix[:, sG])
        balMatrix[:, sG] = transMatrix[:, sG] * groupSizeList[sG] / totalPoints

    # netFluxMatrix[eG, sG] is the net flux from starting group sG to ending group eG.
    # If it is postive, then points are moving from sG to eG. This matrix is always
    # skew symmetric.

    netFluxMatrix = balMatrix - np.transpose(balMatrix)

    return jumpMatrix, transMatrix, balMatrix, netFluxMatrix


# Assign the zones to groups, one zone per group to start.

def newGroupList(zoneList):
    groupList = []
    groupSizeList = []
    groupMap = {}

    groupNum = 0

    # groupNum tracks the index of the list of groups.

    # Find the largest zone number that is occupied.

    maxNum = int(np.max(zoneList))

    # Iterate through all the zone numbers up to the maximum occupied zone.

    for z in range(maxNum + 1):

        # Find the number of data points in the current zone.

        num = np.sum(zoneList == z)

        # If the zone is occupied, make it a group and record the number of
        # data points in the zone.

        if num > 0:
            groupList.append([z])

            groupSizeList.append(num)

            groupMap[z] = groupNum

            groupNum = groupNum + 1

    currentGroups = [i for i in range(len(groupList))]

    # The group hierarchy starts out as the same list as the group list.

    groupHierarchy = copy.deepcopy(groupList)

    # print(groupHierarchy)

    # Return the list of groups and the list of group sizes.

    return (groupList, groupHierarchy, groupSizeList, groupMap, currentGroups)


# Use a probability matrix to determine which two groups should be combined.

def getSimilarGroups(testMatrix, compMatrix, currentGroups, probCutoff=0.98):
    # We will look at the diagonal elements of testMatrix and compMatrix
    # to determine which groups to combine. Note that these might be the
    # same matrices. The minimum value of the unfinished diagonal elements
    # of testMatrix will be one of the combined groups.

    # We use the maximum of the off-diagonal elements of compMatrix to
    # determine the second group to be combined.

    # probCutoff determines which groups are considered finished and will
    # no longer be combined.

    # Find the diagonal elements of the test and comparison matrices.

    diagTest = np.diagonal(testMatrix)
    diagComp = np.diagonal(compMatrix)

    # Determine which groups are eligible to be combined, based on their
    # self-transition probability.

    # August 21, 2022: I think this is a bug, and it should be diagTest here. But
    # it doesn't matter unless we use the 'Bal' method, and we don't. But this should
    # definitely be fixed/rewritten.

    notDoneFlag = diagComp < probCutoff

    # If all the groups are done, return a signal.

    if np.sum(notDoneFlag[currentGroups]) == 0:

        minG = -1
        maxG = -1

        print('Should be impossible!!!')

    else:

        # This picks out the active elements from the diagonal elements of
        # testMatrix.

        currentTest = diagTest[currentGroups]
        currentNotDoneFlag = notDoneFlag[currentGroups]

        currentTest = currentTest[currentNotDoneFlag]
        currentNotDone = np.array(currentGroups)[currentNotDoneFlag]

        minG = currentNotDone[np.argmin(currentTest)]

    # Make sure we don't try to combine a group with itself

    tM = copy.copy(testMatrix[:, minG])
    tM[minG] = 0

    # Find the group that this group jumps to the most

    maxG = np.argmax(tM)

    # Check that the two groups have symmetric transition probabilities
    if testMatrix[maxG, minG] == 0:

        # Remove the past candidate from consideration
        tM[maxG] = 0
        # Find a new grouping candidate
        maxG = findMaxG(testMatrix, minG, tM)

        # If we were not bale to find one return the same group
        if not maxG:
            return minG, minG
        else:
            return minG, maxG

    else:
        return minG, maxG


def findMaxG(testMatrix, minG, minGCol):
    """Function used to find symmetric transition groups """

    # Recursive base case
    if np.sum(minGCol) == 0:
        return False
    else:
        # Pick a grouping candidate
        maxG = np.argmax(minGCol)

        # Check for nonsymmetry
        if testMatrix[minG, maxG] == 0:
            # Remove candidate
            minGCol[maxG] = 0
            return findMaxG(testMatrix, minG, minGCol)
        else:
            return maxG


# Combine two groups. Group g1 will be added to group g2. The jump, transition,
# and balance matrices will also all be updated.

def combineGroups(g1, g2, groupList, groupHierarchy, groupSizeList, groupMap, \
                  currentGroups, jumpMatrix, transMatrix, balMatrix, netFluxMatrix):
    # np.set_printoptions(precision=0, suppress=True, linewidth=200)

    # print('Combining groups ' + str(g1) + ' and ' + str(g2))

    totalPoints = sum(groupSizeList)

    # Find the lowest flags in the group hierarchy lists

    if not groupHierarchy[g2]:
        print('Empty group ' + str(g2))
        print(currentGroups)
        print('-----')
        print(transMatrix)
        print('-----')
        print(netFluxMatrix)
        print('-----')

    mG1 = min(groupHierarchy[g1])
    mG2 = min(groupHierarchy[g2])

    # Append the appropriate flag. The value of the flag is the negative
    # of the number of zones in the combined group.

    if mG1 >= 0 and mG2 >= 0:

        groupHierarchy[g2].append(-2)

    elif mG1 >= 0:

        groupHierarchy[g2].append(mG2 - 1)

    elif mG2 >= 0:

        groupHierarchy[g2].append(mG1 - 1)

    else:

        groupHierarchy[g2].append(mG1 + mG2)

    # Add the first group to the second and update groupMap.

    for g in groupList[g1]:
        groupList[g2].append(g)
        groupMap[g] = g2

    groupSizeList[g2] = groupSizeList[g2] + groupSizeList[g1]

    # Combine the groups in the hierarchy

    for g in groupHierarchy[g1]:
        groupHierarchy[g2].append(g)

    # print(groupHierarchy[g2])

    # Make the first group empty

    groupList[g1] = []
    groupHierarchy[g1] = []

    groupSizeList[g1] = 0
    currentGroups.remove(g1)

    # Add the jumps from the first group to the second group

    jumpMatrix[:, g2] = jumpMatrix[:, g2] + jumpMatrix[:, g1]
    jumpMatrix[g2, :] = jumpMatrix[g2, :] + jumpMatrix[g1, :]

    # Set the removed row and column to zero.

    jumpMatrix[:, g1] = 0
    jumpMatrix[g1, :] = 0

    # Calculate the new rows and columns of the transition and balance matrices.

    transMatrix[:, g2] = jumpMatrix[:, g2] / np.sum(jumpMatrix[:, g2])
    balMatrix[:, g2] = transMatrix[:, g2] * groupSizeList[g2] / totalPoints

    # print('Currently have ' + str(len(currentGroups)) + ' groups: ' + str(currentGroups))

    for sG in currentGroups:
        transMatrix[g2, sG] = jumpMatrix[g2, sG] / np.sum(jumpMatrix[:, sG])
        balMatrix[g2, sG] = transMatrix[g2, sG] * groupSizeList[sG] / totalPoints

    # Set the probabilities for the removed groups to zero in all the probability matrices.

    transMatrix[g1, :] = 0
    transMatrix[:, g1] = 0
    balMatrix[g1, :] = 0
    balMatrix[:, g1] = 0
    netFluxMatrix[g1, :] = 0
    netFluxMatrix[:, g1] = 0

    for sG in currentGroups:
        netFluxMatrix[:, sG] = balMatrix[:, sG] - balMatrix[sG, :]


# 	print(currentGroups)
# 	print('-----')
# 	print(transMatrix)
# 	print('-----')
# 	print(netFluxMatrix)
# 	print('-----')

# Assign colors to the groups, starting with the largest radius in the polar plots

def assignColorstoGroups(groupList, currentGroups, level, zoneDict):
    masterList = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', '#66c2a5', '#fc8d62', '#8da0cb',
                  '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']

    # We'll assemble a list of median radii for the groups.

    medList = []

    # Iterate through the groups.

    for cG in currentGroups:

        medY = []

        # Find the middle of each zone in the group, and append it to a list.

        for g in groupList[cG]:
            y = 0.5 * (zoneDict[level][g]['ymin'] + zoneDict[level][g]['ymax'])
            medY.append(y)

        # Find the median of the radii of all the zones in the group and append it to the median list. We adjust the median by a small amount to prevent duplicates, which messes up the sorting.

        medList.append(np.median(medY) * (1 + 1e-4 * np.random.rand()))

    # Sort the list.

    orderedList = np.argsort(medList)

    # Create a list of colors, and add in the right number of colors, starting with C0.

    colorList = [''] * len(groupList)

    for i in range(len(medList)):
        colorList[currentGroups[orderedList[i]]] = masterList[i % len(masterList)]

    # Return the color list.

    return colorList


# Our function for parsing the group hierarchy.

def parseGroupHierarchy(groupList, maxDepth=0):
    # This function returns a dictionary holding the hierarchy of groups.
    # The first key is the depth of the parsing ... with 0 being unparsed,
    # 1 being parsed into two groups, 2 being parsed into 4, etc.
    # The second key is the index of the actual groups in the parsing.
    #
    # The split dictionary contains the hierarchy including the dividing flags.
    # The parsed dictionary has the flags removed; this is what is returned.
    #
    # If maxDepth = 0, we parse until every group contains a single zone.
    # In that case, max(parsed.keys()) will contain the key for the last
    # grouping, the set of primordial zones for the group.
    #
    # If maxDepth > 0, we just parse to that depth. In both cases,
    # p[0][0] contains the list of all the zones in the group, so
    # len(p[0][0]) is the total number of zones in the group.

    parsed = {}
    split = {}

    parsed[0] = {}
    parsed[0][0] = removeFlags(groupList)
    split[0] = {}
    split[0][0] = groupList

    d = 0

    while True:

        newD = d + 1

        split[newD] = {}
        parsed[newD] = {}

        newK = 0
        maxSize = 1

        for k in range(len(split[d].keys())):

            if len(split[d][k]) == 1:

                split[newD][newK] = split[d][k]
                parsed[newD][newK] = split[d][k]

                newK = newK + 1

            else:

                split[newD][newK], split[newD][newK + 1] = splitGroup(split[d][k])
                parsed[newD][newK] = removeFlags(split[newD][newK])
                parsed[newD][newK + 1] = removeFlags(split[newD][newK + 1])

                maxSize = max([len(parsed[newD][newK]), len(parsed[newD][newK + 1]), maxSize])

                newK = newK + 2

        if maxSize == 1:

            break

        elif maxDepth > 0 and newD > maxDepth:

            break

        else:

            d = d + 1

    return parsed


# Split a group hierarchy in two.

def splitGroup(x):
    if len(x) == 0:

        return [], []

    elif len(x) == 1:

        return [x[0]], []

    elif len(x) == 3:

        return [x[0]], [x[2]]

    else:

        mG = min(x)
        iG = x.index(mG)

        lG = x[:iG]
        rG = x[(iG + 1):]

        return lG, rG


# Remove all the hierarchy flags from a grouping.

def removeFlags(x):
    return list(np.array(x)[np.array(x) >= 0])


# Print out some statistics about the current groupings.

def printGroupStatistics(groupSizeList, currentGroups, transMatrix):
    diagProb = np.diagonal(transMatrix)[currentGroups]

    index = np.where(diagProb == 0)

    # print(index[0])
    # print(diagProb[index[0]])
    # print(list(np.array(groupSizeList)[currentGroups][index[0]]))

    # zeroGroups = list(np.array(currentGroups)[index[0]])

    print('There are ' + str(len(index[0])) \
          + ' zones with 0 self-probability out of ' \
          + str(len(currentGroups)) + ' total')

    zeroDict = {}
    outDict = {}

    for sG in currentGroups:

        n = groupSizeList[sG]
        r = transMatrix[:, sG]
        rNZ = r > 0
        o = sum(rNZ)

        if n == 0:
            # This should never happen!

            print('Group ' + str(sG) + ' has no data points!')

        if n in zeroDict.keys():

            zeroDict[n] = zeroDict[n] + 1

        else:

            zeroDict[n] = 1

        if o in outDict.keys():

            outDict[o] = outDict[o] + 1

        else:

            outDict[o] = 1

    zeroKeys = list(zeroDict.keys())
    zeroKeys.sort()

    for n in zeroKeys:

        print('There are ' + str(zeroDict[n]) + ' zones with ' \
              + str(n) + ' data points')

        if n >= 20:
            break

    outKeys = list(outDict.keys())
    outKeys.sort()

    for o in outKeys:

        print('There are ' + str(outDict[o]) + ' zones that transition to ' \
              + str(o) + ' zones')

        if o >= 20:
            break


def eliminateSmallGroups(minGroupSize, groupList, groupHierarchy, \
                         groupSizeList, groupMap, currentGroups, \
                         jumpMatrix, transMatrix, balMatrix, netFluxMatrix, \
                         theLevel, zoneDict):
    # Go though the list of current groups and look for groups that are smaller
    # than minGroupSize.

    smallGroups = [g for g in currentGroups if groupSizeList[g] < minGroupSize]

    print("Starting the elimination of small groups")
    print("There are currently " + str(len(smallGroups)) \
          + " groups with fewer than " + str(minGroupSize) + " data points")

    # We will try to combine groups at most maxPasses times.

    maxPasses = 10
    passNumber = 0

    # Keep track of isolated groups ... groups that are smaller than minGroupSize
    # but that don't have neighboring populated zones.

    isolatedGroups = []

    while len(smallGroups) > 0 and passNumber < maxPasses:

        # We have to be careful not to change smallGroups as we go through this loop.
        # Each time through this loop, we combine the current small group with a single
        # other group, which eliminates the current group from currentGroups. At the
        # end of this loop, we regenerate smallGroups for the next iteration.

        for g in smallGroups:

            # Group g might no longer be small if another group was already added
            # to it.

            if groupSizeList[g] >= minGroupSize:
                continue

            neighboringZones = cZ.findNeighboringZones(groupList[g], theLevel, zoneDict)

            neighboringGroups = [];

            for z in neighboringZones:

                if z in groupMap.keys():

                    gg = groupMap[z]

                    if gg not in neighboringGroups:
                        neighboringGroups.append(gg)

            if not neighboringGroups:
                isolatedGroups.append(g)
                continue

            neighboringSizes = [groupSizeList[gg] for gg in neighboringGroups]

            sortedSizes = sorted(((e, i) for i, e in enumerate(neighboringSizes)), reverse=True)

            # Combine g with its largest neighbor.

            size = sortedSizes[0][0]
            gg = neighboringGroups[sortedSizes[0][1]]

            # Because group g is the first argument here, it is added to group gg and
            # is then eliminated from the currentGroups list.

            combineGroups(g, gg, groupList, groupHierarchy, \
                          groupSizeList, groupMap, currentGroups, \
                          jumpMatrix, transMatrix, balMatrix, netFluxMatrix)

        smallGroups = [g for g in currentGroups if groupSizeList[g] < minGroupSize]
        smallGroups = [g for g in smallGroups if groupSizeList[g] > 0]
        smallGroups = [g for g in smallGroups if g not in isolatedGroups]
        passNumber = passNumber + 1

        print("After pass number " + str(passNumber) + " there are " \
              + str(len(smallGroups) + len(isolatedGroups)) + " groups with fewer than " \
              + str(minGroupSize) + " data points")
        print("Of these, " + str(len(isolatedGroups)) + " are isolated")

    print("Finished eliminating small groups")
