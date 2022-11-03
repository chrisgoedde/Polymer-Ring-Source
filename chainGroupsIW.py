import numpy as np
import time as tP
import copy
import chainZones as cZ


def groupZones(zoneList, zoneDict, time, theLevel, minG):
    """Group zones using the inchworm method"""

    start = tP.time()

    colors = []

    topDict = genTopDict(time, zoneList)

    # Instantiate any objects that keep track of the groups we have
    # as well as the datapoints  in those groups. These are our zone-based
    # groupings
    (groupList, groupSizeList, groupMap, currentGroups, groupSeries) \
        = newGroupList(zoneList)

    # Instantiate any matrices that keep track of transitions, detailed balance,
    # or connectivity
    (jumpMatrix, transMatrix, balMatrix, netFluxMatrix) \
        = newMatrices(groupList, groupSizeList, groupMap, zoneList)

    # Optional subroutine, pass minG = 0 to avoid
    eliminateSmallGroups(minG, groupList, groupSizeList, groupMap, currentGroups,
                         jumpMatrix, transMatrix, balMatrix, netFluxMatrix,
                         theLevel, zoneDict)

    grouped = False
    first = True
    while not grouped:

        if first:
            # Current group of interest (goi)
            goi = int(groupSeries[0])
            # First occurrence of the group in the series
            fOcc = 0
            first = False

        # Identify all the occurrences of this group in the series
        occ = np.where(groupSeries == goi)

        # If this group occurs multiple times in the series, then proceed
        if len(occ[0]) > 1:
            groupSeries = growGroup(fOcc, occ, goi, groupSeries, groupList, groupSizeList,
                                    groupMap, currentGroups, jumpMatrix, transMatrix, balMatrix, netFluxMatrix)

            # Now, if the intermediate groups that were joined to goi had occurrences in a
            # later time in the series, then we can have more intermediate groups, so we'll
            # have to check for that and update our goi and fOcc accordingly
            prevOcc = occ[0][-1]
            newOcc = np.where(groupSeries == goi)

            # Check if the last occurrence of goi has changed after grouping
            if newOcc[0][-1] != prevOcc:
                # If it has, then there are more groups to join, so retain our current markers
                goi = goi
                fOcc = fOcc
            else:
                # If it hasn't, Then there are no groups left to join past our last occurrence
                # Check: Are we at the end of the series?
                if occ[0][-1] == len(groupSeries) - 1:
                    # Then the grouping is complete
                    grouped = True
                else:
                    # We have more grouping to do, set a new group of interest
                    goi = int(groupSeries[newOcc[0][-1] + 1])
                    # Mark the first occurrence of this new group of interest in the series
                    fOcc = newOcc[0][-1] + 1

        # Else, move onto the next group we can consider
        else:
            goi = int(groupSeries[fOcc + 1])
            fOcc += 1
        # Check: Are we at the end of the series?
        if occ[0][-1] == len(groupSeries) - 1:
            # Then we are done grouping
            grouped = True

        # Time tracking
        end = tP.time()
        if (end - start) > 180:
            print('Still working with ' + str(len(currentGroups)) \
                  + ' groups after ' + '{:.1f}'.format((end - start) / 60) \
                  + ' minutes')
            start = end

    # Complete the algorithm by adding the groups to a dictionary
    addGrouptoDict(topDict, zoneList, time, groupList,
                   groupSizeList, currentGroups, jumpMatrix,
                   transMatrix, balMatrix, netFluxMatrix,
                   theLevel, colors)

    return topDict


def growGroup(fOcc, occ, goi, groupSeries, groupList, groupSizeList,
              groupMap, currentGroups, jumpMatrix, transMatrix, balMatrix, netFluxMatrix):
    """Group the intermediate groups between the first and last goi occurrences,
    and relabel the entries in grouplist"""

    # What is the last occurrence of goi?
    last = occ[0][-1]

    # What is the span in the time series that we are currently focusing on?
    span = groupSeries[fOcc:last + 1]
    # What are the intermediate groups we need to join to goi?
    inter = np.where(span != goi)

    # Is the span just a series of the same group?
    if len(inter[0]) != 0:
        # If not, then create a list of the intermediate groups
        temp = span[inter]

        # For each of these intermediate groups, run the combining subroutine on them
        for entry in temp:
            # We could encounter a group that we already joined in a prior iteration, check if this is the case
            if groupMap[int(entry)] != groupMap[int(goi)]:

                combineGroups(groupMap[int(entry)], groupMap[int(goi)], groupList, groupSizeList, groupMap,
                              currentGroups, jumpMatrix, transMatrix, balMatrix, netFluxMatrix)

                # Where is this intermediate group occurring in the time series?
                interOcc = np.where(groupSeries == entry)

                # Relabel these intermediate occurrences with goi since they are now joined
                groupSeries[interOcc] = goi

    # Return the updated series
    return groupSeries


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


def newGroupList(zoneList):
    groupList = []
    groupSizeList = []
    groupMap = {}
    groupSeries = zoneList
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

    # Return the list of groups and the list of group sizes.

    return groupList, groupSizeList, groupMap, currentGroups, groupSeries


def eliminateSmallGroups(minGroupSize, groupList,
                         groupSizeList, groupMap, currentGroups,
                         jumpMatrix, transMatrix, balMatrix, netFluxMatrix,
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

            combineGroups(g, gg, groupList,
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


def combineGroups(g1, g2, groupList, groupSizeList, groupMap, \
                  currentGroups, jumpMatrix, transMatrix, balMatrix, netFluxMatrix):
    # np.set_printoptions(precision=0, suppress=True, linewidth=200)

    # print('Combining groups ' + str(g1) + ' and ' + str(g2))

    totalPoints = sum(groupSizeList)

    #

    # Add the first group to the second and update groupMap.

    for g in groupList[g1]:
        groupList[g2].append(g)
        groupMap[g] = g2

    groupSizeList[g2] = groupSizeList[g2] + groupSizeList[g1]

    groupList[g1] = []

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


def addGrouptoDict(topDict, zoneList, time, groupList,
                   groupSizeList, currentGroups, jumpMatrix, transMatrix,
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
    topDict['groupSizeList'][numCurrent] \
        = copy.deepcopy([groupSizeList[i] for i in currentGroups])
    topDict['netFlux'][numCurrent] = netFlux

    if colors != []:
        topDict['colors'][numCurrent] = copy.deepcopy([colors[i] for i in currentGroups])


def genTopDict(time, zoneList):
    """Return a new instance of topDict"""

    topDict = {'time': time, 'zones': zoneList, 'groups': {}, 'jumpMatrix': {}, 'transMatrix': {}, 'balMatrix': {},
               'conMatrix': {}, 'netFlux': {}, 'netFluxMatrix': {}, 'colors': {}, 'groupList': {}, 'groupSizeList': {},
               'currentGroups': {}}

    return topDict
