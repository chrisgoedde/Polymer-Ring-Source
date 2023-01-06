import numpy as np
import copy
import math
import plotZones as pz


def visualizeSpan(picturePathName, span, zoneDict):
    """Save an image of the span placed in the configuration space"""
    pz.drawSpan(picturePathName, span, zoneDict)


def visualizeJoins(picturePathName, spans, colors, zoneDict, M):
    """Image the join of two spans"""
    pz.drawSpans(picturePathName, spans, colors, zoneDict, M)


def disMax(Si, residuals, M):
    """Find the maximally dissimilar residual span"""

    # We will need to calculate two quantities to assert the maximally dissimilar group:
    # the percent overlap and the activity delta.
    D = []
    for idx in residuals:
        Sj = Span(idx, M)
        # Acur = Sj.A
        # delta = compDelta(Si, Sj, M)
        joint = getJoint(Si, Sj, M)
        JA = compA(joint)
        ovpCur = len(joint) / len(Sj.S)
        d = math.sqrt(((1 - Sj.A) ** 2) + ((0 - ovpCur) ** 2))
        prod = np.vdot(np.array([[Sj.A], [ovpCur]]), np.array([[1], [0]]))
        D.append(d)
    compArray = np.vstack((residuals, D))
    compArray = compArray[:, compArray[1, :].argsort()]
    return compArray[0, 0]


def compA(span):
    """Compute the activity value of the passed span, S"""
    if len(span) == 1:
        return 0
    else:
        S = copy.deepcopy(span)
        # We need to add 1 to each zone because 0 will be
        # a relevant value in the difference between l and r
        S = S + 1
        l = S[0:len(S) - 1]
        r = S[1:len(S)]
        # Compute the number of transitions contained within the span,
        # Beta
        trans = l - r
        beta = np.sum(trans != 0)
        # Compute the number of unique zones in the span,
        # alpha
        alpha = len(set(S))
        # Return the activity value
        return (alpha * beta) / ((len(S) - 1) ** 2)


def findTypesMapless(span, M):
    """Return a dictionary with all subspans and intersecting spans of the passed span without referencing a mapping"""

    retDict = {'subs': [], 'intercL': [], 'intercR': []}

    mains = span.mains

    if span.homogenous:
        return []
    else:
        done = []
        inter = M[mains[0] + 1:mains[1]]
        for idx in inter:
            if idx not in done:
                if idx != span.S[0]:
                    # We use the mains of the intermediate spans to
                    # deduce the type of intermediate spans we are
                    # dealing with
                    Sj = Span(idx, M)
                    mainsDiff = diffTuple(Sj.mains, mains)
                    # There are only three possible outcomes of this difference
                    # tuple: (-,-) [left-intersect], (+, -) [subspan], (+,+) [right-intersect]
                    if (mainsDiff[0] < 0) and (mainsDiff[1] < 0):
                        retDict['intercL'].append(idx)
                    elif (mainsDiff[0] > 0) and (mainsDiff[1] < 0):
                        retDict['subs'].append(idx)
                    elif (mainsDiff[0] > 0) and (mainsDiff[1] > 0):
                        retDict['intercR'].append(idx)
                    else:
                        if (mainsDiff[0] < 0) and (mainsDiff[1] > 0):
                            pass
                        else:
                            print(f'mains1: {mains}')
                            print(f'mains2: {Sj.mains}')
                            raise "This can't be a case"
                    done.append(idx)
        return retDict


def findTypes(span, sMap, M):
    """Return a dictionary with all subspans and intersecting spans of the passed span"""

    retDict = {'subs': [], 'intercL': [], 'intercR': []}

    mains = span.mains

    if span.homogenous:
        return []
    else:
        done = []
        inter = M[mains[0] + 1:mains[1]]
        for idx in inter:
            if idx not in done:
                if idx != span.S[0]:
                    # We use the mains of the intermediate spans to
                    # deduce the type of intermediate spans we are
                    # dealing with
                    Sj = sMap[idx]
                    mainsDiff = diffTuple(Sj.mains, mains)
                    # There are only three possible outcomes of this difference
                    # tuple: (-,-) [left-intersect], (+, -) [subspan], (+,+) [right-intersect]
                    if (mainsDiff[0] < 0) and (mainsDiff[1] < 0):
                        retDict['intercL'].append(idx)
                    elif (mainsDiff[0] > 0) and (mainsDiff[1] < 0):
                        retDict['subs'].append(idx)
                    elif (mainsDiff[0] > 0) and (mainsDiff[1] > 0):
                        retDict['intercR'].append(idx)
                    else:
                        if (mainsDiff[0] < 0) and (mainsDiff[1] > 0):
                            pass
                        else:
                            print(f'mains1: {mains}')
                            print(f'mains2: {Sj.mains}')
                            raise "This can't be a case"
                    done.append(idx)
        return retDict


def getJoint(Si, Sj, M):
    """Return the joint of spans Si and Sj"""
    if (Si.mains[0] < Sj.mains[0]) and (Si.mains[1] < Sj.mains[0]):
        return []
    elif (Si.mains[0] > Sj.mains[1]) and (Si.mains[1] > Sj.mains[1]):
        return []
    else:
        p = max(Si.mains[0], Sj.mains[0])
        q = min(Si.mains[1], Sj.mains[1])
        return M[p:q + 1]


def diffTuple(mains1, mains2):
    """Return the difference tuple for two main index sets"""
    return tuple((mains1[0] - mains2[0], mains1[1] - mains2[1]))


def compDelta(Si, Sj, M):
    """Compute the joining delta of two spans Si and Sj, J(Si, Sj)"""

    JS = joinSpans(Si, Sj, M)

    A0 = Si.A
    Af = compA(JS)

    return Af - A0


def joinSpans(Si, Sj, M):
    """Return J(Si, Sj)"""

    mainsDiff = diffTuple(Sj.mains, Si.mains)

    # There are only three possible outcomes of this difference
    # tuple: (-,-) [left-intersect], (+, -) [subspan], (+,+) [right-intersect]
    if (mainsDiff[0] < 0) and (mainsDiff[1] < 0):
        join = copy.deepcopy(M[Sj.mains[0]:Si.mains[1] + 1])
    elif (mainsDiff[0] > 0) and (mainsDiff[1] < 0):
        join = copy.deepcopy(M[Si.mains[0]:Si.mains[1] + 1])
    elif (mainsDiff[0] > 0) and (mainsDiff[1] > 0):
        join = copy.deepcopy(M[Si.mains[0]:Sj.mains[1] + 1])
    elif (mainsDiff[0] < 0) and (mainsDiff[1] > 0):
        join = copy.deepcopy(M[Sj.mains[0]:Sj.mains[1] + 1])
    else:
        print(f'mains1: {Si.mains}')
        print(f'mains2: {Sj.mains}')
        raise "This can't be a case"

    instances = np.where(join == Sj.S[0])
    join[instances] = Si.S[0]

    return join


# A class representation of a Span

class Span(object):

    def __init__(self, zone, M):
        """Constructor function"""

        # Instantiate the main indices of the span
        instances = np.where(M == zone)
        if len(instances[0]) == 1:
            self.mains = tuple((instances[0][0], instances[0][0]))
            self.homogenous = True
        elif len(instances[0]) == 0:
            raise "A wrong zone value has been passed"
        else:
            self.mains = tuple((instances[0][0], instances[0][-1]))

        # Instantiate the tuple of the span
        self.S = M[self.mains[0]:self.mains[1] + 1]

        # Instantiate the activity value of the span
        activity = compA(self.S)
        if activity:
            self.A = activity
            self.homogenous = False
        else:
            self.A = activity
            self.homogenous = True
