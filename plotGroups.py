import plotUtils as plU
import matplotlib.pyplot as plt

import numpy as np

def makeMatrixPlot(picturePathName, matrix, plotRange = (), fileName = '', \
				   titleString = '', finderComment = '', printDiagonal = True, \
					   colors = [], xlabels = [], ylabels = [], digits = 1):

	(a, b) = matrix.shape
	
	if a > 16:
	
		textSize = 6
		boxTextSize = 3
		
	else:
	
		textSize = 10
		boxTextSize = 6

	if plotRange == ():

		numZones = a
		plotRange = (0, numZones)

	else:

		numZones = plotRange[1] - plotRange[0]

	f, ax = plU.makeFigure(figsize=(7,7), textSize = textSize)
	
	ax.set_xlim((plotRange[0]-0.5, plotRange[1]-0.5))
	ax.set_ylim((plotRange[0]-0.5, plotRange[1]-0.5))
	plt.xlabel('Starting Group')
	plt.ylabel('Ending Group')
	ax.invert_yaxis()

	for axis in [ax.xaxis, ax.yaxis]:

		axis.set_ticks(np.arange(plotRange[0],plotRange[1]))
		axis.set_ticks(np.arange(plotRange[0]+0.5,plotRange[1]-0.5), minor=True)

	ax.grid(True, which='minor')

	if xlabels != []:

		ax.set_xticklabels(xlabels)

	if ylabels != []:

		ax.set_yticklabels(ylabels)

	fString = '{:.' + str(digits) + 'f}'

	for sG in range(plotRange[0], plotRange[1]):

		for eG in range(plotRange[0], plotRange[1]):

			if sG == eG and not printDiagonal:

				continue

			if matrix[eG,sG] != 0:

				mText = plt.text(sG, eG, fString.format(matrix[eG,sG]), \
									 ha = 'center', va = 'center')

				if colors != []:

					mText.set_color(colors[sG%(len(colors))])

				if sG == eG:

					mText.set_color('red')

				mText.set_fontsize(boxTextSize)

	if titleString != '':

		plt.title(titleString)

	if fileName != '':

		plU.savePlot(picturePathName, fileName, closeFigure = True, finderComment = finderComment)

def makeGroupScatterPlot(picturePathName, numGroups, group, x, y, time, colors, netFlux, \
					fileName = '', titleString = '', finderComment = '', \
						xLim = (0, 1), xLabel = '', yLim = (0, 1), \
							yTicks = (), yLabel = '', polar = True, \
								median = False, connectPoints = False):

	medTimes = []
	medX = []
	medY = []

	for g in range(numGroups):

		if (group == g).any():

			medTimes.append(np.median(time[group == g]))
			medX.append(np.median(x[group == g]))
			medY.append(np.median(y[group == g]))

		else:

			medTimes.append(np.nan)
			medX.append(np.nan)
			medY.append(np.nan)

	timeOrderList = np.argsort(medTimes)

	if polar:

		f, ax = plU.makeFigure(figsize = (6,6), polar = True)

		ax.set_xticks([np.pi/3, 2*np.pi/3])
		ax.set_xticklabels(['oblate','prolate'])
		ax.set_thetamin(60)
		ax.set_thetamax(120)
		ax.set_rlim(0,1)
		ax.set_rticks([0.25, 0.5, 0.75, 1])
		# plt.text(0.8, 0.5, 'flatness', rotation=60)
		plt.text(0.8, 0.5, yLabel, rotation=60)

		for g in timeOrderList:

			if (group == g).any():

				# print('Plotting points at time ' + str(medTimes[g]))

				if connectPoints:

					plt.rcParams['agg.path.chunksize'] = 10000

					plt.plot(2*np.pi/3-x[group == g], y[group == g], \
							 alpha = 0.2, color = colors[g])

				else:

					plt.plot(2*np.pi/3-x[group == g], y[group == g], '.', \
							 alpha = 0.2, markeredgewidth = 0.0, color = colors[g])

				xt = 0.65
				yt = 0.5 - 0.05*g
				tt = np.pi - np.arctan(yt/xt)
				rt = np.sqrt(xt**2 + yt**2)
				plt.text(tt, rt, 'G' + str(g) + ': ' + str(np.sum(group == g)) \
					+ ' pts with net flux ' '{:.2f}'.format(netFlux[g]))
				print('Group ' + str(g) + ' has ' + str(np.sum(group == g)) + ' points')

		if median:

			newX = []
			newY = []

			for g in timeOrderList:

				if (group == g).any():

					newX.append(2*np.pi/3-medX[g])
					newY.append(medY[g])

					plt.text(2*np.pi/3-medX[g], medY[g]+0.02, str(g))

			plt.plot(newX, newY, 'o-k', linewidth=1, markersize = 8, \
					 markeredgecolor = 'k', markerfacecolor='none')


	else:

		f, ax = plU.makeFigure()

		ax.set_xlim(xLim)
		plt.xlabel(xLabel)

		ax.set_ylim(yLim)
		plt.ylabel(yLabel)

		if yTicks != ():

			plt.yticks(yTicks)

		plt.plot(x, y, '.', alpha = 0.25, markeredgewidth = 0.0)

	if titleString != '':

		plt.title(titleString, fontsize = 10)

	if fileName != '':

		plU.savePlot(picturePathName, fileName, type = 'png', dpiArg = 600, \
					 closeFigure = True, finderComment = finderComment)
