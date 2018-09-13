"""
ANM-MM (ANM Mixture Model) python implementation
(Anaconda4.3.30 64-bit for python 2.7.14 on Windows 10)

Shoubo Hu (shoubo.sub AT gmail.com)
13/09/2018

USAGE:
  direction = ANMMM_cd(data, lda)
  labels = ANMMM_clu(data, label, lda)

INPUT:
  XY          - input data, Numpy array with 2 columns and any number of rows. 
                      Rows represent i.i.d. samples, The first column is the 
                      hypothetical $X$ and the second is the hypothetical $Y$.
  label       - List of true labels of each observation.
  lda         - The parameter $lambda$ which controls the importance of HSIC term.
 
OUTPUT: 
  direction      1,  the first column ($X$) is the cause
                -1,  the second column ($Y$) is the cause
                 0,   can not tell
  labels         List of estimated clustering labels of each observation
 
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans

from GPPOM_HSIC import GPPOMC_lnhsic
from HSIC import hsic_gam

def draw_clu(data, label, name):
	## draw the whole data set
	## pts in diff color belong to diff clusters
	colours = ['#F13C20','#E27D60','#BC986A','#4056A1','#D79922','#379683','#379683']#something to colour the dots with...
	markers = ['o', 'p', 's', 'v', '^', '<', '>']
	label_list = np.unique(label)
	plt.figure()
	for ilabel in label_list:
		cu_indices = [i for i, x in enumerate(label) if x == ilabel]
		cu_data = data[cu_indices, :]
		plt.scatter(cu_data[:,0].reshape(-1, 1), cu_data[:,1].reshape(-1, 1), c = colours[ilabel], marker = markers[ilabel])

	plt.axis('equal')
	plt.title(name)
	# plt.show()

def ANMMM_cd(data, lda):

	X = data[:,0].reshape(-1 ,1)
	Y = data[:,1].reshape(-1 ,1)

	# apply GPLVM: x --> y
	myGPLVM1 = GPPOMC_lnhsic(X, Y, 1, lda, nhidden=20)
	myGPLVM1.learn()

	w = myGPLVM1.pack()
	ll1 = myGPLVM1.ll(w)

	z_x2y = myGPLVM1.GP_z.Z
	stat1, thresh1 = hsic_gam(z_x2y, X, 0.05)
	r1 = stat1 / thresh1

	# apply GPLVM: y --> x
	myGPLVM2 = GPPOMC_lnhsic(Y, X, 1, lda, nhidden=20)
	myGPLVM2.learn()
	w = myGPLVM2.pack()
	ll2 = myGPLVM2.ll(w)

	z_y2x = myGPLVM2.GP_z.Z
	stat2, thresh2 = hsic_gam(z_y2x, Y, 0.05)
	r2 = stat2 / thresh2

	print 'r1 = ', r1
	print 'r2 = ', r2

	if r1 < r2:
		print 'X --> Y'
		return 1
	elif r1 > r2:
		print 'Y --> X'
		return -1
	else:
		return 0


def ANMMM_clu(data, label_true, ilda):

	X = data[:,0].reshape(-1 ,1)
	Y = data[:,1].reshape(-1 ,1)
	nclu = len(np.unique(label_true))

	# ----- apply GPLVM
	plt_flag = 0
	myGPLVM = GPPOMC_lnhsic(X, Y, 1, ilda, nhidden=20)

	w = myGPLVM.pack()
	print 'initial objective value:', myGPLVM.ll(w)

	myGPLVM.learn()
	w = myGPLVM.pack()
	ll = myGPLVM.ll(w)
	print 'optimized objective value:', myGPLVM.ll(w)

	# ----- post-process Z
	z_params = myGPLVM.GP_z.Z
	k1 = KMeans(init='k-means++', n_clusters=nclu, n_init=50)
	clu_label = k1.fit_predict( z_params )

	print 'ARI of ANM-MM:', metrics.adjusted_rand_score(label_true, clu_label), '\n'

	if (z_params.shape[1] <= 2) & (plt_flag == 1):
		draw_clu(z_params, label_true, 'estimated Z with true labels')
		draw_clu(data, clu_label, 'Clustering results')
		draw_clu(data, label_true, 'Ground truth')
		plt.show()
	else:
		pass

	return clu_label