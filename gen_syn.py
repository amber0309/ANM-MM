from __future__ import division
import numpy as np
import itertools
import numpy.matlib

def gen_each(inlist):
	"""
	Function for generating synthetic data
	inlist[0] - dist of x
	inlist[1] - f1
	inlist[2] - dist of noise
	inlist[3] - f2
	inlist[4] - number of pts
	"""
	if len(inlist) != 5:
		print 'not enough parameters!'
		return -1

	n = inlist[4]
	# ----- dist of x -----
	if inlist[0] == 1:
		x = np.random.rand(n, 1)
	elif inlist[0] == 2:
		x = np.random.randn(n, 1)
	elif inlist[0] == 3:
		x = np.random.exponential(0.5, [n, 1])
	elif inlist[0] == 4:
		x = np.random.laplace(0, 1, [n, 1])
	elif inlist[0] == 5:
		x = np.random.lognormal(0, 1, [n, 1])

	# ----- f_1 -----
	if inlist[1] == 0:
		f1 = -x 
	elif inlist[1] == 1:
		f1 = np.exp( -(np.random.rand() * 0.1 + 1) * x )
	elif inlist[1] == 2:
		f1 = np.exp( -(np.random.rand() * 0.1 + 3) * x )

	# ----- noise -----
	if inlist[2] == 0:
		t = f1
	elif inlist[2] == 1:
		t = f1 + 0.2 * np.random.rand(n, 1)
	elif inlist[2] == 2:
		t = f1 + 0.05*np.random.randn(n, 1)
	elif inlist[2] == 3:
		t = f1 + np.random.exponential(0.5, [n, 1])
	elif inlist[2] == 4:
		t = f1 + np.random.laplace(0, 1, [n, 1])
	elif inlist[2] == 5:
		t = f1 + np.random.lognormal(0, 1, [n, 1])

	# ----- f_2 -----
	if inlist[3] == 0:
		y = t
	elif inlist[3] == 1:
		y = 1 / (t**2 + 1)
	elif inlist[3] == 2:
		y = t**2
	elif inlist[3] == 3:
		y = np.sin(t)
	elif inlist[3] == 4:
		y = np.cos(t)
	elif inlist[3] == 5:
		y = np.cbrt(t)

	xy = np.hstack((x, y))
	return xy

def row_permute_independently(array):
	"""
	permute each row of a matrix independently
	"""
	nrows, ncols = array.shape
	all_perm = np.array((list(itertools.permutations(range(ncols)))))
	b = all_perm[np.random.randint(0, all_perm.shape[0], size=nrows)]
	return array.take((b+3*np.arange(nrows)[...,np.newaxis]).ravel()).reshape(array.shape)


def gen_randD(n_mech, N):
	"""
	generate a random data set
	n_mech - number of underlying mechanisms
	N      - number of pts from each mechanism
	"""
	model_ind = numpy.matlib.repmat([1, 2, 3, 4, 5], n_mech, 1)
	model_ind = row_permute_independently(model_ind)
	D = []
	label_true = []
	for i in range(0, n_mech):
		func = model_ind[:,i].tolist()
		func = func + [0, N]
		D.append(gen_each(func))
		label_true = label_true + [i] * N

	D = np.concatenate(D, axis = 0)
	label_true = np.asarray(label_true)
	return D, label_true

def gen_D(inlist):
	"""
	generate a specified data set
	inlist - list of input list for gen_each()
	"""
	data = []
	label_true = []
	label_cnt = 0
	for sublist in inlist:
		data.append(gen_each(sublist))
		label_true = label_true + [label_cnt] * sublist[4]
		label_cnt += 1

	data = np.vstack(data)
	return data, label_true


if __name__ == '__main__':
	"""
	Function for generating synthetic data
	inlist[0] - dist of x
	inlist[1] - f1
	inlist[2] - dist of noise
	inlist[3] - f2
	inlist[4] - number of pts
	"""
	d = gen_each([3, 2, 5, 0, 400])
	