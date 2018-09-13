# -*- coding: utf-8 -*-

"""
GPPOM with HSIC enforcement python implementation 
(Anaconda4.3.30 64-bit for python 2.7.14 on Windows 10)
(modified from https://github.com/jameshensman/pythonGPLVM)

Shoubo Hu (shoubo.sub AT gmail.com)
13/09/2018
"""

from __future__ import division
import numpy as np
from scipy.optimize import fmin_cg
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA
import kernels
import GP_z
import MLP


class GPLVM_z_lnhsic:
	def __init__(self,X,Y,dim):
		self.Zdim = dim

		self.N,self.Ydim = Y.shape
		self.Xdim = X.shape[1]
		self.XY = np.concatenate((X, Y), axis = 1)
		self.lda_2 = 1e-4

		"""Use PCA to initalise the problem."""
		pca = PCA( n_components=self.Zdim )
		self.Z = pca.fit_transform(self.XY)
		
		self.GP_z = GP_z.GP_z(self.Z, X, Y)

	def updateM(self):
		vals, vecs = eigsh(self.GP_z.kernelz(self.GP_z.Z, self.GP_z.Z), k=2, which = 'LM')
		self.M = np.real(vecs)

	def learn(self,niters):
		for i in range(niters):
			self.optimise_latents()
			self.optimise_GP_kernel()
			
	def optimise_GP_kernel(self):
		"""optimisation of the GP_z's kernel parameters"""
		self.GP_z.find_kernel_params()
		print self.GP_z.marginal(), 0.5*np.sum(np.square(self.GP_z.X))
	
	def ll(self,xx,i):
		"""The log likelihood function - used when changing the ith latent variable to xx"""
		H = np.identity(self.N) - np.ones((self.N, self.N), dtype = float) / self.N

		self.GP_z.Z[i] = xx
		self.GP_z.update()
		return -self.GP_z.marginal()+ 0.5*np.sum(np.square(xx)) + \
			self.lda_2 * np.log( np.trace( np.dot(np.dot(np.dot(self.GP_z.kernelx(self.GP_z.X, self.GP_z.X), H ), self.GP_z.kernelz(self.GP_z.Z, self.GP_z.Z)), H) ) )
	
	def ll_grad(self,xx,i):
		"""the gradient of the likelihood function for us in optimisation"""
		H = np.identity(self.N) - np.ones((self.N, self.N), dtype = float) / self.N

		self.GP_z.Z[i] = xx
		self.GP_z.update()
		self.GP_z.update_grad()
		matrix_grads = [self.GP_z.kernelz.gradients_wrt_data(self.GP_z.Z,i,jj) for jj in range(self.GP_z.Zdim)]
		grads = [-0.5*np.trace(np.dot(self.GP_z.alphalphK + \
			self.lda_2 * ( 1/np.trace( np.dot(np.dot(np.dot(self.GP_z.kernelx(self.GP_z.X, self.GP_z.X), H ), \
			self.GP_z.kernelz(self.GP_z.Z, self.GP_z.Z)), H) ) ) * np.dot(np.dot(H , \
			self.GP_z.kernelx(self.GP_z.X, self.GP_z.X)), H), e)) for e in matrix_grads]
		return np.array(grads) + xx
		
	def optimise_latents(self):
		"""Direct optimisation of the latents variables."""
		ztemp = np.zeros(self.GP_z.Z.shape)
		for i,yy in enumerate(self.GP_z.XY):
			original_z = self.GP_z.Z[i].copy()
			zopt = fmin_cg(self.ll,self.GP_z.Z[i],fprime=self.ll_grad,disp=True,args=(i,))

			self.GP_z.Z[i] = original_z
			ztemp[i] = zopt
		self.GP_z.Z = ztemp.copy()


class GPLVM_zin_lnhsic:
	def __init__(self,X,Y,Z):
		self.Zdim = Z.shape[1]
		self.N,self.Ydim = Y.shape
		self.Xdim = X.shape[1]
		self.XY = np.concatenate((X, Y), axis = 1)
		self.lda_2 = 1.0

		"""Use PCA to initalise the problem. Uses EM version in this case..."""
		self.Z = Z
		self.GP_z = GP_z.GP_z(Z, X, Y)#choose particular kernel here if so desired.

	def updateM(self):
		vals, vecs = eigsh(self.GP_z.kernelz(self.GP_z.Z, self.GP_z.Z), k=2, which = 'LM')
		self.M = np.real(vecs)

	def learn(self,niters):
		for i in range(niters):
			self.optimise_latents()
			self.optimise_GP_kernel()
			
	def optimise_GP_kernel(self):
		"""optimisation of the GP_z's kernel parameters"""
		self.GP_z.find_kernel_params()
		print self.GP_z.marginal(), 0.5*np.sum(np.square(self.GP_z.X))
	
	def ll(self,xx,i):
		"""The log likelihood function - used when changing the ith latent variable to xx"""
		H = np.identity(self.N) - np.ones((self.N, self.N), dtype = float) / self.N

		self.GP_z.Z[i] = xx
		self.GP_z.update()
		return -self.GP_z.marginal()+ 0.5*np.sum(np.square(xx)) + \
			self.lda_2 * np.log( np.trace( np.dot(np.dot(np.dot(self.GP_z.kernelx(self.GP_z.X, self.GP_z.X), H ), self.GP_z.kernelz(self.GP_z.Z, self.GP_z.Z)), H) ) )
	
	def ll_grad(self,xx,i):
		"""the gradient of the likelihood function for us in optimisation"""
		H = np.identity(self.N) - np.ones((self.N, self.N), dtype = float) / self.N

		self.GP_z.Z[i] = xx
		self.GP_z.update()
		self.GP_z.update_grad()
		matrix_grads = [self.GP_z.kernelz.gradients_wrt_data(self.GP_z.Z,i,jj) for jj in range(self.GP_z.Zdim)]
		grads = [-0.5*np.trace(np.dot(self.GP_z.alphalphK + \
			self.lda_2 * ( 1/np.trace( np.dot(np.dot(np.dot(self.GP_z.kernelx(self.GP_z.X, self.GP_z.X), H ), \
			self.GP_z.kernelz(self.GP_z.Z, self.GP_z.Z)), H) ) ) * np.dot(np.dot(H , \
			self.GP_z.kernelx(self.GP_z.X, self.GP_z.X)), H), e)) for e in matrix_grads]
		return np.array(grads) + xx
		
	def optimise_latents(self):
		"""Direct optimisation of the latents variables."""
		ztemp = np.zeros(self.GP_z.Z.shape)
		for i,yy in enumerate(self.GP_z.XY):
			original_z = self.GP_z.Z[i].copy()
			zopt = fmin_cg(self.ll,self.GP_z.Z[i],fprime=self.ll_grad,disp=True,args=(i,))

			self.GP_z.Z[i] = original_z
			ztemp[i] = zopt
		self.GP_z.Z = ztemp.copy()


class GPPOMC_lnhsic(GPLVM_z_lnhsic):
	"""A(back) constrained version """
	def __init__(self, X, Y, Zdim, lda, nhidden=5, mlp_alpha=2):
		GPLVM_z_lnhsic.__init__(self, X, Y, Zdim)
		
		self.MLP = MLP.MLP((self.Xdim + self.Ydim ,nhidden, self.Zdim ),alpha=mlp_alpha)
		self.MLP.train(self.GP_z.XY, self.GP_z.Z)#create an MLP initialised to the PCA solution
		self.GP_z.Z = self.MLP.forward(self.GP_z.XY)
		self.lda_2_mlp = lda
		
	def unpack(self,w):
		""" Unpack the np array into the free variables of the current instance"""
		assert w.size == self.MLP.nweights + self.GP_z.kernelz.nparams + self.GP_z.kernelx.nparams + 1,"bad number of parameters for unpacking"
		self.MLP.unpack(w[:self.MLP.nweights])
		self.GP_z.Z = self.MLP.forward(self.GP_z.XY)
		self.GP_z.set_params(w[self.MLP.nweights:])
	
	def updateM(self):
		vals, vecs = eigsh(self.GP_z.kernelz(self.GP_z.Z, self.GP_z.Z), k=2, which = 'LM')
		self.M = np.real(vecs)

	def pack(self):
		""" 'Pack up' all of the free variables in the model into a np array"""
		return np.hstack((self.MLP.pack(),self.GP_z.get_params()))
		
	def ll(self,w):
		"""Calculate and return the -ve log likelihood of the model (actually, the log probabiulity of the model). To be used in optimisation routine"""
		self.unpack(w)
		self.GP_z.update()
		self.updateM()
		H = np.identity(self.N) - np.ones((self.N, self.N), dtype = float) / self.N

		return  self.GP_z.ll() + 0.5*np.sum(np.square(self.GP_z.Z)) - self.MLP.prior() + \
				self.lda_2_mlp * np.log( np.trace( np.dot(np.dot(np.dot(self.GP_z.kernelx(self.GP_z.X, self.GP_z.X), H ), self.GP_z.kernelz(self.GP_z.Z, self.GP_z.Z)), H) ) )

	def ll_grad(self,w):
		"""The gradient of the ll function - used for quicker optimisation via fmin_cg"""
		self.unpack(w)
		H = np.identity(self.N) - np.ones((self.N, self.N), dtype = float) / self.N

		GP_grads = self.GP_z.ll_grad(w[self.MLP.nweights:])
		
		gradient_matrices = self.GP_z.kernelz.gradients_wrt_data(self.GP_z.Z)
		
		dldtheta = self.GP_z.alphalphK * self.GP_z.kernelx(self.GP_z.X, self.GP_z.X)

		Z_gradients = np.array([-0.5* np.trace(np.dot( (dldtheta - \
			self.lda_2_mlp * ( 1/np.trace( np.dot(np.dot(np.dot(self.GP_z.kernelx(self.GP_z.X, self.GP_z.X), H ), \
			self.GP_z.kernelz(self.GP_z.Z, self.GP_z.Z)), H) ) ) * np.dot(np.dot(H , \
			self.GP_z.kernelx(self.GP_z.X, self.GP_z.X)), H)).T, e)) for e in gradient_matrices]).reshape(self.GP_z.Z.shape) + \
			self.GP_z.Z

		# backpropagate...
		weight_gradients = self.MLP.backpropagate(self.GP_z.XY,Z_gradients) - self.MLP.prior_grad()
		return np.hstack((weight_gradients,GP_grads))
		
	def learn(self,callback=None,gtol=1e-4, maxiter = 1000):
		"""'Learn' by optimising the weights of the MLP and the GP_z hyper parameters together.  """
		w_opt = fmin_cg(self.ll, np.hstack((self.MLP.pack(), self.GP_z.kernelz.get_params(), self.GP_z.kernelx.get_params(), np.log(self.GP_z.beta))), self.ll_grad, args=(), callback=callback,gtol=gtol, maxiter = maxiter, disp=1)
		final_cost = self.ll(w_opt) # sets all the parameters...