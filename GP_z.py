# -*- coding: utf-8 -*-

"""
GPPOM python implementation (modified from https://github.com/jameshensman/pythonGPLVM)
(Anaconda4.3.30 64-bit for python 2.7.14 on Windows 10)

Shoubo Hu (shoubo.sub AT gmail.com)
13/09/2018
"""

import numpy as np
from scipy.optimize import fmin_cg
from scipy import linalg
import kernels 

class GP_z:
	def __init__(self, Z, X, Y,kernelz=None, kernelx=None,parameter_priors=None):
		""" a simple GP with optimisation of the Hyper parameters via the marginal likelihood approach.  
		There is a Univariate Gaussian Prior on the Hyper parameters (the kernel parameters and the noise parameter). 
		SCG is used to optimise the parameters (MAP estimate)"""
		self.N = Y.shape[0]
		self.setX(X)
		self.setY(Y)

		self.setZ(Z)
		self.setXY(self)

		if kernelz==None:
			self.kernelz = kernels.RBF_full(-1,-np.ones(self.Zdim))
		else:
			self.kernelz = kernelz
		if kernelx==None:
			self.kernelx = kernels.RBF_full(-1,-np.ones(self.Xdim))
		else:
			self.kernelx = kernelx
		
		if parameter_priors==None:
			self.parameter_prior_widths = np.ones(self.kernelz.nparams + self.kernelx.nparams+1)
		else:
			assert parameter_priors.size==(self.kernelz.nparams + self.kernelx.nparams +1)
			self.parameter_prior_widths = np.array(parameter_priors).flatten()
		self.beta=0.1
		self.update()
		self.n2ln2pi = 0.5*self.Ydim*self.N*np.log(2*np.pi) # constant in the marginal. precompute for convenience. 

	def setZ(self,newZ):
		self.Z = newZ.copy()
		N,self.Zdim = newZ.shape
		assert N == self.N, "bad shape"
		# normalize...
		self.zmean = self.Z.mean(0)
		self.zstd = self.Z.std(0)
		self.Z -= self.zmean
		self.Z /= self.zstd

	def setX(self,newX):
		self.X = newX.copy()
		N,self.Xdim = newX.shape
		assert N == self.N, "bad shape"
		# normalize...
		self.xmean = self.X.mean(0)
		self.xstd = self.X.std(0)
		self.X -= self.xmean
		self.X /= self.xstd

	def setY(self,newY):
		self.Y = newY.copy()
		N,self.Ydim = newY.shape
		assert N == self.N, "bad shape"
		# normalize...
		self.ymean = self.Y.mean(0)
		self.ystd = self.Y.std(0)
		self.Y -= self.ymean
		self.Y /= self.ystd

	def setXY(self,newZ):
		self.XY = np.concatenate((self.X, self.Y), axis = 1)


	def hyper_prior(self):
		"""return the log of the current hyper paramters under their prior"""
		return -0.5*np.dot(self.parameter_prior_widths,np.square(self.get_params()))
	
	def hyper_prior_grad(self):
		"""return the gradient of the (log of the) hyper prior for the current parameters"""
		return -self.parameter_prior_widths*self.get_params()
		
	def get_params(self):
		"""return the parameters of this GP: that is the kernel parameters and the beta value"""
		# return np.hstack((self.kernel.get_params(),np.log(self.beta)))
		return np.hstack((self.kernelz.get_params(), self.kernelx.get_params(), np.log(self.beta)))

	def set_params(self,params):
		""" set the kernel parameters and the noise parameter beta"""
		assert params.size == self.kernelz.nparams + self.kernelx.nparams + 1
		self.beta = np.exp(params[-1])
		self.kernelz.set_params(params[:self.kernelz.nparams])
		self.kernelx.set_params(params[self.kernelz.nparams:(self.kernelx.nparams + self.kernelz.nparams)])

	def ll(self,params=None):
		"""  A cost function to optimise for setting the kernel parameters. Uses current parameter values if none are passed """
		if params is not None:
			self.set_params(params)
		try:
			self.update()
		except:
			return np.inf
		return -self.marginal() - self.hyper_prior()
		
	def ll_grad(self,params=None):
		""" the gradient of the ll function, for use with conjugate gradient optimisation. uses current values of parameters if none are passed """
		if params is not None:
			self.set_params(params)
		try:
			self.update()
		except:
			return np.ones(params.shape)*np.NaN
		self.update_grad()

		matrix_grads = [self.kernelx(self.X, self.X) * e for e in self.kernelz.gradients(self.Z)] + [self.kernelz(self.Z, self.Z) * e for e in self.kernelx.gradients(self.X)]
		matrix_grads.append(-np.eye(self.K.shape[0])/self.beta) #noise gradient matrix

		grads = [0.5*np.trace(np.dot(self.alphalphK,e)) for e in matrix_grads]
			
		return -np.array(grads) - self.hyper_prior_grad()
		
	def find_kernel_params(self,iters=1000):
		"""Optimise the marginal likelihood. work with the log of beta - fmin works better that way.  """
		new_params = fmin_cg(self.ll,np.hstack((self.kernelz.get_params(), self.kernelx.get_params(), np.log(self.beta))),fprime=self.ll_grad,maxiter=iters)
		final_ll = self.ll(new_params) # sets variables - required!
		
	def update(self):
		"""do the Cholesky decomposition as required to make predictions and calculate the marginal likelihood"""
		self.K = self.kernelx(self.X,self.X) * self.kernelz(self.Z, self.Z)
		self.K += np.eye(self.K.shape[0])/self.beta
		self.L = np.linalg.cholesky(self.K + 1e-2 *np.eye(self.K.shape[0]))
		self.A = linalg.cho_solve((self.L,1),self.Y)
	
	def update_grad(self):
		"""do the matrix manipulation required in order to calculate gradients"""
		self.Kinv = np.linalg.solve(self.L.T,np.linalg.solve(self.L,np.eye(self.L.shape[0])))
		self.alphalphK = np.dot(self.A,self.A.T)-self.Ydim*self.Kinv
		
	def marginal(self):
		"""The Marginal Likelihood. Useful for optimising Kernel parameters"""
		return -self.Ydim*np.sum(np.log(np.diag(self.L))) - 0.5*np.trace(np.dot(self.Y.T,self.A)) - self.n2ln2pi

	def predict(self,x_star):
		"""Make a prediction upon new data points"""
		x_star = (np.asarray(x_star)-self.xmean)/self.xstd

		k_x_star_x = self.kernel(x_star,self.X) 
		k_x_star_x_star = self.kernel(x_star,x_star) 
		
		#find the means and covs of the projection...
		means = np.dot(k_x_star_x, self.A)
		means *= self.ystd
		means += self.ymean
		
		v = np.linalg.solve(self.L,k_x_star_x.T)
		variances = (np.diag( k_x_star_x_star - np.dot(v.T,v)).reshape(x_star.shape[0],1) + 1./self.beta) * self.ystd.reshape(1,self.Ydim)
		return means,variances

if __name__=='__main__':
	#generate data:
	Ndata = 50
	X = np.linspace(-3,3,Ndata).reshape(Ndata,1)
	Y = np.sin(X) + np.random.standard_normal(X.shape)/20
	Z = 0.5 * np.random.rand(X.shape[0], 1)

	myGP = GP_z(Z, X, Y)
	print myGP.get_params()
	myGP.find_kernel_params()
	print myGP.get_params()




