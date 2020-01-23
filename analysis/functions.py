import numpy as np
'''
Functions used the bee cell size quantification
'''


def gen_log( t, alpha,t0,nu):
   """Generalised logistc function

   [-int,+inf] -> [0, 1]
   monotonic increasing function

   Arguments:
   t     -- function argument
   alpha -- slope of the function at t=t0
   t0    -- location of midpoint f(t0) == 0.5
   nu    -- controls speed the function approahces 0
   """
   Q=-1+(1/0.5)**nu
   #Q=1
   return 1/(1+ Q*np.exp(-alpha*(t-t0)) )**(1/nu)


def rise_only( x, Vmax, a0,t0,nu0 ):
	'''Describes the growth of cell sizes'''
	return Vmax*gen_log(x,a0,t0,nu0)

def rise_and_fall( x, Vmax, a0,t0,nu0, a1,t1,nu1 ):
	"""
	Describes a growth and decay function, suitable for describing nurse cells
	"""
	return Vmax * gen_log(x,a0,t0,nu0) * (1-gen_log(x,a1,t1,nu1))

