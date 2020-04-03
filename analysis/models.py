import pymc3 as pm
import theano.tensor as tt
import numpy as np

import pymc3_utils as pmu
import bee_util  as bu
import indexTools

def oocyte_model( df):

	xo = df['pos'].values.copy()
	yo = df['Oc_size'].values.copy()
	
	ind = df['i_ind']
	nind = len(np.unique(ind))
	
	treatday = df['i_treatday']
	ntreatday = len(np.unique(treatday))
	
	ind_in_treatday = indexTools.embeded_index(df,'i_ind','i_treatday')




	with pm.Model() as fit_o:
		vdict = {}
		Vmax_o_l  = pm.Normal( 'Vmax_o_l',mu=np.log(150),sd=0.5)  # hyper prior for grand mean
		Vmax_o	 = pm.Deterministic('Vmax_o',tt.exp(Vmax_o_l))  

		a_o_l	 = pmu.add_2level_hierarchy('a_o_l',vdict, np.log(0.5),0.3, 0.2,0.2, ind_in_treatday )
		a_o		= pm.Deterministic('a_o',tt.exp(a_o_l))	 

		t_o_l	  = pmu.add_2level_hierarchy('t_o_l',vdict, 0,6, 4,4, ind_in_treatday )
		t_o		 = pm.Deterministic('t_o',t_o_l)

		Vmin_o_l = pmu.add_1level_hierarchy('Vmin_o_l',vdict, np.log(10), 0.3, 0.1, nind )
		Vmin_o	= pm.Deterministic('Vmin_o',tt.exp(Vmin_o_l))

		#nu_o = pm.Lognormal('nu_o', mu=0.,sd=0.1,shape=nind)
		nu_o = pm.Deterministic('nu_o',tt.ones(nind))

		f = bu.rise_only(xo,Vmax_o,t_o[ind],a_o[ind],nu_o[ind],Vmin_o[ind])

		sdo_o = pm.Lognormal('sdo_o', mu=np.log(5.),sd=0.25)
		yobs = pm.Normal('yobs',f,sd=sdo_o, observed = yo)
	return fit_o


def nurse_model( df ):
	xn = df['pos'].values.copy()
	yn = df['Ns_size'].values.copy()
	
	ind = df['i_ind']
	nind = len(np.unique(ind))
	
	treatday = df['i_treatday']
	ntreatday = len(np.unique(treatday))

	ind_in_treatday = indexTools.embeded_index(df,'i_ind','i_treatday')


	with pm.Model() as fit_n:
		vdict = {}
		Vmax_n_l  = pm.Normal( 'Vmax_n_l',mu=np.log(90),sd=0.4)  # hyper prior for grand mean  
		#Vmax_n_l=pmu.add_1level_hierarchy('Vmax_n_l',vdict,np.log(90), 0.2, 0.2, nind)
		Vmax_n = pm.Deterministic('Vmax_n',tt.exp(Vmax_n_l))		  

		a0_l	 = pmu.add_2level_hierarchy('a0_l',vdict, np.log(0.5),0.2, 0.1,0.1, ind_in_treatday )
		#a0_l = pm.Normal('a0_l',np.log(0.5),0.4,shape=nind)
		a0		= pm.Deterministic('a0',tt.exp(a0_l))		  
	 
		t0_l = pmu.add_2level_hierarchy('t0_l',vdict, np.log(5),0.2,0.1,0.1, ind_in_treatday )
		#t0_l = pm.Normal('t0_l',mu=6,sd=4,shape=nind)
		t0	= pm.Deterministic('t0',tt.exp(t0_l)-2)

		
		#a1_l	 = pmu.add_2level_hierarchy('a1_l',vdict, np.log(1.5),0.2, 0.075,0.075, ind_in_treatday )
		a1_l	 = pmu.add_1level_hierarchy('a1_l',vdict, 0,3, 1, nind)
		#a1_l  =pm.TruncatedNormal('a1_l', mu=np.log(1.5),sd=0.2,upper=np.log(4),shape=nind) 
		#a1_l = pm.Normal('a1_l',np.log(1),0.4,shape=nind)
		a1 = pm.Deterministic( 'a1',0.25+4.25*pmu.invlogit(a1_l))
		#a1		= pm.Deterministic('a1',tt.exp(a1_l))		  
	 
		t1_l = pmu.add_1level_hierarchy('t1_l',vdict, -1,6,4, nind )
		#t1_l = pm.Normal('t1_l',mu=-2,sd=4,shape=nind)
		t1	= pm.Deterministic('t1',t1_l)
	
		Vmin_n_l = pmu.add_1level_hierarchy('Vmin_n_l',vdict, np.log(10), 0.4, 0.05, nind )
		#Vmin_n_l = pm.Normal( 'Vmin_n_l',mu=np.log(10), sd=0.4,shape=nind)
		Vmin_n	= pm.Deterministic('Vmin_n',tt.exp(Vmin_n_l))
		
		nu0 = pm.Deterministic('nu0',tt.ones(nind))
		nu1 = pm.Deterministic('nu1',tt.ones(nind))
	
		fn	 = bu.rise_and_fall(xn, Vmax_n, t0[ind],a0[ind],nu0[ind], \
														 t1[ind],a1[ind],nu1[ind],Vmin_n[ind]) 
		
		sdo_n = pm.HalfCauchy('sdo_n', 20.)
		yobs_n = pm.Normal('yn',fn,sd=sdo_n, observed = yn)
	return fit_n
