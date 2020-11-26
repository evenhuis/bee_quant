import pymc3 as pm
import theano.tensor as tt
import numpy as np

import pymc3_utils as pmu
import bee_util  as bu
import indexTools

import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Functions
#	Definitions of functions used in the fitting
def logit(p):
	return np.log(p/(1-p))
def invlogit(a):
	return 1./(1.+np.exp(-a))

def oocycte_model_p0( df ):
	xo = df['pos'].values.copy()
	yo = df['Oc_size'].values.copy()   

	ind,_ = pmu.multilabel_encoder(df,['i_ind'])
	nind = len(np.unique(ind))

	trt,_ = pmu.multilabel_encoder(df,['i_treatday'])
	ntrt = len(np.unique(trt))

	trt_in_ind = pmu.embeded_index(df,'i_ind', 'i_treatday')

	with pm.Model() as oc_p0_mod:

		# single
		Vmax_o_mu = pm.Normal('Vmax_o_mu',mu=np.log(160),sd=0.15)
		Vmax_o_l = pm.Deterministic( 'Vmax_o_l',Vmax_o_mu*tt.ones(nind))
		Vmax_o	= pm.Deterministic('Vmax_o',tt.exp(Vmax_o_l))
		
		r_o_mu  = pm.Normal     ('r_o_mu', mu=np.log(3),sd=0.25,shape=nind)	# location of grand mean
		r_o_l	= pm.Deterministic('r_o_l',r_o_mu)
		r_o	 = pm.Deterministic('r_o',tt.exp(r_o_l) )
			
		t_o_mu = pm.Normal	  ('t_o_mu',mu=3,sd=4,shape=nind)
		t_o = pm.Deterministic	 ('t_o',t_o_mu)
		 
		Vmin_o_mu = pm.Normal('Vmin_o_mu',mu=np.log(10),sd=0.25,shape=nind)
		Vmin_o = pm.Deterministic('Vmin_o',tt.exp(Vmin_o_mu))

		nu_o= pm.Deterministic('nu_o',tt.ones(nind))
		
		f=bu.rise_only_r( xo, Vmax_o[ind], t_o[ind], r_o[ind], nu_o[ind], Vmin_o[ind])
									 
		sdo_o = pm.Lognormal('sdo_o', mu=np.log(5.),sd=0.25)
		yobs = pm.Normal('yobs',f,sd=sdo_o, observed = yo)  
	return oc_p0_mod


def nurse_model_p0( df ):
	xo = df['pos'].values.copy()
	yo = df['Ns_size'].values.copy()   

	ind,_ = pmu.multilabel_encoder(df,['i_ind'])
	nind = len(np.unique(ind))

	trt,_ = pmu.multilabel_encoder(df,['i_treatday'])
	ntrt = len(np.unique(trt))

	trt_in_ind = pmu.embeded_index(df,'i_ind', 'i_treatday')

	with pm.Model() as oc_p0_mod:

		# single
		Vmax_n_mu = pm.Normal       ('Vmax_n_mu',mu=np.log(80),sd=0.15)
		Vmax_n_l  = pm.Deterministic('Vmax_n_l',Vmax_n_mu*tt.ones(nind))
		Vmax_n	 = pm.Deterministic ('Vmax_n',tt.exp(Vmax_n_l))
		
		r_n_mu  = pm.Normal        ('r_n_mu', mu=np.log(3),sd=0.25,shape=nind)	# location of grand mean
		r_n_l	= pm.Deterministic ('r_n_l',r_n_mu)
		r_n	 = pm.Deterministic    ('r_n',tt.exp(r_n_l) )
			
		t_n_mu = pm.Normal	   	   ('t_n_mu',mu=3,sd=3,shape=nind)
		t_n = pm.Deterministic	   ('t_n',t_n_mu)

		r_d_h = 0.5
		r_d_mu  = pm.Normal        ('r_d_mu', mu=np.log(r_d_h),sd=0.10,shape=nind)  # location of grand mean
		r_d_l   = pm.Deterministic ('r_d_l',r_d_mu)
		r_d     = pm.Deterministic ('r_d',tt.exp(r_d_l) )

		t_d_mu = pm.HalfNormal     ('t_d_mu',sd=2*r_d_h,shape=nind)
		t_d    = pm.Deterministic  ('t_d',  -1*r_d_h+t_d_mu)
		 
		Vmin_n_mu = pm.Normal       ('Vmin_n_mu',mu=np.log(10),sd=0.15,shape=nind)
		Vmin_n    = pm.Deterministic('Vmin_n',tt.exp(Vmin_n_mu))

		nu_n= pm.Deterministic('nu_n',tt.ones(nind))
		nu_d= pm.Deterministic('nu_d',tt.ones(nind))
		
		f=bu.rise_and_fall_r( xo, Vmax_n[ind], t_n[ind], r_n[ind], nu_n[ind], \
                                               t_d[ind], r_d[ind], nu_d[ind], Vmin_n[ind])
									 
		sdo_o = pm.Lognormal('sdo_o', mu=np.log(5.),sd=0.25)
		yobs = pm.Normal('yobs',f,sd=sdo_o, observed = yo)  
	return oc_p0_mod

def nurse_model_p1( df ):
	xo = df['pos'].values.copy()
	yo = df['Ns_size'].values.copy()   

	ind,_ = pmu.multilabel_encoder(df,['i_ind'])
	nind = len(np.unique(ind))

	trt,_ = pmu.multilabel_encoder(df,['i_treatday'])
	ntrt = len(np.unique(trt))

	trt_in_ind = pmu.embeded_index(df,'i_ind', 'i_treatday')

	with pm.Model() as ns_p1_mod:

		# single
		Vmax_n_mu = pm.Normal       ('Vmax_n_mu',mu=np.log(80),sd=0.15)
		Vmax_n_l  = pm.Deterministic('Vmax_n_l',Vmax_n_mu*tt.ones(nind))
		Vmax_n	 = pm.Deterministic ('Vmax_n',tt.exp(Vmax_n_l))
		
		r_n_mu  = pm.Normal        ('r_n_mu', mu=np.log(3),sd=0.25)	# location of grand mean
		r_n_sd  = pm.HalfNormal    ('r_n_sd', sd = 0.1)
		r_n_vn  = pm.Normal        ('r_n_vn', mu=0, sd=1,shape=nind)
		r_n_l	= pm.Deterministic ('r_n_l',r_n_mu+r_n_sd*r_n_vn)
		r_n	 = pm.Deterministic    ('r_n',tt.exp(r_n_l) )
			
		t_n_mu = pm.Normal	   	   ('t_n_mu',mu=3,sd=3)
		t_n_sd = pm.HalfNormal     ('t_n_sd', sd = 3)
		t_n_vn = pm.Normal         ('t_n_vn',mu=0,sd=1,shape=nind)
		t_n = pm.Deterministic	   ('t_n',t_n_mu+t_n_sd*t_n_vn)

		r_d_mu  = pm.Normal        ('r_d_mu', mu=np.log(0.85),sd=0.20)  # location of grand mean
		r_d_sd  = pm.HalfNormal    ('r_d_sd', sd =0.25 )
		r_d_vn  = pm.Normal        ('r_d_vn', mu=0,sd=1,shape=nind)
		r_d_l   = pm.Deterministic ('r_d_l',r_d_mu+r_d_sd*r_d_vn)
		r_d     = pm.Deterministic ('r_d',tt.exp(r_d_l) )

		t_d_mu = pm.Normal         ('t_d_mu',mu=np.log(1),sd=0.25)
		t_d_sd = pm.HalfNormal     ('t_d_sd',sd=0.25)
		t_d_vn = pm.Normal         ('t_d_vn',mu=0,sd=1,shape=nind)
		t_d    = pm.Deterministic  ('t_d', -2.50+tt.exp(t_d_mu+t_d_sd*t_d_vn))
		 
		Vmin_n_mu = pm.Normal       ('Vmin_n_mu',mu=np.log(10),sd=0.25)
		Vmin_n    = pm.Deterministic('Vmin_n',tt.exp(Vmin_n_mu)*tt.ones(nind))

		nu_n= pm.Deterministic('nu_n',tt.ones(nind))
		nu_d= pm.Deterministic('nu_d',tt.ones(nind))
		
		f=bu.rise_and_fall_r( xo, Vmax_n[ind], t_n[ind], r_n[ind], nu_n[ind], \
                                               t_d[ind], r_d[ind], nu_d[ind], Vmin_n[ind])
									 
		sdo_o = pm.Lognormal('sdo_o', mu=np.log(5.),sd=0.25)
		yobs = pm.Normal('yobs',f,sd=sdo_o, observed = yo)  
	return ns_p1_mod

def oocycte_model_p1( df ):
	xo = df['pos'].values.copy()
	yo = df['Oc_size'].values.copy()   

	ind,_ = pmu.multilabel_encoder(df,['i_ind'])
	nind = len(np.unique(ind))

	trt,_ = pmu.multilabel_encoder(df,['i_treatday'])
	ntrt = len(np.unique(trt))

	trt_in_ind = pmu.embeded_index(df,'i_ind', 'i_treatday')

	with pm.Model() as oc_p1_mod:
	

		# single
		Vmax_o_mu = pm.Normal('Vmax_o_mu',mu=np.log(150),sd=0.15)
		Vmax_o_l = pm.Deterministic( 'Vmax_o_l',Vmax_o_mu*tt.ones(nind))
		Vmax_o	= pm.Deterministic('Vmax_o',tt.exp(Vmax_o_l))
		
		
		vdict={}
		if( True ):
			r_o_mu  = pm.Normal ('r_o_mu', mu=np.log(3),sd=0.15)	# location of grand mean
			r_o_sd  = pm.HalfNormal('r_o_sd',sd=0.2)
			
			r_o_vn  = pm.Normal('r_o_vn',mu=0,sd=1,shape=nind)
			r_o_l	= pm.Deterministic('r_o_l',r_o_mu+r_o_sd*r_o_vn)
			r_o	 = pm.Deterministic('r_o',tt.exp(r_o_l) )
			
		else:
			add_2level( vdict,'a_o_l', trt_in_ind, np.log(0.5),0.2, 0.2, 0.2 )
			a_o = pm.Deterministic('a_o',tt.exp(vdict['a_o_l']) )
			

		if( True ):
			 t_o_mu = pm.Normal	  ('t_o_mu',mu=3,sd=6)
			 t_o_sd = pm.HalfNormal  ('t_o_sd',sd=6)

			 t_o_vn = pm.Normal	  ('t_o_vn',mu=0,sd=1,shape=nind)

			 t_o = pm.Deterministic	 ('t_o',t_o_mu+t_o_sd*t_o_vn)

		else:
			 add_2level( vdict,'t_o', trt_in_ind, 2,4, 4, 1 )
			 t_o = vdict['t_o']
		 #a_o = pm.Deterministic('a_o',tt.exp(vdict['a_o_l']) )	
		 
		if( False):
			Vmin_o_mu = pm.Normal('Vmin_o_mu',mu=np.log(10),sd=0.2)
			Vmin_o_sd = pm.HalfNormal('Vmin_o_sd',sd=0.1)
			
			Vmin_o_vn = pm.Normal('Vmin_o_vn',mu=0,sd=1,shape=nind)
			Vmin_o_l = pm.Deterministic('Vmin_o_l',Vmin_o_mu+Vmin_o_sd*Vmin_o_vn)

			#Vmin_o_l = pm.Normal('Vmin_o_l',mu=Vmin_o_mu,sd=Vmin_o_sd,shape=nind)
			Vmin_o	= pm.Deterministic('Vmin_o',tt.exp(Vmin_o_l))

		if(True):
			Vmin_o_mu = pm.Normal('Vmin_o_mu',mu=np.log(10),sd=0.2)
			Vmin_o	= pm.Deterministic('Vmin_o',tt.exp(Vmin_o_mu)*tt.ones(nind))
		nu_o= pm.Deterministic('nu_o',tt.ones(nind))
		
		f=bu.rise_only_r( xo, Vmax_o[ind], t_o[ind], r_o[ind], nu_o[ind], Vmin_o[ind])
									 
		sdo_o = pm.Lognormal('sdo_o', mu=np.log(5.),sd=0.25)
		yobs = pm.Normal('yobs',f,sd=sdo_o, observed = yo)  
	return oc_p1_mod

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
		Vmax_o_l  = pm.Normal( 'Vmax_o_l',mu=np.log(150),sd=0.2)  # hyper prior for grand mean
		Vmax_o	 = pm.Deterministic('Vmax_o',tt.exp(Vmax_o_l))  

		pmu.add_2level_hierarchy('a_o_l',vdict, np.log(0.5),0.2, 0.1,0.1, ind_in_treatday )
		a_o		= pm.Deterministic('a_o',tt.exp(vdict['a_o_l']))	 

		pmu.add_2level_hierarchy('t_o_l',vdict, 0,6, 4,4, ind_in_treatday )
		t_o		 = pm.Deterministic('t_o',vdict['t_o_l'])

		pmu.add_1level_hierarchy('Vmin_o_l',vdict, np.log(10), 0.3, 0.1, ind )
		Vmin_o	= pm.Deterministic('Vmin_o',tt.exp(vdict['Vmin_o_l']))

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
		Vmax_n_l=pm.Normal( 'Vmax_n_l',mu=np.log(90),sd=0.4)  # hyper prior for grand mean  
		#Vmax_n_l=pmu.add_1level_hierarchy('Vmax_n_l',vdict,np.log(90), 0.2, 0.2, nind)
		Vmax_n = pm.Deterministic('Vmax_n',tt.exp(Vmax_n_l))		  

		pmu.add_2level_hierarchy('a0_l',vdict, np.log(0.5),0.2, 0.1,0.1, ind_in_treatday )
		#a0_l = pm.Normal('a0_l',np.log(0.5),0.4,shape=nind)
		a0		= pm.Deterministic('a0',tt.exp(vdict['a0_l']))		  
	 
		pmu.add_2level_hierarchy('t0_l',vdict, np.log(5),0.2,0.1,0.1, ind_in_treatday )
		#t0_l = pm.Normal('t0_l',mu=6,sd=4,shape=nind)
		t0	= pm.Deterministic('t0',tt.exp(vdict['t0_l'])-2)

		
		#a1_l	 = pmu.add_2level_hierarchy('a1_l',vdict, np.log(1.5),0.2, 0.075,0.075, ind_in_treatday )
		pmu.add_1level_hierarchy('a1_l',vdict, 0,3, 1, range(nind))
		#a1_l  =pm.TruncatedNormal('a1_l', mu=np.log(1.5),sd=0.2,upper=np.log(4),shape=nind) 
		#a1_l = pm.Normal('a1_l',np.log(1),0.4,shape=nind)
		a1 = pm.Deterministic( 'a1',0.25+4.25*invlogit(vdict['a1_l']))
		#a1		= pm.Deterministic('a1',tt.exp(a1_l))		  
	 
		pmu.add_1level_hierarchy('t1_l',vdict, -1,6,4, range(nind) )
		#t1_l = pm.Normal('t1_l',mu=-2,sd=4,shape=nind)
		t1	= pm.Deterministic('t1',vdict['t1_l'])
	
		pmu.add_1level_hierarchy('Vmin_n_l',vdict, np.log(10), 0.4, 0.05, range(nind) )
		#Vmin_n_l = pm.Normal( 'Vmin_n_l',mu=np.log(10), sd=0.4,shape=nind)
		Vmin_n	= pm.Deterministic('Vmin_n',tt.exp(vdict['Vmin_n_l']))
		
		nu0 = pm.Deterministic('nu0',tt.ones(nind))
		nu1 = pm.Deterministic('nu1',tt.ones(nind))
	
		fn	 = bu.rise_and_fall(xn, Vmax_n, t0[ind],a0[ind],nu0[ind], \
														 t1[ind],a1[ind],nu1[ind],Vmin_n[ind]) 
		
		sdo_n = pm.HalfCauchy('sdo_n', 20.)
		yobs_n = pm.Normal('yn',fn,sd=sdo_n, observed = yn)
	return fit_n
