import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pymc3 as pm

import functions as ff
import data_handling as dh

var_names_oocyte = 'Vmax_o a_o t_o nu_o Vmin_o sdo_o'.split()
var_names_nurse  = 'Vmax_n a0 t0 nu0 a1 t1 nu1 Vmin_n sdo_n'.split()
var_names = var_names_oocyte + var_names_nurse

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def perc_plot( ax, x, samp,mask=None, **kwargs  ):
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if( mask is None): mask = range(len(x))
    percs = np.percentile( samp, [2.5,25,50,75,97.5], axis=0)
    ax.fill_between( x[mask], percs[0,mask],percs[4,mask], alpha=0.2, **kwargs )
    ax.fill_between( x[mask], percs[1,mask],percs[3,mask], alpha=0.5, **kwargs )
    ax.plot( x[mask], percs[2,mask], lw=1.5, **kwargs )
    return

def fit( filename, string ):
	#var_names_oocyte = 'Vmax_o a_o t_o nu_o Vmin_o'.split()
	#var_names_nurse  = 'Vmax_n a0 t0 nu0 a1 t1 nu1 Vmin_n'.split()
	#var_names = var_names_oocyte + var_names_nurse
	try:
		res_df = pd.read_pickle('res.pkl')
	except:
		#results_summ = var_names='Vmax_o a_o t_o nu_o Vmin_o Vmax_n a0_n t0_n nu0_n a1_n t1_n nu1_n Vmin_n'.split()
		col_ind = pd.MultiIndex.from_product( [var_names, 'm lo hi'.split()] )
		res_df =pd.DataFrame(columns=col_ind)


	# create the code for the sample
	code=(filename.split('.')[0]).split('/')[-1]+"_{}".format(string)
	df =pd.read_csv(filename)
	strings = df['string'].unique()
	if( string in strings ):
		print("processing chain {} of {}".format(string,strings))
	else:
		print("Error: string out of range")
		return

	# unpack the data
	yo = dh.unpack_results(df,string,'o',volume=False)
	yn = dh.unpack_results(df,string,'n',volume=False)
	xo = np.arange(len(yo))
	xn = np.arange(len(yn))

 	# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	# fit oocyte model
	with pm.Model() as fit_o:

		Vmax_o   =  pm.Lognormal('Vmax_o',   mu=np.log(150),sd=0.5)
		t_o      =  pm.Normal('t_o',        mu=6  ,sd=6)
		
		a_o      = pm.Deterministic('a_o',-1*pm.HalfNormal('a_nt',sd=2))
		nu_o     =  pm.Lognormal('nu_o',   mu=np.log(1), sd=0.25)
		Vmin_o   =  pm.Lognormal('Vmin_o',   mu=np.log( 15),sd=0.6)

		f        = pm.Deterministic('f',ff.rise_only(xo,Vmax_o,a_o,t_o,nu_o )+Vmin_o )

		sdo_o = pm.HalfCauchy('sdo_o', 20.)
		yobs = pm.Normal('yobs',f,sd=sdo_o, observed = yo)	
	# fit the model
	with fit_o:
		 trace_o=pm.sample()

	# take some samples from the postier
	nsamp=2000
	xop = np.linspace(0,len(yo),101)

	samps_o = np.zeros([nsamp,len(xop)])
	pulls = np.array([trace_o.get_values(var) for var in var_names_oocyte ]).T
	for i in range(nsamp):
		samps_o[i] = ff.rise_only(xop,*pulls[i,:-1])+pulls[i,-1]

	# - - - - - - - - - - - - - - - - - - - - -
	# specify the nurse model
	with pm.Model() as fit_n:
		Vmax_n  = pm.Lognormal('Vmax_n', mu=np.log(100),sd=0.25)
		Vmin_n  =  pm.Lognormal('Vmin_n',   mu=np.log(  5),sd=0.6)
		a0    =pm.Deterministic('a0',-1*pm.HalfNormal('a0_t',sd=3))
		t0     =  pm.Normal('t0',        mu=6  ,sd=4)
		nu0 = pm.Lognormal('nu0',    mu=np.log(1), sd=0.25 )

		a1    = pm.Deterministic('a1',-1*pm.HalfNormal('a1_t',sd=1))
		t1  = pm.HalfNormal('t1',    sd=2)
		nu1 = pm.Lognormal('nu1',   mu=np.log(1), sd=0.25 )


		fn    = pm.Deterministic('f', ff.rise_and_fall(xn, Vmax_n, a0,t0,nu0, a1,t1,nu1)+Vmin_n)

		sdo_n = pm.HalfCauchy('sdo_n', 20.)
		yobs_n = pm.Normal('yn',fn,sd=sdo_n, observed = yn)

	# fit the nurse model
	with fit_n:
		trace_n = pm.sample(start={'a1':0})

	# get some samples
	xnp = np.linspace(0,len(yn),101)
	samps_n = np.zeros([nsamp,len(xnp)])
	pulls = np.array([trace_n.get_values(var) for var in var_names_nurse ]).T
	for i in range(nsamp):
		samps_n[i] = ff.rise_and_fall(xnp,*pulls[i,:-1])+pulls[i,-1]

	# create a plot
	fig,ax = plt.subplots(1,1)

	perc_plot(ax, xop,samps_o, color='Red')
	ax.plot(xo,yo,'o',color='Black')


	perc_plot(ax, xnp,samps_n, color='Blue')
	ax.plot(xn,yn,'x',color='Black')
	plt.title(code)
	plt.savefig("figs/fit1_{}.png".format(code))

	# save the result
	for names,trace in [[var_names_oocyte,trace_o],[var_names_nurse,trace_n]]:
		for var_name in names:
			vals = trace.get_values(var_name)
			res_df.loc[code,(var_name,['m'])]=np.median(vals)
			res_df.loc[code,(var_name,['lo','hi'])]=pm.hpd(vals)	
	res_df.to_pickle("res.pkl")
