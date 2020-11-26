import numpy as np

#--------------------------------------------------------------------------------
# Trace operations

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def pull_post(  trace,varnames, nind, nsamp=100):
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	''' get samples from the trace object corresponding to entry in the vector variable
	trace: the trace object
	varnames : [string] a list of the strings of the variable names
	nind	  : index of the variable to extract
	nsamp	 : number of samples to draw

	If the variable is not a vector i; is ignored 
	'''
	post = np.zeros([nsamp,len(varnames)])

	# create the random sampling array
	vals  = trace.get_values(varnames[0])
	isamp = np.random.randint(vals.shape[0],size=nsamp)

	for i,var in enumerate(varnames):
		vals = trace.get_values(var)
		if( len((vals.shape))==1 ):
			post[:,i]=vals[isamp]
		if( len((vals.shape))==2 ):
			post[:,i]=vals[isamp,nind]
		
	return post

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def compare_percentiles( df, trace, varnames, nsamp=1000):
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	''' compare experimental unit level parameters in a simple box and whisher plot
	'''
	import matplotlib.pyplot as plt
	fig,axs = plt.subplots(len(varnames),1,figsize=(10,8),sharex=True)

	for j,col in enumerate(varnames):
		ax=axs[j]
		treatdays = 'cont_0 caged_4 caged_7 caged_10 banked_10'.split()
		cdict = dict(zip(treatdays, 'blue green orange red grey'.split()))

		istart = 0
		treatday_prev = None
		for i in df['i_ind'].unique():

			mask = df['i_ind']==i
			dft = df[mask]
			treatday = dft['treatday'].values[0]
			if( i==0): 
				treatday_prev = treatday
				ip=0
			if( treatday!=treatday_prev):
				if(ip%2==0): ax.axvspan(istart,i,alpha=0.1,color='grey')
				istart=i
				ip=ip+1
				treatday_prev = treatday

			vals = pull_post(trace,[col],i, nsamp=nsamp)
			ax.plot( [i,i],np.percentile(vals,[ 2.5,97.5]),color=cdict[treatday],lw=1)
			ax.plot( [i,i],np.percentile(vals,[25., 50. ]),color=cdict[treatday],lw=4)
		if(ip%2==0): ax.axvspan(istart,i,alpha=0.1,color='grey')
		ax.set_ylabel(col)

		# make the xtics
		if( col=='t_o'):
			locs=[]
			for treatday in treatdays:
				mask = df['treatday']==treatday
				dft=df[mask]
				locs.append(np.average(dft['i_ind'].unique()))
			#ax.set_xticks(locs)
			#ax.set_xticklabels(treatdays)
			#for i in np.unique(ind):
			#	ax.annotate("{}".format(i),(i,0.7+(i%4)/4.*0.2),xycoords=('data','axes fraction'))

def compare_groups(df, trace ,varnames, vref=None, exp=False):
	import matplotlib.pyplot as plt
	nvar = len(varnames)
	treatdays = 'cont_0 caged_4 caged_7 caged_10 banked_10'.split()
	cdict = dict(zip(treatdays, 'blue green orange red grey'.split()))
	fig,axs = plt.subplots(nvar,1,figsize=(6,8),sharex=True)

	exp_val= np.zeros(nvar,dtype=bool)
	exp_val[:]=exp

	for j,col in enumerate(varnames):
		ax = axs[j]

		for i,treatday in enumerate(treatdays):

			mask = df['treatday']==treatday
			dft = df[mask]

			i_treatday =  dft['i_treatday'].values[0]
			pulls=pull_post(trace,[col],i_treatday)
			if( (vref is not None) and (vref[j] is not None ) ):
				pulls = pulls + pull_post(trace,vref[j],i_treatday)
			if( exp_val[j]): pulls = np.exp(pulls)
			ax.plot( [i,i],np.percentile(pulls,[2.5,97.5]),color=cdict[treatday] )
			ax.plot( [i,i],np.percentile(pulls,[25,75]),color=cdict[treatday] ,lw=4)
			if(i%2==0 ): ax.axvspan(i-0.5,i+0.5,alpha=0.1,color='grey')
		ax.set_ylabel(col)

	ax.set_xticks(range(len(treatdays)));
	ax.set_xticklabels(treatdays);

def compare_groups2(df, trace ,varnames, imask=None):
	import matplotlib.pyplot as plt
	import indexTools
	nvar = len(varnames)
	treatdays = 'cont_0 caged_4 caged_7 caged_10 banked_10'.split()
	cdict = dict(zip(treatdays, 'blue green orange red grey'.split()))
	fig,axs = plt.subplots(nvar,1,figsize=(6,8),sharex=True)


	ind_in_treatday = indexTools.embeded_index(df,'i_ind','i_treatday')

	for j,col in enumerate(varnames):
		ax = axs[j]
		pulls = trace.get_values(col)
		for i,treatday in enumerate(treatdays):

			mask = df['treatday']==treatday
			dft = df[mask]
			i_treatday =  dft['i_treatday'].values[0]

			mask = ind_in_treatday==i_treatday
			if( imask is not None ):
			   mask = np.logical_and(mask,  imask)
			samp = pulls[:,mask].flatten()
			ax.plot( [i,i],np.percentile(samp,[2.5,97.5]),color=cdict[treatday] )
			ax.plot( [i,i],np.percentile(samp,[25,75]),color=cdict[treatday] ,lw=4)
			if(i%2==0 ): ax.axvspan(i-0.5,i+0.5,alpha=0.1,color='grey')
		ax.set_ylabel(col)

	ax.set_xticks(range(len(treatdays)));
	ax.set_xticklabels(treatdays);

#--------------------------------------------------------------------------------
# functions

def sigmoid_curve_db( x, th, td, nu):
	 Q=-1+(1/0.5)**nu
	#return					  1/(1+Q*np.exp(alpha*(np.log(t+1e-18)-np.log(t0  ))))**(1./nu)
	 #return 1./(1+Q*np.exp((x-th)*td ))**(1./nu)
	 return 1./(1.+Q*np.exp((x-th)*np.log(2)/td) )**(1./nu)
def sigmoid_curve( x, th, a, nu):
	 Q=-1+(1/0.5)**nu
	#return					  1/(1+Q*np.exp(alpha*(np.log(t+1e-18)-np.log(t0  ))))**(1./nu)
	 #return 1./(1+Q*np.exp((x-th)*td ))**(1./nu)
	 return 1./(1.+Q*np.exp((x-th)*a  ) )**(1./nu)		# the x2 looks like a mistake

def rise_only(x,Vmax_o,t_o,a_o,nu_o,Vmin_o):
	 return (Vmax_o-Vmin_o)*sigmoid_curve(x,t_o,a_o,nu_o)+Vmin_o

def rise_only_r( x, Vmax_o,t_o,r_o,nu_o,Vmin_o):
   	return (Vmax_o-Vmin_o)*sigmoid_curve(x,t_o,2.*np.log(3)/r_o,nu_o)+Vmin_o


def rise_and_fall		( x,Vmax, t0,a0,nu0, t1,a1,nu1, Vmin):
	 return	 rise_only( x,Vmax, t0,a0,nu0,Vmin)*(1.-sigmoid_curve(x,t1,a1,nu1))

def rise_and_fall_r     ( x,Vmax, t0,r0,nu0, t1,r1,nu1, Vmin):
     return  rise_only( x,Vmax, t0,2.*np.log(3)/r0,nu0,Vmin)*(1.-sigmoid_curve(x,t1,2.*np.log(3)/r1,nu1))

def rise_and_fall_vol_trans( x,Vmax, t0,a0,nu0, t1,a1,nu1, Vmin):
	 tol = t1+1.*a
	 p=3
	 return (rise_and_fall( np.where(x<tol,tol,x),Vmax, t0,a0,nu0, t1,a1,nu1, Vmin)**p
				-rise_and_fall( x,Vmax, t0,a0,nu0, t1,a1,nu1, Vmin)**p)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def perc_plot( ax, x, samp,mask=None, **kwargs  ):
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	if( mask is None): mask = range(len(x))
	percs = np.percentile( samp, [2.5,25,50,75,97.5], axis=0)
	ax.fill_between( x[mask], percs[0,mask],percs[4,mask], alpha=0.2, **kwargs )
	ax.fill_between( x[mask], percs[1,mask],percs[3,mask], alpha=0.5, **kwargs )
	ax.plot( x[mask], percs[2,mask], lw=1.5, **kwargs )
	return

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def posterior_draws( trace, ind,code, X, samp_err=False,nsamp=1000 ):
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	'''draw samples from the posterior and apply the model for oocyte or nurse 
	cells. 
	input
	   trace  :  MCMC chain from PyMC3
	   code   :  'o'/'n'  o for oocyte, n for nurse
	   X      :  array of positions
	   nsamp  :  numper of draws

	output
       draws  : nparray [nsamp,len(X)]		

	'''
	if( code=='o'): 
		cols = 'Vmax_o,t_o,a_o,nu_o,Vmin_o'.split(',')
		func = rise_only
	if( code=='n'): 
		cols = 'Vmax_n t0 a0 nu0 t1 a1 nu1 Vmin_n sdo_n'.split()
		func = rise_and_fall
	pulls = pull_post(trace,cols,ind,nsamp=nsamp)
	samps = np.zeros([nsamp,len(X)])
	for i in range(nsamp):
		samps[i] = func(X, *pulls[i,:-1] )
	if( samp_err ):
		for i in range(nsamp):
			samps[i]+= np.random.rand(len(X))*pulls[i,-1]
		
		
	return samps	

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def posterior_draws_r( trace, ind,code, X, samp_err=False,nsamp=1000 ):
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	'''draw samples from the posterior and apply the model for oocyte or nurse 
	cells. 
	input
	   trace  :  MCMC chain from PyMC3
	   code   :  'o'/'n'  o for oocyte, n for nurse
	   X      :  array of positions
	   nsamp  :  numper of draws

	output
       draws  : nparray [nsamp,len(X)]		

	'''
	if( code=='o'): 
		cols = 'Vmax_o t_o r_o nu_o Vmin_o sdo_o'.split()
		func = rise_only_r
	if( code=='n'): 
		cols = 'Vmax_n t_n r_n nu_n t_d r_d nu_d Vmin_n sdo_n'.split()
		func = rise_and_fall_r
	pulls = pull_post(trace,cols,ind,nsamp=nsamp)
	samps = np.zeros([nsamp,len(X)])
	for i in range(nsamp):
		samps[i] = func(X, *pulls[i,:-1] )
	if( samp_err ):
		for i in range(nsamp):
			samps[i]+= np.random.rand(len(X))*pulls[i,-1]
		
	return samps	

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def posterior_comp( ax, df, trace, ind, code='o', ptype='spag', color='red', nsamp=None, index='i_ind', **kwargs):
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	''''''
	if( code=='o'): cols = 'Vmax_o,t_o,a_o,nu_o,Vmin_o'.split(',')	
	if( code=='n'): cols = 'Vmax_n t0 a0 nu0 t1 a1 nu1 Vmin_n'.split()


	if( (ptype=='spag') & (nsamp is None) ) : nsamp=50
	if( (ptype=='perc') & (nsamp is None) ) : nsamp=400
	if( (ptype=='median') & (nsamp is None) ) : nsamp=100
	
	#cols = 'Vmax_n t0 a0 nu0 t1 a1 nu1 Vmin_n'.split() Nurse
	# get the experimental unit
	pulls = pull_post(trace,cols,ind, nsamp=nsamp)

	nmax = max(df.loc[df[index]==ind,'pos'])+2
	
	xp = np.linspace(-2,nmax,(nmax+4)*4+1)	# xaxis for plotting
	if(ptype=='median'): 
	   xp = np.linspace(0,nmax,(nmax+4)*4)

	samps = np.zeros([nsamp,len(xp)])

	if(code=='o'):
		for i in range(nsamp):
			samps[i] = rise_only(xp, *pulls[i] )
	if(code=='n'):
		for i in range(nsamp):
			samps[i] = rise_and_fall(xp, *pulls[i] )		
	if( ptype=='spag'): 
		for i in range(nsamp):
			ax.plot( -xp, samps[i],color=color,lw=1,alpha=0.25)
	if( ptype=='perc'): perc_plot(ax,-xp,samps,color=color)
	if( ptype=='median'): 
		theta = np.median(pulls,axis=0)
		if( code=='o'):
			ax.plot(-xp,rise_only(xp,*theta),color=color,**kwargs)
		if( code=='n'):
			ax.plot(-xp,rise_and_fall(xp,*theta),color=color,**kwargs)

		
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def posterior_comp_r( ax, df, trace, ind, code='o', ptype='spag', color='red', nsamp=None, index='i_ind', **kwargs):
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	''''''
	if( code=='o'): cols = 'Vmax_o t_o r_o nu_o Vmin_o'.split()	
	if( code=='n'): cols = 'Vmax_n t_n r_n nu_n t_d r_d nu_d Vmin_n'.split()


	if( (ptype=='spag') & (nsamp is None) ) : nsamp=50
	if( (ptype=='perc') & (nsamp is None) ) : nsamp=400
	if( (ptype=='median') & (nsamp is None) ) : nsamp=100
	
	#cols = 'Vmax_n t0 a0 nu0 t1 a1 nu1 Vmin_n'.split() Nurse
	# get the experimental unit
	pulls = pull_post(trace,cols,ind, nsamp=nsamp)

	nmax = max(df.loc[df[index]==ind,'pos'])+2
	
	xp = np.linspace(-2,nmax,(nmax+4)*4+1)	# xaxis for plotting
	if(ptype=='median'): 
	   xp = np.linspace(0,nmax,(nmax+4)*4+1)

	samps = np.zeros([nsamp,len(xp)])

	if(code=='o'):
		for i in range(nsamp):
			samps[i] = rise_only_r(xp, *pulls[i] )
	if(code=='n'):
		for i in range(nsamp):
			samps[i] = rise_and_fall_r(xp, *pulls[i] )		
	if( ptype=='spag'): 
		for i in range(nsamp):
			ax.plot( -xp, samps[i],color=color,lw=1,alpha=0.25)
	if( ptype=='perc'): perc_plot(ax,-xp,samps,color=color)
	if( ptype=='median'): 
		theta = np.median(pulls,axis=0)
		if( code=='o'):
			ax.plot(-xp,rise_only_r(xp,*theta),color=color,**kwargs)
		if( code=='n'):
			ax.plot(-xp,rise_and_fall_r(xp,*theta),color=color,**kwargs)

		
