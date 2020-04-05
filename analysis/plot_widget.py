
from ipywidgets import widgets,interactive,interact
from ipywidgets import Layout, Button, Box, FloatText, Textarea, Dropdown, Label, FloatSlider
from IPython.display import display,clear_output
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import functions as ff
import bee_util as bu
import pymc3 as pm
import models
import indexTools
global o_obs,o_x, df, trace_o, trace_n

def show_plot():
	plt.close('all')
	plt.ioff()


	# load in the trace
	from glob import glob
	import os
	global df,trace_o,trace_n
	df = pd.read_csv('merged_data.csv')

	# read in the models
	fit_o = models.oocyte_model(df)
	fit_n = models.oocyte_model(df)

	# read in the traces
	trace_o = pm.load_trace('trace_o',fit_o)
	trace_n = pm.load_trace('trace_n',fit_n)


	fig,ax = plt.subplots(1,1,figsize=(8,6))
	out=widgets.Output()

	#res_df = pd.read_pickle("res.pkl")
	def click(b):
		global o_obs,o_x, n_obs,n_x

		Vmax_o = Vmax_o_slide.value
		ao   = ao_slide.value
		to   = to_slide.value
		Vmin_o = Vmin_o_slide.value
		
		Vmax_n = Vmax_n_slide.value
		a0   = a0_slide.value
		t0   = t0_slide.value
		a1   = a1_slide.value
		t1   = t1_slide.value
		Vmin_n = Vmin_n_slide.value
		#plt.figure(2)

		
		ax.clear()
		ax.plot(-o_x,o_obs,'o',color='red')
		ax.plot(-n_x,n_obs,'o',color='blue')
		if( True):
			x = np.linspace(-2, 16, num=100)
			ax.plot(-x,bu.rise_only	   (x,Vmax_o,to,ao,1.0,        Vmin_o),color='red')
			ax.plot(-x,bu.rise_and_fall(x,Vmax_n,t0,a0,1.0,t1,a1,1,Vmin_n),color='blue')

			ax.plot([-to,-to],[0,bu.rise_only    (to,Vmax_o,to,ao,1.0,        Vmin_o)],ls='--',color='red',lw=1.5)
			ax.plot([-t0,-t0],[0,bu.rise_and_fall(t0,Vmax_n,t0,a0,1.0,t1,a1,1,Vmin_n)],ls='--',color='blue',lw=1.5)
			ax.plot([-t1,-t1],[0,bu.rise_and_fall(t1,Vmax_n,t0,a0,1.0,t1,a1,1,Vmin_n)],ls='--',color='blue',lw=1.5)
			
			
			ax.set_xticks(range(-14,2,2))
			ax.set_ylim(0, 200)
			ax.axhline(Vmax_o,ls='--',color='red')
			ax.axvline(0,ls='--',color='grey',lw=1)
			
			[xmin,xmax]=ax.get_xlim()
			[ymin,ymax]=ax.get_ylim()
			ax.annotate(r'$t_o$',(to+0.2,ymax-10))
			ax.annotate(r'$Vmax_{o}}$',(xmax-2,Vmax_o+10))
			


		with out:
			clear_output(wait=True)
			display(ax.figure)

	#choice=Dropdown(
	#	
	#	options='cont_r1 cont_r2 cont_r3 caged_d04_r1 caged_d07_r1'.split(),
	#	value='cont_r1',
	#	description='Number:',
	#	disabled=False,
	#)
	choice=widgets.IntText(
		value=0,
		min=0,
		max=len(df['i_ind'].unique()),
		step=1,
		description='Test:',
		disabled=False,
		continuous_update=False,
		readout=True,
		readout_format='d'
	)
	
	

	
	Vmax_o_slide=FloatSlider(description=r'$V$max$_o$',value=150,min=0,max=300,continuous_update=False)
	Vmax_o_slide.observe(click, names='value')
	Vmin_o_slide=FloatSlider(description=r'$V$min$_o$',value=15 ,min=0,max=30 ,continuous_update=False)
	Vmin_o_slide.observe(click, names='value')
	ao_slide  =FloatSlider(description=r'$a_o$',value=0.2,min=0.,max=0.75,continuous_update=False)
	ao_slide.observe(click, names='value')
	to_slide  =FloatSlider(description=r'$t_o$',value=1,min=-2,max=6,continuous_update=False)
	to_slide.observe(click, names='value')
	
	Vmax_n_slide=FloatSlider(description=r'$V$max$_{n}$',value=150,min=0,max=300,continuous_update=False)
	Vmax_n_slide.observe(click, names='value')
	Vmin_n_slide=FloatSlider(description=r'$V$min$_n$',value=15 ,min=0,max=30 ,continuous_update=False)
	Vmin_n_slide.observe(click, names='value')
	a0_slide  =FloatSlider(description=r'$a_0$',value= 0.4,min=0.0 ,max=1.5,continuous_update=False)
	a0_slide.observe(click, names='value')
	t0_slide  =FloatSlider(description=r'$t_0$',value=0,min=-4,max=8,continuous_update=False)
	t0_slide.observe(click, names='value')	
	
	a1_slide  =FloatSlider(description=r'$a_1$',value=0.4,min=0.0 ,max=8,continuous_update=False)
	a1_slide.observe(click, names='value')
	t1_slide  =FloatSlider(description=r'$t_1$',value=0.5,min=-2,max=6,continuous_update=False)
	t1_slide.observe(click, names='value')	   
	
	
	def choice_selected(b):
		global o_obs,o_x, n_obs,n_x, df, trace_o, trace_n
		if( False ):
			name = choice.value
			df=pd.read_csv("results_analyse/{}.csv".format(name))
			o_obs=dh.unpack_results(df,1,'o',volume=False)
			o_x = np.arange(len(o_obs))
			n_obs=dh.unpack_results(df,1,'n',volume=False)
			n_x = np.arange(len(n_obs))
		else:
			iexp = choice.value
			mask = df['i_ind']==iexp	
			if( sum(mask)>0 ):	
			   o_obs = df.loc[mask,'Oc_size'] ; o_x = df.loc[mask,'pos'] 
			   n_obs = df.loc[mask,'Ns_size'] ; n_x = o_x

			   vars_o = 'Vmax_o,t_o,a_o,Vmin_o'.split(',')
			   vars_n = 'Vmax_n t0 a0 t1 a1 Vmin_n'.split()
			   theta_o = np.median(bu.pull_post(trace_o,vars_o,iexp),axis=0)
			   theta_n = np.median(bu.pull_post(trace_n,vars_n,iexp),axis=0)
			   for slide,val in zip( [Vmax_o_slide,to_slide,ao_slide,Vmin_o_slide],theta_o):
				   slide.value=val
			   for slide,val in zip( [Vmax_n_slide,t0_slide,a0_slide,t1_slide,a1_slide,Vmin_n_slide],theta_n):
				   slide.value=val			   
		
		#rown = "{}_1".format(name)
		#Vmax_n_slide.value= res_df.loc[rown,('Vmax_n','m')]
		#a0_slide.value= res_df.loc[rown,('a0','m')]
		#t0_slide.value= -res_df.loc[rown,('t0','m')]
		#a1_slide.value= res_df.loc[rown,('a1','m')]
		#t1_slide.value= -res_df.loc[rown,('t1','m')]
		
		#Vmax_o_slide.value= res_df.loc[rown,('Vmax_o','m')]
		#ao_slide.value= res_df.loc[rown,('a_o','m')]
		#to_slide.value= -res_df.loc[rown,('t_o','m')]
		click(None)
		#f(Vmax_slide.value, a0_slide.value, t0_slide.value)
		return
	choice_selected(None)
	choice.observe(choice_selected)
		
	
	
	#display(VBox([mslide,cslide]))
	oocyte_params=widgets.VBox([Label(value="Oocyte"),Vmax_o_slide,ao_slide,to_slide,Vmin_o_slide])
	nurse_params =widgets.VBox([Vmax_n_slide,a0_slide,t0_slide,a1_slide,t1_slide,Vmin_n_slide])
	box = widgets.VBox([choice,widgets.HBox([oocyte_params,nurse_params]),out])
	display(box)
	
	
	click(None)
	#interact(f,Vmax=Vmax_slide,a0=a0_slide,t0=t0_slide);


