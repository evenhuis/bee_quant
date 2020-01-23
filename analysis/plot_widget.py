
from ipywidgets import widgets,interactive,interact
from ipywidgets import Layout, Button, Box, FloatText, Textarea, Dropdown, Label, FloatSlider
from IPython.display import display,clear_output
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import data_handling as dh
import functions as ff
global o_obs,o_x

def show_plot():
    plt.close('all')
    plt.ioff()
    fig,ax = plt.subplots(1,1,figsize=(8,6))
    out=widgets.Output()

    res_df = pd.read_pickle("res.pkl")
    def click(b):
        global o_obs,o_x, n_obs,n_x

        Vmax_o = Vmax_o_slide.value
        ao   = ao_slide.value
        to   = to_slide.value
        
        Vmax_n = Vmax_n_slide.value
        a0   = a0_slide.value
        t0   = t0_slide.value
        a1   = a1_slide.value
        t1   = t1_slide.value
        #plt.figure(2)

        
        ax.clear()
        ax.plot(-o_x,o_obs,'o',color='red')
        ax.plot(-n_x,n_obs,'o',color='blue')
        if( True):
            x = np.linspace(-8, 16, num=100)
            ax.plot(-x,ff.rise_only    (x,Vmax_o,ao,-to,1.0),color='red')
            ax.plot(-x,ff.rise_and_fall(x,Vmax_n,a0,-t0,1.0,a1,-t1,1),color='blue')
            ax.plot([to,to],[0,ff.rise_only(-to,Vmax_o,ao,-to,1.0)],ls='--',color='red',lw=1.5)
            
            ax.plot([t0,t0],[0,ff.rise_and_fall(-t0,Vmax_n,a0,-t0,1.0,a1,-t1,1)],ls='--',color='blue',lw=1.5)
            ax.plot([t1,t1],[0,ff.rise_and_fall(-t1,Vmax_n,a0,-t0,1.0,a1,-t1,1)],ls='--',color='blue',lw=1.5)
            
            
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

    choice=Dropdown(
        options='cont_r1 cont_r2 cont_r3 caged_d04_r1 caged_d07_r1'.split(),
        value='cont_r1',
        description='Number:',
        disabled=False,
    )
    
    

    
    Vmax_o_slide=FloatSlider(description=r'$V$max$_o$',value=150,min=0,max=300,continuous_update=False)
    Vmax_o_slide.observe(click, names='value')
    ao_slide  =FloatSlider(description=r'$a_o$',value=-0.4,min=-1.5,max=0,continuous_update=False)
    ao_slide.observe(click, names='value')
    to_slide  =FloatSlider(description=r'$t_o$',value=0,min=-4,max=4,continuous_update=False)
    to_slide.observe(click, names='value')
    
    Vmax_n_slide=FloatSlider(description=r'$V$max$_{n}$',value=150,min=0,max=300,continuous_update=False)
    Vmax_n_slide.observe(click, names='value')
    a0_slide  =FloatSlider(description=r'$a_0$',value=-0.4,min=-1.5,max=0,continuous_update=False)
    a0_slide.observe(click, names='value')
    t0_slide  =FloatSlider(description=r'$t_0$',value=-4,min=-8,max=4,continuous_update=False)
    t0_slide.observe(click, names='value')    
    
    a1_slide  =FloatSlider(description=r'$a_1$',value=-0.4,min=-1.5,max=0,continuous_update=False)
    a1_slide.observe(click, names='value')
    t1_slide  =FloatSlider(description=r'$t_1$',value=0,min=-4,max=4,continuous_update=False)
    t1_slide.observe(click, names='value')       
    
    
    def choice_selected(b):
        global o_obs,o_x, n_obs,n_x
        name = choice.value
        df=pd.read_csv("results_analyse/{}.csv".format(name))
        o_obs=dh.unpack_results(df,1,'o',volume=False)
        o_x = np.arange(len(o_obs))
        n_obs=dh.unpack_results(df,1,'n',volume=False)
        n_x = np.arange(len(n_obs))
        
        rown = "{}_1".format(name)
        Vmax_n_slide.value= res_df.loc[rown,('Vmax_n','m')]
        a0_slide.value= res_df.loc[rown,('a0','m')]
        t0_slide.value= -res_df.loc[rown,('t0','m')]
        a1_slide.value= res_df.loc[rown,('a1','m')]
        t1_slide.value= -res_df.loc[rown,('t1','m')]
        
        Vmax_o_slide.value= res_df.loc[rown,('Vmax_o','m')]
        ao_slide.value= res_df.loc[rown,('a_o','m')]
        to_slide.value= -res_df.loc[rown,('t_o','m')]
        click(None)
        #f(Vmax_slide.value, a0_slide.value, t0_slide.value)
        return
    choice_selected(None)
    choice.observe(choice_selected)
        
    
    
    #display(VBox([mslide,cslide]))
    oocyte_params=widgets.VBox([Label(value="Oocyte"),Vmax_o_slide,ao_slide,to_slide])
    nurse_params =widgets.VBox([Vmax_n_slide,a0_slide,t0_slide,a1_slide,t1_slide])
    box = widgets.VBox([choice,widgets.HBox([oocyte_params,nurse_params]),out])
    display(box)
    
    
    click(None)
    #interact(f,Vmax=Vmax_slide,a0=a0_slide,t0=t0_slide);


