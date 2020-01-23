

import numpy as np

def unpack_results( df, istring, code, volume=True):

    '''Unpacks the oocyte and nurse cell data from a df'''
    istrings = df['string'].unique()
    if( istring not in istrings):
        return 'Nothing here'

    mask = (df['string']==istring)&(df['code']==code)
    if( volume ):
        return df.loc[mask,'Volume'].values
    else:
        return np.power(df.loc[mask,'Volume'].values*3/(4*np.pi),1./3.)
