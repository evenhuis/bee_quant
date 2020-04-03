import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def multilabel_encoder(df,cols,delim='_'):
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    '''Creates a unique index for a set of cols in a dataframe
    A generalisation of sklearn LabelEncoder
    input: 
      df - dataframe
      cols - list of column names 
    '''
    # string merge the col to create a hashable set of cols
    mrg_lab=df[cols[0]].astype(str)
    for col in cols[1:]:
        mrg_lab+=delim+df[col].astype(str)

    # create a dictionary for cols with cols as key and 0..N and values
    unique_lab=mrg_lab.unique()
    ucode=dict(zip(unique_lab,range(len(unique_lab))))
    return [ ucode[v] for v in mrg_lab],mrg_lab # reverse encode cols


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def embeded_index( df, col1, col2):
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    '''returns the index arrays that surjectivily maps col1 into col2
    * col1 must have more distinct elements than col2
    * each unique element col1 must to 1 element of col2
    * every element of col2 mapped ()
    '''
    # check that there are less elements in col2
    ncol1 = len(df[col1].unique())
    ncol2 = len(df[col2].unique())          
    if( ncol1 < ncol2):
        raise Exception('col {} has more elements than col {}'.format(col1,col2))
    
    # make unique combinations
    delim='_##_'
    i_lab, u_lab = multilabel_encoder(df,[col1,col2],delim=delim)
    if( ncol1< len(u_lab.unique())):
        print( u_lab)
        raise Exception('col1 maps to multiple elements in col2')    
    # get the list of unique label mappings
    KV = np.array([list(map(int,lab.split(delim))) for lab in u_lab.unique() ]).T
    
    # just in case col1 is not sorted
    isort = np.argsort(KV[0])
    return KV[1,isort]
    
