import numpy as np
import pandas as pd

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def unpack_results( df, istring, code, volume=True):
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	'''Unpacks the oocyte and nurse cell volume/radius from a df
	 istring : the string index (there are multiple string is a data set
	 code	: o for oocyte , n for nurse
	 volume  : if false calcualte the effective radius 
	 '''
	istrings = df['string'].unique()
	if( istring not in istrings):
		return 'Nothing here'

	mask = (df['string']==istring)&(df['code']==code)
	if( volume ):
		return df.loc[mask,'Volume'].values
	else:
		return np.power(df.loc[mask,'Volume'].values*3/(4*np.pi),1./3.)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def read_data( files ):
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -	
	import os
	df = pd.DataFrame(columns='treat day rep string pos Oc_size Ns_size'.split())

	for file in files:
		# split the filename up to work out the treat, rep and day
		filename=os.path.split(file)[1]
		parts=filename.split('_')

		treat = parts[0]
		rep   = parts[-1].split('.')[0][1:]
		if( parts[0]=='cont'):
			day=0
		else:
			day=int(parts[1][1:])
			  
		dft = pd.read_csv(file)
		strings = dft['string'].unique()
		for i in strings:
			r0 = len(df)
			Oc_sizes = unpack_results(dft,i,'o',volume=False)
			Ns_sizes = unpack_results(dft,i,'n',volume=False)
			for j,(Oc,Ns) in enumerate(zip(Oc_sizes,Ns_sizes)):
				df.loc[r0+j]=[treat,day,rep,i,j,Oc,Ns]

	# convert indices to integer
	df['pos']=df['pos'].astype(int)
	df['rep']=df['rep'].astype(int)

	# correct the size on some of the images
	if( True):
		for treat,day,rep in zip( 'caged caged caged'.split(),[4,10,10],[3,2,4]):
			mask=(df['treat']==treat)&(df['day']==day)&(df['rep']==rep)
			df.loc[mask,'Oc_size']=df.loc[mask,'Oc_size']/2.
			df.loc[mask,'Ns_size']=df.loc[mask,'Ns_size']/2.

	return df
