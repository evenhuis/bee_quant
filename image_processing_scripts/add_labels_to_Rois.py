def main():
	from ij import IJ
	from ij.plugin.frame import RoiManager

	istring=0
	area_last=0
	jj=0

	
	roim = RoiManager.getInstance()
	istart = roim.getSelectedIndex()
	#istart =0

	ctypes='o n'.split()
	for ii,i in enumerate(range( istart , roim.getCount())):
		print(ii)
		roi = roim.getRoi(i)
		if( roi is not None ):
			ctype=ctypes[i%2]
			if( ctype=='o'):
				area_curr = roi.getStatistics().area
				if( area_curr > 4*area_last ):
					istring += 1
					jj=0
				area_last = area_curr
			jj+=1
			name = "{}-{}-{}".format(istring,jj,ctype)
			roi.setName(name)
			roi.setStrokeWidth(1)
			
			roim.rename(i,name)
			roim.setRoi(roi,i)

			
if( __name__ in "__main__ __builtin__".split()):
	main()