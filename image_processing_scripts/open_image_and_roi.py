from ij import IJ

def open_czi(filepath,series):
    from loci.plugins import BF
    from loci.plugins.in import ImporterOptions
    from loci.common import Region
    options = ImporterOptions()
    options.setId( filepath )
    options.setSeriesOn(series-1,True)
    imp = BF.openImagePlus(options)[0]

    cal=imp.getCalibration()

    cal.pixelWidth  = cal.pixelWidth*2 **(series)
    cal.pixelHeight = cal.pixelHeight*2**(series)
    return imp    

def open_files_and_roi(filepath):
	imp=open_czi(filepath,2)
	imp.show()

	import os
	from ij.plugin.frame import RoiManager
	roifile = os.path.splitext(filepath)[0]+".zip"
	roim = RoiManager.getInstance()
	if(roim is None): roim = RoiManager()
	roim.reset()
	roim.runCommand("Open",roifile)
	print(roifile)
	return imp
	
def main():
	from ij.io import OpenDialog
	od=OpenDialog("Select a slide scanner file")

	filepath = od.getPath()
	print(filepath)
	
	open_files_and_roi(filepath)

if( __name__ in '__builtin__ __main__'.split()):
	main()