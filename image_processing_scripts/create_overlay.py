
def main():
	from ij import IJ
	from ij.plugin.frame import RoiManager
	from ij.gui import Overlay,TextRoi
	from java.awt import Color,Font
	istring=2

	imp = IJ.getImage()
	ov = Overlay()
	
	roim = RoiManager.getInstance()


	ctypes='o n'.split()
	colors= [Color.RED,Color.BLUE]
	font = Font("Helvetica", Font.PLAIN, 24)
	for ii,i in enumerate(range(0,roim.getCount())):
		roi = roim.getRoi(i)
		if( roi is not None ):
			name = "{}-{}-{}".format(istring,ii+1,ctypes[i%2])
			name = roi.getName()
			roi.setStrokeColor(colors[i%2])
			roi.setStrokeWidth(2)
			

			tmp = name.split('-')
			string = int(tmp[0])
			name_short = tmp[1]+'-'+tmp[2]
			#roim.rename(i,name)
			#roim.setRoi(roi,i)

			if( True or (string==istring) ):
				ov.add(roi)
				
				stats = roi.getStatistics()
				yoff=20
				text=TextRoi(stats.xCentroid,stats.yCentroid-yoff,name,font)
				print(name)
				text.setStrokeColor(colors[i%2])
				ov.add(text)
			
	imp.setOverlay(ov)

if( __name__ in "__main__ __builtin__".split()):
	main()