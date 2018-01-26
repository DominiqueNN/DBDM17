import sqlite3 as lite
import csv
import pyfits
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn import mixture
from matplotlib.colors import LogNorm
from sklearn.neighbors import KernelDensity

def maketables(con):	
	con.execute("""CREATE TABLE IF NOT EXISTS InfoTable (FieldID INT,
		Filename NOT NULL, Filter varchar(1), MJD FLOAT);""")
	
	with open('Tables/file_info_for_problem.csv') as csvfile:
		readCSV = csv.reader(csvfile, delimiter=',')
		next(readCSV)
		for row in readCSV:
			command = "INSERT INTO InfoTable Values("+row[1]+",'"+row[2]+"','"+row[3]+"',"+row[4]+");"
			con.execute(command)
	
	for filt in ["Y","Z","J","H","Ks"]:
		con.execute("""CREATE TABLE IF NOT EXISTS DataTable{0} (FieldID INT, 
		Flux1 FLOAT, dFlux1 FLOAT, Mag1 FLOAT, dMag1 FLOAT, StarID INT);""".format(filt))
		fitslist = glob.glob("Tables/*{0}*.fits".format(filt))
		for fitsfile in fitslist:
			table = pyfits.open(fitsfile)[1].data
			for row in table:
				for i in np.arange(3,18):
					row[i] = np.nan_to_num(row[i])
				con.execute("INSERT INTO DataTable{15} Values({0},{1},{2},{3},{4},{5});".format(
				fitsfile[13], row[3],row[4],row[12],row[13],row[18]))
	
	con.commit()

def R1():
	print "Images observed between MJD=56800 and MJD=57300 and the number of stars in them with S/N>5"
	for filt in ["Y","Z","J","H","Ks"]:
		rows = con.execute("""SELECT count(DataTable{0}.StarID),InfoTable.Filename 
		FROM DataTable{0} INNER JOIN InfoTable on InfoTable.FieldID=DataTable{0}.FieldID 
		GROUP BY InfoTable.Filename HAVING 56800<InfoTable.MJD<57300 AND 
		InfoTable.Filter='{0}' AND DataTable{0}.Flux1/DataTable{0}.dFlux1>5""".format(filt))
		for row in rows:
			print row

def R2():
	rows = con.execute("""Select DataTableJ.StarID, DataTableJ.Mag1-DataTableH.Mag1 
	FROM DataTableJ INNER JOIN DataTableH ON DataTableJ.StarID = DataTableH.StarID 
	WHERE DataTableJ.Mag1-DataTableH.Mag1>1.5""")
	print "Objects with J-H>1.5"
	values = np.array([])
	stars = np.array([])
	for row in rows:
		values = np.append(values,row[1])
		stars = np.append(stars,row[0])
		print row
	plt.hist(values,bins=np.arange(1.5,2.5,.05))
	plt.title("J-H magnitudes")
	plt.xlabel("J-H")
	plt.ylabel("Frequency")
	plt.savefig("R2.jpg")
	plt.show()
	np.savetxt("StarIDs_R2.txt",stars)

def R3():
	rows = con.execute("""Select StarID, Mag1, avg(Mag1), dMag1 FROM DataTableKs WHERE abs(Mag1-(Select avg(Mag1)
		FROM DataTableKs))>20*dMag1""")
	print "Objects with mag more than 20 times away from the mean"
	vals = np.empty([0,2])
	for row in rows:
		print row

def R4():
	for field in np.arange(1,4):
		rows = con.execute("""Select Filename FROM InfoTable WHERE FieldID={0}""".format(field))
		print "The files for field {0}".format(field)
		for row in rows:
			print row

def R5():
	rows = con.execute("""Select DataTable{0}.Mag1, DataTable{1}.Mag1, DataTable{2}.Mag1, 
	DataTable{3}.Mag1, DataTable{4}.Mag1 FROM DataTable{0}, DataTable{1}, DataTable{2}, 
	DataTable{3}, DataTable{4} WHERE DataTable{0}.Flux1/DataTable{0}.dFlux1>30 AND
	DataTable{2}.Flux1/DataTable{2}.dFlux1>30 AND DataTable{3}.Flux1/DataTable{3}.dFlux1>30 AND
	DataTable{4}.Flux1/DataTable{4}.dFlux1>30 AND DataTable{0}.StarID = DataTable{1}.StarID 
	AND DataTable{1}.StarID = DataTable{2}.StarID AND DataTable{2}.StarID = DataTable{3}.StarID
	AND DataTable{3}.StarID = DataTable{4}.StarID""".format("Y","Z","J","H","Ks"))
	print "Magnitudes of all stars with S/N>30"
	vals = np.empty([0,5])
	for row in rows:
		vals = np.append(vals,[row],axis=0)
	
	filters = np.array(["Y","Z","J","H","Ks"])
	xs = np.arange(10,18,.001)
	for i in np.arange(0,5):
		density = gaussian_kde(vals[:,i])
		plt.plot(xs,density(xs),label=filters[i])
	plt.title("Densities of magnitudes in each filter")
	plt.xlabel("Magnitude")
	plt.ylabel("Density")
	plt.legend(loc="best")
	plt.savefig("R5.jpg")
	plt.show()    
    
con = lite.connect('Q1_database.db')
"""
#Making the Database
"""
#maketables(con)

"""
First Querry
"""
#R1()

"""
Second Querry
"""
#R2()

"""
Third Query
"""
#R3()

"""
Fourth Querry
"""
#R4()
		
"""
Fifth Querry
"""
#R5()

"""
Euclid
"""
print "Euclid mission"
rows = con.execute("""Select DataTableY.Mag1-DataTableJ.Mag1, DataTableJ.Mag1-
	DataTableH.Mag1 FROM DataTableY, DataTableJ, DataTableH WHERE DataTableY.StarID
	= DataTableJ.StarID AND DataTableJ.StarID = DataTableH.StarID""")
vals = np.empty([0,2])
for row in rows:
	vals = np.append(vals,[[row[0],row[1]]],axis=0)
np.savetxt("Euclid.txt",vals)

bicvals = np.array([]) 
aicvals = np.array([])
for comp in np.arange(2,20):
	gmm = mixture.GaussianMixture(n_components=comp, covariance_type='full').fit(vals)
	aicvals = np.append(aicvals,gmm.aic(vals))
	bicvals = np.append(bicvals,gmm.bic(vals))

plt.plot(np.arange(2,20),bicvals,label="BIC")
plt.plot(np.arange(2,20),aicvals,label="AIC")
plt.title("Bayesiann and Akaike Information Criteria")
plt.xlabel("Components")
plt.ylabel("BIC")
plt.legend(loc="best")
plt.savefig("Euclid0.jpg")
plt.clf()

print np.argmin(bicvals)+2
gmm = mixture.GaussianMixture(n_components=4, covariance_type='full').fit(vals)
newstars, samples = gmm.sample(100000)

plt.scatter(vals[:,0],vals[:,1])
plt.title("Stars from the database")
plt.xlabel("Y-J")
plt.ylabel("J-H")
plt.savefig("Euclid1.jpg")
plt.clf()

x = np.linspace(-3.0, 1.5)
y = np.linspace(-10., 20.)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = 10**gmm.score_samples(XX)
Z = Z.reshape(X.shape)


fig, ax = plt.subplots()
img = ax.imshow(Z, origin='lower', extent=[-3.,1.5,-10.,20.], norm=LogNorm(vmin=Z.min(), vmax=Z.max()))
ax.set_aspect(.15)
fig.colorbar(img, ax=ax)
plt.clf()

Z = gmm.score_samples(XX).reshape(X.shape)
print Z.max(), Z.min()
CS = plt.contour(X, Y, Z, levels=np.linspace(-276,-1,40))
CB = plt.colorbar(CS, shrink=0.8, extend='both')

plt.scatter(newstars[:,0],newstars[:,1])
plt.xlim([-3.,1.5])
plt.ylim([-10.,20.])
plt.title("Negative log probability distribution and simulated stars")
plt.xlabel("Y-J")
plt.ylabel("J-H")
plt.savefig("Euclid2.jpg")
plt.show()
