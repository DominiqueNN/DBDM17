import sqlite3 as lite

con = lite.connect("DBDM17.db")
cur = con.cursor()

command="""CREATE TABLE IF NOT EXISTS MagTable (Name varchar(10), Ra NOT NULL, Dec NOT NULL, B DOUBLE, R DOUBLE);"""
cur.execute(command)

command="INSERT INTO MagTable('VO-001','12:34:04.2','-00:00:23.4',15.4,13.5);"
cur.execute(command)
command="INSERT INTO MagTable Values('VO-002','12:15:00.0','-14:23:15', 15.9,13.6);"
cur.execute(command)
command="INSERT INTO MagTable Values('VO-003','11:55:43.1','-02:34:17.2',17.2,16.8),('VO-004','11:32:42.1','-00:01:17.3',16.5,14.3);"
cur.execute(command)

command = """CREATE TABLE IF NOT EXISTS PhysTable (Name varchar(10), T INT, FeH FLOAT);"""
cur.execute(command)
command = "INSERT INTO PhysTable Values('VO-001', 4501, 0.13),('VO-002',5321,-0.53),('VO-003',6600,-0.32);"
cur.execute(command)

#RA and Dec of all objects B>16
rows = cur.execute('Select Ra, Dec FROM MagTable WHERE B>16')
for row in rows:
	print "RA:{0} Dec:{1}".format(row[0],row[1])

#B, R, Teff and FeH for all stars in both tables, as the question is not well defined for stars that have no B, R information
rows = cur.execute('Select MagTable.B, MagTable.R, PhysTable.T, PhysTable.FeH FROM MagTable, PhysTable WHERE MagTable.Name=PhysTable.Name')
for row in rows:
	print "Data:  {0},{1},{2},{3}".format(row[0],row[1],row[2],row[3])

#The same for objects with FeH>0
rows = cur.execute('Select MagTable.B, MagTable.R, PhysTable.T, PhysTable.FeH FROM MagTable, PhysTable WHERE MagTable.Name=PhysTable.Name AND PhysTable.FeH>0')
for row in rows:
	print "Data:  {0},{1},{2},{3}".format(row[0],row[1],row[2],row[3])
