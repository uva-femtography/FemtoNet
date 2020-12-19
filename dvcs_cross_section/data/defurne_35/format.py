import csv

xbj = 0.36
t = -0.364
Q2 = 2.00
Eb = 4.455
L = 1
file_num = 40


file_name = "data_" + str(file_num) + ".csv"
f = open(file_name, "w")

to_open = "Table" + str(file_num) + ".csv"
index = 0
print(to_open)
with open(to_open) as csvfile:
    index = 0
    readCSV = csv.reader(csvfile, delimiter=",")
    for x in readCSV:
        if index > 13:
            phi = x[0]
            xsx = float(x[3]) / 1000
            er = float(x[4]) / 1000
            to_write = (
                str(xbj)
                + ","
                + str(t)
                + ","
                + str(Q2)
                + ","
                + str(Eb)
                + ","
                + phi
                + ","
                + str(L)
                + ","
                + str(xsx)
                + ","
                + str(er)
                + "\n"
            )
            f.write(to_write)
            index += 1
        else:
            index += 1
