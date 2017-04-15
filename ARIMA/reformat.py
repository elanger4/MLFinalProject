import re
import csv

#This file will prepare the data so that it can be run by the panda ARIMA function

myDays = []

yearNum = 0
dayNum = 1
for j in range (0,117):
	myDays.append(str(yearNum)+"-" + str(dayNum))
	if dayNum == 12:
		yearNum = yearNum + 1
		dayNum = 0
	dayNum = dayNum + 1

print myDays
	

openFile = open("product_distribution_training_set.txt")
for i in range(0, 100):
	content = openFile.readline()
	splitList = re.split(r'\t+', content.rstrip('\t'))
	fileName = splitList[0] + ".csv"	
	del splitList[0]
	outfile = open("data/" + fileName, "w+")
	outfile.write("Day,Amount Sold \n")
	for x in range(1,len(splitList)-1):
		data = myDays[x-1] + "," + str(splitList[x] + ".0")
		print data
		outfile.write(data + "\n")
