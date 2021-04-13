import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt

def createDataFrame(file_name):
    data = []
    headers = None
    with open(file_name) as f:
        line = f.readline()
        headers = line.split(",")

        line = f.readline()
        while line:
            data.append(line.split(","))
            line = f.readline()

    df = pd.DataFrame(data=[sub for sub in data], columns=headers)
    return df

def splitBasedOnTime(df, time_step, total):
    time = [0 for i in range(total//time_step)]
    num_vehicles  = [0 for i in range(total//time_step)]
    for i, j, z in zip(df['ActualTravelTime'], df['BestTravelTime'], df['timeStamp']):
        index = math.floor(float(z)/time_step)
        time[index] += float(i)-float(j)
        num_vehicles[index] += 1

    average = [ x/max(1,y) for x,y in zip(time, num_vehicles)]
    average = [[index, val] for index, val in enumerate(average)]
    print(average[1][:])
    print(time)
    print(num_vehicles)

    d = pd.DataFrame({
        "EpisodeNumber": [row[0] for row in average],
        "Average travel time": [row[1] for row in average]
        },
        index = [row[0] for row in average]
    )
    print(d)
    ax2 = plt.plot([0, 250], [21.088, 21.088], linewidth=2)
    ax1 = sns.lineplot(x="EpisodeNumber", y="Average travel time", data=d)

    #ax1.legend(["VSL"])
    ax1.legend(["No Speed limit Control", "VSL"])
    #plt.xlim(0, 199)
    #plt.ylim(0, 25)
    plt.xlabel("Episode Number", fontsize='x-large')
    plt.ylabel("Average travel time(s)", fontsize='x-large')

    plt.show()
    print(d)

def getTotalTravelTime(df):
    time = 0
    for i,j in zip(df['ActualTravelTime'], df['BestTravelTime']):
       time += float(i) - float(j)
    return time/max(1,df.shape[0])

def getTravelTimeShift(df, per):
    totalCount = 0
    outCount = 0
    for i,j in zip(df['ActualTravelTime'], df['BestTravelTime']):
       if (float(i)/float(j)) > per:
           outCount += 1
       totalCount += 1
    return outCount/max(1, totalCount)

def sort(df, column_name):
    for index, item in enumerate(df[column_name]):
        df[column_name][index] = float(index)

    df.sort_values([column_name])

#dir= "7x7"+"/Conflict/"
dir= "7x7/Demand_amount/unidirectional/"
#file_names = ["VD_CLLA_28.txt", "VD_noLA_28.txt"]

import os
path = 'C:/Users/pgunarathna/PycharmProjects/SMARTS_interface/Test/7x7/Demand_amount/'
path = "/home/student.unimelb.edu.au/pgunarathna/IdeaProjects/Temporary_update_smarts/download/RQ2/Vehicles/"
#path = "/home/student.unimelb.edu.au/pgunarathna/IdeaProjects/Data/ECML/I-AIM/"

for dirname, _, file_names in os.walk(path):
    file_names.sort(key=lambda  x : os.path.getmtime(dirname+x), reverse=True)
    for index, file_name in enumerate(file_names):
        if (file_name.find('VD') == 0):
            name = file_name.split('_2020')
            print(name[0])
            df = createDataFrame(dirname+file_name)
            print(df.shape, file_name)
            #splitBasedOnTime(df, 3600, 3600*250)
            print(getTotalTravelTime(df))
            print(getTravelTimeShift(df, 6))


            #if name[0].find('txt') == -1:
            #    os.rename(dirname+file_name, dirname+name[0]+".txt")

        if index > 54:
            break

    break


#for dirname, _, file_names in os.walk(path):
#    for file_name in file_names:
#        if (file_name.find('VD') == 0):
#            df = createDataFrame(dirname+file_name)
#            print(df.shape, file_name)
#            #splitBasedOnTime(df, 3600, 3600*250)
#            print(getTotalTravelTime(df))
#            print(getTravelTimeShift(df, 6))

#path = 'C:/Users/pgunarathna/IdeaProjects/Temporary_update_smarts/download/'
#file_name = 'VD_noLA_20_3000_2020-05-29_08-53-24.txt'
#
#df = createDataFrame(path+file_name)
#print(df.shape, file_name)
#splitBasedOnTime(df, 3600, 3600*2)
#print(getTotalTravelTime(df))
#print(getTravelTimeShift(df, 6))