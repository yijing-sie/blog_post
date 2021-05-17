import os
import pandas as pd
import glob
import numpy as np
import math
from pyspark.sql import SparkSession
from pyspark.sql.functions import when
from statistics import mean


x_T,y_T = np.sqrt(72), np.sqrt(72) #one of the critical points (the purple one)
x_F,y_F = -x_T, -y_T #the other critical point (the blue one)

path = r'/jet/home/ysie/csv' #path to all csv files
#path = r'.spyder-py3/csv' 
all_files = glob.glob(path + "/*.csv")



spark = SparkSession \
    .builder \
    .appName("Lorenz Attractor") \
    .getOrCreate()

len_1 = [] #the total number of time steps that these 1000 lorenz spend while looping around one of the critical point 
for filename in all_files:
    df = spark.read.csv(filename)
    #df = df.select(df[0].alias("x"), df[1].alias("y"),df[2].alias("z"))
    df_cp = df.withColumn("critical points", when((df[0] - x_T)**2 + (df[1] - y_T)**2 < (df[0] - x_F)**2 + (df[1] - y_F)**2, 1) \
          .otherwise(0))
    df_count = df_cp.groupBy('critical points').count()
    len_1 += [df_count.filter(df_count[0] == 1).collect()[0][1]] 
    df.unpersist()
    df_cp.unpersist()
    df_count.unpersist()
len_0 = list(map(lambda x: 10000 - x, len_1)) #the total time steps that these 1000 lorenz spend while looping around the other critical point

np.savetxt("1.csv", len_1, delimiter=',') #save to a csv file
np.savetxt("0.csv", len_0, delimiter=',') #save to a csv file
print("on average, the 1000 lorenz attractors spending on the critical point (8.49, 8.49, 27) = ", mean(len_1))
print("on average, the 1000 lorenz attractors spending on the critical point (-8.49, -8.49, 27) = ", mean(len_0))