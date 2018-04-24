import DataRead_and_PreClean as dr
import DataCleaning_and_Prediction as dc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt

#####################################################################
### Function to generate FuelMaxVoltage & FuelMinVoltage, to be sent
### to the main devices database for records.
def Gen_FuelMaxMin(df):
    df = dr.perform_PreFormating(df)
    dff, dff2 = dr.perform_postFormating(df)

    fmax = dff.fuelVoltage.max()
    fmin = dff.fuelVoltage.min()

    df_clean = dc.Clean_NoiseData(dff, 6, fmax, fmin)

    return df_clean.fuelVoltage.max(), df_clean.fuelVoltage.min()

def avg_NeigbourDistance(dff):
    dd = dff.fuelVoltage.shift(-1) - dff.fuelVoltage
    dd = dd.dropna()
    #print(dd)
    # plt.hist(dd, bins=100)
    # plt.axvline(dd.median(), color = 'black')
    # plt.axvline(2*dd.std(), color = 'black')
    # plt.semilogy()
    return dd


folderpath = r"G:\Analytics\FuelAnalysis\test3"
df_list, filesname = dr.read__MultipleCSVs(folder_path= folderpath, nfiles=9)
ctr = 0
distDF = pd.DataFrame(columns=['median','2*Meddev','2*std','MaxMin','MaxMin0'])
for df in df_list:
    #df_list[0].info()
    fuelMax, fuelMin = Gen_FuelMaxMin(df)
    #print (fuelMax, fuelMin)
    df = dr.perform_PreFormating(df)
   # print ("Dataset_"+str(ctr+1) +" Preformatting Done")

    dff, dff2 = dr.perform_postFormating(df)
   # print("Dataset_" + str(ctr + 1) + " Postformatting Done")

    Dmax = dff.distance.max()
    dd = avg_NeigbourDistance(dff)
    meddev = abs(dd - dd.median()).median()
    print (dd.median(),2*meddev, 2*dd.std(), 0.01*(fuelMax - fuelMin), 0.01*(fuelMax - 0.05*fuelMax))
    distDF.loc[ctr] = [dd.median(),2*meddev, 2*dd.std(), 0.01*(fuelMax - fuelMin), 0.01*(fuelMax - 0.05*fuelMax)]
    ctr+=1

    #df_clean = dc.Clean_NoiseData(dff, 6, fuelMax, fuelMin)
    #print("Dataset_" + str(ctr + 1) + " Data Cleaning Done")