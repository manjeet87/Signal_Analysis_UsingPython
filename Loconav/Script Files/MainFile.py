import DataRead_and_PreClean as dr
import DataCleaning_and_Prediction as dc
import matplotlib.pyplot as plt
import pandas as pd

######################################################
### Function to plot Data profile in terms of fuelvoltage, Distance over time
def plotData_profiles(df):
    plt.rcParams['figure.figsize'] = [16, 12]

    #### 1. Checking order of Data, as per datetime
    plt.subplot(3,1,1)
    plt.plot(df.index, df.datetime, 'g.', markersize=1, linewidth=1);

    plt.subplot(3,1,2)
    plt.plot(df.index, df.fuelVoltage, 'g.', markersize=2, linewidth=1);

    plt.subplot(3,1,3)
    plt.plot(df.index, df.distance, 'g-', markersize=2, linewidth=1);
    plt.title("Cumulative Distance vs Time", fontsize=15)
    plt.show()

#######################################################
### Function to Plot theft Points over Cleaned Data
def plot_theftpts(cleanedDf , theftpts = []):

    plt.rcParams['figure.figsize'] = [16, 4]
    # plt.subplot(6,1,1)
    plt.plot(cleanedDf.index, cleanedDf.fuelVoltage, 'g.', markersize=2, linewidth=1);
    plt.plot(cleanedDf.index, cleanedDf.distance, 'b-', markersize=2, linewidth=1);
    # plt.xlim(8445,8470)
    for pt in theftpts:
        plt.axvline(pt)
    plt.show()

def plot_Results(df, df_clean, result_df):
    plt.subplot(2, 2, 1)
    plt.plot(df.datetime, df.fuelVoltage, 'g.', markersize=1, linewidth=1);
    plt.plot(df.datetime, df.distance, 'b-', markersize=1, linewidth=1);

    plt.subplot(2, 2, 2)
    plt.plot(df_clean.index, df_clean.fuelVoltage, 'g.', markersize=1, linewidth=1);
    plt.plot(df_clean.index, df_clean.distance, 'b-', markersize=1, linewidth=1);

    plt.subplot(2, 2, 3)
    plt.plot(result_df.theft_time, result_df.fuel_jump, 'g-', markersize=2, linewidth=1);
    
    #plt.plot(df_clean.index, df_clean.distance, 'b-', markersize=1, linewidth=1);

    plt.show()


######################################################################
#### Main Code Starts

folderpath = r"G:\Analytics\FuelAnalysis\test2"
filepath = r""
df_list = dr.read__MultipleCSVs(folder_path= folderpath)

df_list[0].info()
Dmax = df_list[0].distance.max()
df = dr.perform_PreFormating(df_list[0])
df = dr.perform_postFormating(df)
dff = dc.Clean_NoiseData(df, level= 6)
theft_pts = dc.theft_point(dff, level= 0.02)

plotData_profiles(df)

#plot_theftpts(dff,theft_pts)
result_df = dc.generate_PredictTable(dff,theft_pts,Dmax)
plot_Results(df,dff,result_df)

print(result_df)
