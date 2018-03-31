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
def plot_theftpts(cleanedDf , theftpts = [], xlim=[], ylim = []):

    plt.rcParams['figure.figsize'] = [20, 4]
    # plt.subplot(6,1,1)
    plt.plot(cleanedDf.index, cleanedDf.fuelVoltage, 'g.', markersize=2, linewidth=1);
    plt.plot(cleanedDf.index, cleanedDf.distance, 'b-', markersize=2, linewidth=1);
    if len(xlim) !=0:
        plt.xlim(xlim)
    if len(ylim)!=0:
        plt.ylim(ylim)
    for pt in theftpts:
        plt.axvline(pt)
    plt.show()

def plot_Results(df, df_clean, result_df, xlim = [], ylim = []):

    plt.rcParams['figure.figsize']=[20,4]
    plt.subplot(3, 1, 1)
    plt.plot(df.datetime, df.fuelVoltage, 'g.', markersize=1, linewidth=1);
    plt.plot(df.datetime, df.distance, 'b-', markersize=1, linewidth=1);

    plt.subplot(3, 1, 2)
    plt.plot(df_clean.index, df_clean.fuelVoltage, 'g.', markersize=1, linewidth=1);
    plt.plot(df_clean.index, df_clean.distance, 'b-', markersize=1, linewidth=1);

    if len(xlim) !=0:
        plt.xlim(xlim)
    if len(ylim)!=0:
        plt.ylim(ylim)

    for pt in result_df.theft_index:
        plt.axvline(pt)

    plt.subplot(3, 1, 3)
    plt.plot(result_df.theft_index, result_df.fuel_jump, 'g.', markersize=3, linewidth=1);

    if len(xlim) !=0:
        plt.xlim(xlim)
    if len(ylim)!=0:
        plt.ylim(ylim)

    #plt.plot(df_clean.index, df_clean.distance, 'b-', markersize=1, linewidth=1);

    plt.show()


######################################################################
#### Main Code Starts

folderpath = r"G:\Analytics\FuelAnalysis\test2"
savePath = r"G:\Analytics\FuelAnalysis\results"
filepath = r""
fuelMax = 100
df_list, filesname = dr.read__MultipleCSVs(folder_path= folderpath, nfiles=1)
ctr = 0
for df in df_list:
    #df_list[0].info()
    Dmax = df.distance.max()
    df = dr.perform_PreFormating(df)
    print ("Dataset_"+str(ctr+1) +" Preformatting Done")

    dff = dr.perform_postFormating(df, fuelMax)
    print("Dataset_" + str(ctr + 1) + " Postformatting Done")

    df_clean = dc.Clean_NoiseData(dff, level= 6)
    print("Dataset_" + str(ctr + 1) + " Data Cleaning Done")

    theft_pts = dc.theft_point(df_clean, level= 0.01)
    print("Dataset_" + str(ctr + 1) + " Theft points Indentified")

    #plotData_profiles(df)
    xlim = [13200,14800]
    plot_theftpts(df_clean,theft_pts, xlim = xlim)
    
    #####################################################################
    ### Generating results table
    result_df = dc.generate_PredictTable(df_clean,theft_pts,Dmax, fuelMax)

    build_savePath = savePath + r"\result_dataset_" + filesname[ctr].replace(folderpath,"").replace('\\', "")
    result_df.to_csv(build_savePath)

    plot_Results(dff,df_clean,result_df, xlim = xlim)
    ctr+=1
    print(result_df)
