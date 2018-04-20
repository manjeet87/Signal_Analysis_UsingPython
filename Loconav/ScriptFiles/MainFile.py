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
def plot_theftpts(cleanedDf, theftpts=[], refPts=[], xlim=[], ylim1=[], ylim2=[]):
    plt.rcParams['figure.figsize'] = [20, 4]
    # plt.subplot(6,1,1)
    fig, ax1 = plt.subplots()
    ax1.plot(cleanedDf.index, cleanedDf.fuelVoltage, 'g.', markersize=2, linewidth=1)
    ax1.set_xlabel('time index')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('FuelVoltage', color='b')
    ax1.tick_params('y', colors='b')
    if len(ylim1) != 0:
        ax1.set_ylim(ylim1)

    ax2 = ax1.twinx()
    ax2.plot(cleanedDf.index, cleanedDf.distance, 'b-', markersize=2, linewidth=1)
    ax2.set_ylabel('Distance', color='b')
    if len(ylim2) != 0:
        ax2.set_ylim(ylim2)
    plt.title("Initial Prediction theft Points - ZOOMED")
    if len(xlim) != 0:
        plt.xlim(xlim)

    for pt in theftpts:
        ax1.axvline(pt, color='black')

    for pt in refPts:
        ax1.axvline(pt, color='Red')
    plt.show()


def plot_Results(df, df_clean, result_df, smooth_df, theftpts=[], refPts=[], xlim=[], ylim1=[], ylim2=[]):
    plt.rcParams['figure.figsize'] = [16, 12]
    fig, axi = plt.subplots(4, 1)
    axi[0].plot(df.datetime, df.fuelVoltage, 'g.', markersize=1, linewidth=1);
    # plt.plot(df.datetime, df.distance, 'b-', markersize=1, linewidth=1);
    axi[0].set_title('Original FuelData vs Time')
    axi[0].set_xlabel('time')

    axi[1].plot(df_clean.index, df_clean.fuelVoltage, 'g.', markersize=2, linewidth=1)
    axi[1].set_xlabel('time index')
    # Make the y-axis label, ticks and tick labels match the line color.
    axi[1].set_ylabel('FuelVoltage', color='b')
    axi[1].tick_params('y', colors='b')
    if len(ylim1) != 0:
        axi[1].set_ylim(ylim1)
    else:
        axi[1].set_ylim(df_clean.fuelVoltage.min(), 1.05*df_clean.fuelVoltage.max())

    ax2 = axi[1].twinx()
    ax2.plot(df_clean.index, df_clean.distance, 'b-', markersize=2, linewidth=1)
    ax2.set_ylabel('Distance', color='b')
    if len(ylim2) != 0:
        ax2.set_ylim(ylim2)
    if len(xlim) != 0:
        plt.xlim(xlim)
    axi[1].set_title('Cleaned Data')

    for pt in refPts:
        axi[1].axvline(pt, color='Red')

    axi[2].plot(df_clean.index, df_clean.fuelVoltage, 'g.', markersize=2, linewidth=1)
    axi[2].set_xlabel('time index')
    # Make the y-axis label, ticks and tick labels match the line color.
    axi[2].set_ylabel('FuelVoltage', color='b')
    axi[2].tick_params('y', colors='b')
    if len(ylim1) != 0:
        axi[2].set_ylim(ylim1)
    else:
        axi[2].set_ylim(df_clean.fuelVoltage.min(), 1.05 * df_clean.fuelVoltage.max())

    ax2 = axi[2].twinx()
    ax2.plot(df_clean.index, df_clean.distance, 'b-', markersize=2, linewidth=1)
    ax2.set_ylabel('Distance', color='b')
    if len(ylim2) != 0:
        axi[2].set_ylim(ylim2)

    if len(xlim) != 0:
        plt.xlim(xlim)
    plt.title('Final Predicted Theft Pts - ZOOMED')

    for pt in result_df.theft_index:
        plt.axvline(pt, color='black')


    axi[3].plot(smooth_df.datetime, smooth_df.SmoothVoltage, 'g-', markersize=2, linewidth=1);
    axi[3].set_title('Smoothed FuelVoltage Level')

    axi[3].set_xlabel('time index')
    if len(xlim) != 0:
        plt.xlim(xlim)
    if len(ylim1) != 0:
        axi[3].set_ylim(ylim1)
    else:
        axi[3].set_ylim(df_clean.fuelVoltage.min(), 1.05*df_clean.fuelVoltage.max())


    fig.tight_layout()

    #     if len(xlim) !=0:
    #         plt.xlim(xlim)
    #     if len(ylim)!=0:
    #         plt.ylim(ylim)

    # plt.plot(df_clean.index, df_clean.distance, 'b-', markersize=1, linewidth=1);

    plt.show()

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

######################################################################
#### Main Code Starts

folderpath = r"G:\Analytics\FuelAnalysis\test3"
savePath = r"G:\Analytics\FuelAnalysis\results"
filepath = r""

########################################################################
#### MAX MIN to passed on to function to examine fueldata on small dataset.
#### To be read originally from Main Database of devices.
#########################################################################
fuelMax = 100
fuelMin = 0

df_list, filesname = dr.read__MultipleCSVs(folder_path= folderpath, nfiles=5)
ctr = 0
for df in df_list:
    #df_list[0].info()
    fuelMax, fuelMin = Gen_FuelMaxMin(df)
    print (fuelMax, fuelMin)
    df = dr.perform_PreFormating(df)
    print ("Dataset_"+str(ctr+1) +" Preformatting Done")

    dff, dff2 = dr.perform_postFormating(df)
    print("Dataset_" + str(ctr + 1) + " Postformatting Done")

    Dmax = dff.distance.max()
    df_clean = dc.Clean_NoiseData(dff, 6, fuelMax, fuelMin)
    print("Dataset_" + str(ctr + 1) + " Data Cleaning Done")

    theft_pts, refpts = dc.jump_point(df_clean, 0.01, fuelMax, 0)
    print("Dataset_" + str(ctr + 1) + " Theft points Indentified")

    #plotData_profiles(df)
    xlim = []
    plot_theftpts(df_clean,theftpts=[], refPts=refpts, xlim = xlim)

    #################################################################
    #### Find Avg Consumption Rate Range
    max_DecayRate = dc.findMax_decayRate(df_clean, fuelMax, fuelMin)
    print("Dataset_" + str(ctr + 1) + " Max Decay Rate evaluated")
    print(max_DecayRate)
    
    #####################################################################
    ### Generating results table for theft points
    result_df = dc.generate_PredictTable(df_clean, theft_pts, max_DecayRate, fuelMax, fuelMin)

    build_savePath = savePath + r"\result_dataset_" + filesname[ctr].replace(folderpath,"").replace('\\', "")
    result_df.to_csv(build_savePath)

    ######################################################################
    #### Generate Refuel Table
    refuel_df = dc.generate_ReFuelTable(df_clean, ref_pts= refpts, fuelMax=fuelMax, fuelMin=fuelMin)

    ######################################################################
    #### Generate Smooth Equivalent Curve
    smooth_df = dc.generate_SmoothCurve(df_clean)

    ######################################################################
    ##### Finding Refueling distance, or avg distance vehicle can travel from current fuel Level
    curr_fuelVoltage = 0.25*(fuelMax - fuelMin) + fuelMin
    refuel_Distance = dc.findAvg_RefuelDistance(df_clean, curr_fuelVoltage,fuelMax, fuelMin)
    print ("Avg. Distance that can be travelled Before Refueling = ", round(refuel_Distance,2), " Kms")

    predDF = dc.predit_MissingData(dff2, df_clean)
    ####Plotting complete results
    plt.plot(predDF.index, predDF.predictFuelVolt,'g.', markersize =2)
    plot_Results(dff,df_clean,result_df, smooth_df, theftpts= theft_pts, refPts=refpts, xlim = xlim)
    ctr+=1
    print(result_df)
    print(refuel_df)