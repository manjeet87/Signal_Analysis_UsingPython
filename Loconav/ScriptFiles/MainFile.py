import DataRead_and_PreClean as dr
import DataCleaning_and_Prediction as dc
import matplotlib.pyplot as plt
import pandas as pd
import traceback
from scipy import signal
import scipy as sp
import ctypes
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


def plot_Results(df, smooth_df, result_df, theftpts=[], refPts=[], xlim=[], ylim1=[], ylim2=[], savepth=" "):
    plt.rcParams['figure.figsize'] = [16, 16]
    fig, axi = plt.subplots(4, 1)
    axi[0].plot(df.datetime, df.fuelVoltage, 'g.', markersize=1, linewidth=1);
    # plt.plot(df.datetime, df.distance, 'b-', markersize=1, linewidth=1);
    axi[0].set_title('Original FuelData vs Time')
    axi[0].set_xlabel('time')

    axi[1].plot(smooth_df.index, smooth_df.fuelVoltage, 'g.', markersize=2, linewidth=1)
    axi[1].set_xlabel('time index')
    # Make the y-axis label, ticks and tick labels match the line color.
    axi[1].set_ylabel('FuelVoltage', color='b')
    axi[1].tick_params('y', colors='b')
    if len(ylim1) != 0:
        axi[1].set_ylim(ylim1)
    else:
        axi[1].set_ylim(smooth_df.fuelVoltage.min(), 1.05 * smooth_df.fuelVoltage.max())

    ax2 = axi[1].twinx()
    ax2.plot(smooth_df.index, smooth_df.distance, 'b-', markersize=2, linewidth=1)
    ax2.set_ylabel('Distance', color='b')
    if len(ylim2) != 0:
        ax2.set_ylim(ylim2)
    if len(xlim) != 0:
        plt.xlim(xlim)
    axi[1].set_title('Cleaned Data')

    for pt in refPts:
        axi[1].axvline(pt, color='Red')

    axi[2].plot(smooth_df.index, smooth_df.fuelVoltage, 'g.', markersize=2, linewidth=1)
    axi[2].set_xlabel('time index')
    # Make the y-axis label, ticks and tick labels match the line color.
    axi[2].set_ylabel('FuelVoltage', color='b')
    axi[2].tick_params('y', colors='b')
    if len(ylim1) != 0:
        axi[2].set_ylim(ylim1)
    else:
        axi[2].set_ylim(smooth_df.fuelVoltage.min(), 1.05 * smooth_df.fuelVoltage.max())

    ax2 = axi[2].twinx()
    ax2.plot(smooth_df.index, smooth_df.distance, 'b-', markersize=2, linewidth=1)
    ax2.set_ylabel('Distance', color='b')
    if len(ylim2) != 0:
        axi[2].set_ylim(ylim2)

    if len(xlim) != 0:
        plt.xlim(xlim)
    plt.title('Final Predicted Theft Pts - ZOOMED')

    for pt in result_df.theft_index:
        plt.axvline(pt, color='black')


    axi[3].plot(smooth_df.datetime, smooth_df.fuelVoltage, 'g-', markersize=2, linewidth=1);
    axi[3].set_title('Smoothed FuelVoltage Level')

    axi[3].set_xlabel('time index')
    if len(xlim) != 0:
        plt.xlim(xlim)
    if len(ylim1) != 0:
        axi[3].set_ylim(ylim1)
    else:
        axi[3].set_ylim(smooth_df.fuelVoltage.min(), 1.05 * smooth_df.fuelVoltage.max())


    fig.tight_layout()

    #     if len(xlim) !=0:
    #         plt.xlim(xlim)
    #     if len(ylim)!=0:
    #         plt.ylim(ylim)

    # plt.plot(df_clean.index, df_clean.distance, 'b-', markersize=1, linewidth=1);

    plt.savefig(savepth + '_R2.png')
    plt.close()

#####################################################################
### Function to generate FuelMaxVoltage & FuelMinVoltage, to be sent
### to the main devices database for records.
def Gen_FuelMaxMin(df):
    df = dr.perform_PreFormating(df)
    dff, dff2 = dr.perform_postFormating(df)


    y_smooth = sp.signal.medfilt(dff.fuelVoltage, 99)
    fmax = max(y_smooth)
    fmin = min(y_smooth)
    #df_clean = dc.Clean_NoiseData(dff, fmax, fmin, 0)

    return fmax, fmin

def avg_NeigbourDistance(dff):
    dd = dff.fuelVoltage.shift(-1) - dff.fuelVoltage
    dd = dd.dropna()
    dd2 = dd[abs(dd - dd.mean()) < 1.5 * dd.std()]
    meddev = abs(dd2 - dd2.median()).median()
    meddev2 = abs(dd2 - dd2.median()).mean()
    print(dd.median(), 2 * meddev, 2 * dd.std(), 0.01 * (fuelMax - fuelMin), 0.01 * (fuelMax - 0.05 * fuelMax))
    distrow = [dd2.median(), 2 * meddev2, 2 * meddev, 2 * dd2.std(), 0.01 * (fuelMax - fuelMin),
                       0.01 * (fuelMax - 0.05 * fuelMax)]
    #print(dd)
    #plt.hist(dd, bins=100)
   # plt.axvline(dd.median(), color = 'black')
    #plt.axvline(2*dd.std(), color = 'black')
    #plt.semilogy()
    neb_dist = 2 * meddev
    return neb_dist, distrow

######################################################################
#### Main Code Starts

folderpath = r"H:\Analytics\FuelAnalysis\test\nf"
savePath = r"H:\Analytics\FuelAnalysis\results"
filepath = r""

########################################################################
#### MAX MIN to passed on to function to examine fueldata on small dataset.
#### To be read originally from Main Database of devices.
#########################################################################
fuelMax = 100
fuelMin = 0


filesname = dr.read__MultipleCSVs(folder_path= folderpath, nfiles=45)
ctr = 0
error_log = pd.DataFrame()
distDF = pd.DataFrame(columns=['median','2*Medev2', '2*Meddev','2*std','MaxMin','MaxMin0'])
filename=[]
errLog = []
for file in filesname:
    #df_list[0].info()
    df = dr.read_SingleCSV(file)
    build_savePath = savePath + r"\dataset_" + str(ctr) + "_results"

    try:
        fuelMax, fuelMin = Gen_FuelMaxMin(df)
        print("MaxMin : ",fuelMax, fuelMin)
        df = dr.perform_PreFormating(df)
        print("Dataset_" + str(ctr + 1) + " Preformatting Done")

        dff, dff2 = dr.perform_postFormating(df)
        print("Dataset_" + str(ctr + 1) + " Postformatting Done")

        neb_dist, distrow = avg_NeigbourDistance(dff)

        distDF.loc[ctr] = distrow


        ######################################################################
        #### Generate Smooth Equivalent Curve
        smooth_df = dc.generate_SmoothCurve(dff)

        #Dmax = dff.distance.max()
        #df_clean = dc.Clean_NoiseData(dff, fuelMax, fuelMin, neb_dist)
        print("Dataset_" + str(ctr + 1) + " Data Cleaning Done")

        fig, axi = plt.subplots(2,1)
        axi[0].plot(dff.index, dff.fuelVoltage,'g.', markersize =1)
        axi[1].plot(smooth_df.index, smooth_df.fuelVoltage, 'r.', markersize=1)
        #axi[2].plot(df_clean.index, df_clean.fuelVoltage, 'b.', markersize=1)

        plt.savefig(build_savePath + '_R1.png')
        #plt.show()
        plt.close()
        # plt.plot(df_clean.index, df_clean.fuelVoltage, 'g.', markersize=1)
        #
        # plt.savefig(build_savePath + '_R1.png')
        # plt.close()

        theft_pts, refpts = dc.jump_point(smooth_df, 0.01, fuelMax, fuelMin, 0)
        print("Dataset_" + str(ctr + 1) + " Theft points Indentified")
        #
        xlim = []
        #plot_theftpts(smooth_df,theftpts=[], refPts=refpts, xlim = xlim)
        #plot_theftpts(smooth_df,theftpts=theft_pts, refPts=[], xlim = xlim)
        if len(refpts) ==0:
            raise Exception("Zero Refuel Points. This datasets donot supports Refuel Points Anaylsis")

        if len(theft_pts) ==0:
            raise Exception("Zero Theft Points. This datasets donot supports Theft Points Anaylsis")


        ######################################################################
        #### Generate Refuel Table
        refuel_df = dc.generate_ReFuelTable(smooth_df, ref_pts= refpts, fuelMax=fuelMax, fuelMin=fuelMin)
        #
        #################################################################
        #### Find Avg Consumption Rate Range
        max_DecayRate = dc.findMax_decayRate(smooth_df, fuelMax, fuelMin)
        print("Dataset_" + str(ctr + 1) + " Max Decay Rate evaluated")
        print(max_DecayRate)
        #
        #####################################################################
        ### Generating results table for theft points
        theft_df = dc.generate_TheftTable(smooth_df, theft_pts, max_DecayRate, fuelMax, fuelMin)

        # build_savePath = savePath + r"\result_dataset_" + filesname[ctr].replace(folderpath,"").replace('\\', "")
        # #result_df.to_csv(build_savePath)

        ######################################################################
        ##### Finding Refueling distance, or avg distance vehicle can travel from current fuel Level
        curr_fuelVoltage = 0.25*(fuelMax - fuelMin) + fuelMin
        refuel_Distance = dc.findAvg_RefuelDistance(smooth_df, curr_fuelVoltage,fuelMax, fuelMin)
        print ("Avg. Distance that can be travelled Before Refueling = ", round(refuel_Distance,2), " Kms")


        ####Plotting complete results

        plot_Results(dff,smooth_df, theft_df, theftpts= theft_pts, refPts=refuel_df.refuel_index, xlim = xlim, savepth = build_savePath)


        ctr+=1
        #print(result_df)
        #print(refuel_df)
    except Exception as err:
        filename.append(build_savePath)

        ctypes.windll.user32.MessageBoxW(0, str(err), "Error Exception", 1)
        errLog.append(str(traceback.format_exc()))
        print(traceback.format_exc())
        ctr+=1
error_log['Filename'] = pd.Series(filename)
error_log['error_Log'] = pd.Series(errLog)
error_log.to_csv("ErrorLog.csv")
distDF.to_csv("distanceDf.csv")