import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

desired_width = 360
pd.set_option('display.width', desired_width)


def Clean_NoiseData(dff, level):
    x = np.array(dff.index)
    y = np.array(dff.fuelVoltage)

    from sympy.geometry import Point
    i = 0
    dd00000 = [0, 0, 0, 0, 0, 0]
    dd0000 = [0, 0, 0, 0, 0]
    dd000 = [0, 0, 0, 0]
    dd00 = [0, 0, 0]
    dd0 = [0, 0]
    dd1 = [0]

    dd2 = [y[1] - y[0]]
    dd3 = [y[2] - y[0]]
    dd4 = [y[3] - y[0]]
    dd5 = [y[4] - y[0]]
    dd6 = [y[5] - y[0]]
    dd7 = [y[6] - y[0]]

    for i in range(1, len(x)):
        try:
            d00000 = abs(y[i] - y[i - 6])
            d0000 = abs(y[i] - y[i - 5])
            d000 = abs(y[i] - y[i - 4])
            d00 = abs(y[i] - y[i - 3])
            d0 = abs(y[i] - y[i - 2])
            d1 = abs(y[i] - y[i - 1])
            d2 = abs(y[i + 1] - y[i])
            d3 = abs(y[i + 2] - y[i])
            d4 = abs(y[i + 3] - y[i])
            d5 = abs(y[i + 4] - y[i])
            d6 = abs(y[i + 5] - y[i])
            d7 = abs(y[i + 6] - y[i])
        except:
            continue

        dd00000.append(d00000)
        dd0000.append(d0000)
        dd000.append(d000)
        dd00.append(d00)
        dd0.append(d0)
        dd1.append(d1)
        dd2.append(d2)
        dd3.append(d3)
        dd4.append(d4)
        dd5.append(d5)
        dd6.append(d6)
        dd7.append(d7)

        # print (i)

        # dd1.append(0)
        # dd2.append(0)
    dff['dd00000'] = pd.Series(dd00000)
    dff['dd0000'] = pd.Series(dd0000)
    dff['dd000'] = pd.Series(dd000)
    dff['dd00'] = pd.Series(dd00)
    dff['dd0'] = pd.Series(dd0)
    dff['dd1'] = pd.Series(dd1)
    dff['dd2'] = pd.Series(dd2)
    dff['dd3'] = pd.Series(dd3)
    dff['dd4'] = pd.Series(dd4)
    dff['dd5'] = pd.Series(dd5)
    dff['dd6'] = pd.Series(dd6)
    dff['dd7'] = pd.Series(dd7)

    p = dff['dd1']
    ## Removing Error Data
    if level == 1:
        dff1 = dff[(dff.dd1 <= 0.01) & (dff.dd2 <= 0.01)]
        dff1 = dff1.reset_index(drop=True)  ## Reseting index

    if level == 2:
        dff1 = dff[(dff.dd1 <= 0.005) & (dff.dd2 <= 0.005) & (dff.dd0 <= 0.01) & (dff.dd3 <= 0.01)]
        dff1 = dff1.reset_index(drop=True)  ## Reseting index

    if level == 3:
        dff1 = dff[(dff.dd1 <= 0.005) & (dff.dd2 <= 0.005) & (dff.dd0 <= 0.01) & (dff.dd3 <= 0.01) &
                   (dff.dd00 <= 0.015) & (dff.dd4 <= 0.015)]
        dff1 = dff1.reset_index(drop=True)  ## Reseting index

    if level == 4:
        dff1 = dff[(dff.dd1 <= 0.005) & (dff.dd2 <= 0.005) & (dff.dd0 <= 0.01) & (dff.dd3 <= 0.01) &
                   (dff.dd00 <= 0.015) & (dff.dd4 <= 0.015) & (dff.dd000 <= 0.02) & (dff.dd5 <= 0.02)]
        dff1 = dff1.reset_index(drop=True)  ## Reseting index

    if level == 5:
        dff1 = dff[(dff.dd1 <= 0.02) & (dff.dd2 <= 0.02) & (dff.dd0 <= 0.0364) & (dff.dd3 <= 0.04) &
                   (dff.dd00 <= 0.06) & (dff.dd4 <= 0.06) & (dff.dd000 <= 0.08) & (dff.dd5 <= 0.08) &
                   (dff.dd0000 <= 0.1) & (dff.dd6 <= 0.1)]
        dff1 = dff1.reset_index(drop=True)  ## Reseting index

    if level == 6:
        dff1 = dff[(dff.dd1 <= 0.02) & (dff.dd2 <= 0.02) & (dff.dd0 <= 0.04) & (dff.dd3 <= 0.04) &
                   (dff.dd00 <= 0.06) & (dff.dd4 <= 0.06) & (dff.dd000 <= 0.08) & (dff.dd5 <= 0.08) &
                   (dff.dd0000 <= 0.1) & (dff.dd6 <= 0.1) & (dff.dd00000 <= 0.12) & (dff.dd7 <= 0.12)]
        dff1 = dff1.reset_index(drop=True)  ## Reseting index

    # plt.rcParams['figure.figsize'] = [16, 4]
    # plt.plot(p, 'b.')
    # plt.title('Histogram - Consecutive Fuel Difference ', fontsize=15)
    # plt.ylim(0, 0.05)
    #
    # #     plt.ylim(0.02,1)
    # #     plt.savefig("test.png")
    # plt.rcParams['figure.figsize'] = [16, 4]
    # plt.plot(dff.index[:], dff.fuelVoltage[:], 'g.', markersize=2, linewidth=1);
    # plt.ylim(0, 1.1)

    return dff1


def theft_point(dff, level = 0.05):
    x = np.array(dff.index)
    y = np.array(dff.fuelVoltage)

    theft_pts = []
    ctr = 0

    from sympy.geometry import Point
    i = 0
    dd1 = [0]

    dd2 = [y[1] - y[0]]
    for i in range(1, len(x)):
        try:
            #             d1 =abs(y[i+1] - y[i])
            #             d2 =abs(y[i+2] - y[i])
            d_forward = (y[i + 1:i + 15] - y[i])
            d_backward = (y[i + 1] - y[i - 15:i])
            # print (d_backward)
        except:
            print("**")
        # dd1.append(d1)
        # dd2.append(d2)
        # if (d1 >= 0.05) & (d2 >= 0.05)&(d3 >= 0.05)&(d4 >= 0.05)&(d5 >= 0.05)&(d5 >= 0.05):
        if ((sum(d_forward < -1 * level) in list(range(14, 16)))):  # & (sum(d_forward<0.1) == 19)):
            if (sum(d_backward < -1 * level) in list(range(14, 16))):
                theft_pts.append(dff.index[i])
                ctr += 1
                # print(theft_pts, ctr)

        if (ctr >= 2):
            if ((theft_pts[ctr - 1] - theft_pts[ctr - 2]) in list(range(1, 6))):
                theft_pts.pop(ctr - 2)
                ctr -= 1
                # print (i)

                # dd1.append(0)
                # dd2.append(0)
    # dff['dd1'] = pd.Series(dd1)
    # dff['dd2'] = pd.Series(dd2)
    print(len(theft_pts))
    return theft_pts

    return theft_pts


def predit_MissingData(df_old, df_cleaned):
    ### Combining all data on common axiz
    ### After removing noise and OFF_State data, replacing them with last known predicted value.

    j = 0  ## Counter for cleaned Data
    i = 0  ## Counter for Old_df

    predict_Data = []
    lastdata_value = df_old.loc[0, 'fuelVoltage']

    while i < (len(df_old)):

        if j < len(df_cleaned):

            if (df_cleaned.loc[j, 'datetime'] > df_old.loc[i, 'datetime']):
                predict_Data.append(lastdata_value)
                # print ('i = ',i)

            elif (df_cleaned.loc[j, 'datetime'] == df_old.loc[i, 'datetime']):
                lastdata_value = df_old.loc[i, 'fuelVoltage']
                j += 1
                predict_Data.append(lastdata_value)
                # print ('*j = ',j)

        else:
            predict_Data.append(df_old.loc[i, 'fuelVoltage'])
            # print (i)

        i += 1

    return predict_Data


def generate_PredictTable(df_cleaned, theft_pts, DMax):
    result_df = pd.DataFrame()
    result_df['theft_index'] = [df_cleaned.index[i] for i in theft_pts]
    result_df['lat'] = [df_cleaned.lat[i] for i in theft_pts]
    result_df['long'] = [df_cleaned.long[i] for i in theft_pts]
    result_df['theft_time'] = [df_cleaned.datetime[i] for i in theft_pts]
    result_df['fuel_jump'] = [(df_cleaned.fuelVoltage[i] - df_cleaned.fuelVoltage[i + 1]) * 500 for i in theft_pts]
    result_df['dist_jump(KM)'] = [(df_cleaned.distance[i + 1] - df_cleaned.distance[i]) * (.001) * DMax for i in theft_pts]
    result_df['time_jump'] = [(df_cleaned.datetime[i + 1] - df_cleaned.datetime[i]) for i in theft_pts]

    result_df['Possibility'] =  (result_df['dist_jump(KM)']/result_df['fuel_jump']) < 1
    result_df['FuelPerKM'] =  result_df['fuel_jump'] /result_df['dist_jump(KM)']

    result_df.to_csv(r"G:\Analytics\FuelAnalysis\results\reults.csv")

    # plt.plot(result_df.theft_time, result_df.FuelPerKM)
    # plt.semilogy()
    # plt.show()
    result_df = result_df[result_df['FuelPerKM'] >2]
    print (len(result_df))
    return result_df
