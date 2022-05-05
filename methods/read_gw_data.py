import pandas as pd

from libs.modules.utils import movingaverage, find_nearest_index, find_nearest_value

def rain_power_spectra(name_df):
    """
    reads pickled dataframe from lucas_script() input made previously and creates power spectra
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.mlab as mlab
    import matplotlib.gridspec as gridspec

    work_dir = os.getcwd()
    dataframe = os.path.join(work_dir, r'files\dataframe')
    df_path = os.path.join(dataframe, name_df)
    # df_path = r'E:\PythonProjects\pythonProject_Tauranga_Simon\files\dataframe\data_df'
    plot_size = (10, 8)
    plot_file_type = '.png'
    plots_dir = r'files\plots_rain_psd'

    # set and make folder for outputs (if it doesn't exist already)
    output_path = os.path.join(work_dir, plots_dir)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    print('Reading pickled dataframe object')
    data_df = pd.read_pickle(df_path)
    plot_list = [col for col in data_df]

    print('calculating and plotting power spectral densities')

    # https://www.geeksforgeeks.org/plot-the-power-spectral-density-using-matplotlib-python/

    for col in plot_list:
        col_df = data_df[col]

        # col_df=col_df.resample('60T').mean() # resample to hourly (T = minute)

        [Pxx, freqs] = mlab.psd(col_df.dropna(axis='index'),
                                NFFT=16384,
                                noverlap=2048,
                                # Fs = 1/(3600), # if hourly data
                                Fs=1 / (15 * 60),  # 15-min data
                                scale_by_freq=False,
                                detrend='linear'
                                )

        plt.subplot(211)
        col_plot = col_df.plot(title=col, figsize=plot_size)
        col_plot.set_ylabel("rain (mm per time unit)")
        col_plot.set_xlabel("")
        plt.xticks(rotation=0, ha='center')
        plt.grid('on')

        plt.subplot(212)
        plt.semilogx(1 / freqs[1:], 10 * np.log10(Pxx[1:]))
        plt.grid('on')
        plt.ylabel('PSD (dB)')
        plt.xlabel('Period')  # unit is s
        plt.xticks([3600, 6 * 3600, 12 * 3600, 86400, 7 * 86400, 30.5 * 86400, 92 * 86400],
                   ['hourly', '6-hourly', '12-hourly', 'daily', 'weekly', 'monthly', '3-monthly'],
                   rotation=45)
        # plt.show()

        plot_name = col + '_ts_psd' + plot_file_type
        plot_name = os.path.join(output_path, plot_name)
        print(plot_name)
        plt.savefig(plot_name)
        plt.clf()
        plt.close('all')
    print("Finished!")


def gw_power_spectra():
    """
    reads pickled dataframe from lucas_script() input made previously and creates power spectra
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.mlab as mlab
    import matplotlib.gridspec as gridspec

    work_dir = r'E:\PythonProjects\pythonProject_Tauranga_Simon\files'
    dataframe = os.path.join(work_dir, 'dataframe')
    df_path = os.path.join(dataframe, 'data_df')
    # df_path = r'E:\PythonProjects\pythonProject_Tauranga_Simon\files\dataframe\data_df'
    plot_size = (10, 8)
    plot_file_type = '.png'
    plots_dir = r'plots_psd'

    # set and make folder for outputs (if it doesn't exist already)
    output_path = os.path.join(work_dir, plots_dir)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    print('Reading pickled dataframe object')
    data_df = pd.read_pickle(df_path)
    plot_list = [col for col in data_df]

    print('calculating and plotting power spectral densities')

    # https://www.geeksforgeeks.org/plot-the-power-spectral-density-using-matplotlib-python/

    for col in plot_list:
        # col = plot_list[0] # for col in plot_list:
        # col_df = data_df.loc[date_start:date_end][col]
        col_df = data_df[col]

        # col_df=col_df.resample('60T').mean() # resample to hourly (T = minute)

        [Pxx, freqs] = mlab.psd(col_df.dropna(axis='index'),
                                NFFT=16384,
                                noverlap=2048,
                                # Fs = 1/(3600), # if hourly data
                                Fs=1 / (15 * 60),  # 15-min data
                                scale_by_freq=False,
                                detrend='linear'
                                )

        plt.subplot(211)
        col_plot = col_df.plot(title=col, figsize=plot_size)
        col_plot.set_ylabel("gw elevation (to NZVD2016 datum)")
        col_plot.set_xlabel("")
        plt.xticks(rotation=0, ha='center')
        plt.grid('on')

        # plt.subplot(212)
        # [Pxx,freqs] = plt.psd(col_df.dropna(axis = 'index'),
        #         NFFT = 16384,
        #         noverlap = 2048,
        #         # Fs = 1/(3600), # if hourly data
        #         Fs = 1/(15*60),  # 15-min data
        #         scale_by_freq= False,
        #         detrend = 'linear'
        #         )
        # plt.xticks(rotation=45)
        # options for psd (https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.psd.html):

        # plt.subplot(223)
        #
        # plt.semilogx(freqs[1:],10*np.log10(Pxx[1:]))
        # plt.grid('on')
        # plt.ylabel('PSD')
        # plt.xlabel('Frequency') # unit is Hz
        # plt.xticks([1/(92*86400),1/(30.5*86400),1/(7*86400),1/86400,1/3600],
        #            ['3-monthly', 'monthly', 'weekly', 'daily', 'hourly'],
        #            rotation = 45)

        plt.subplot(212)
        plt.semilogx(1 / freqs[1:], 10 * np.log10(Pxx[1:]))
        plt.grid('on')
        plt.ylabel('PSD (dB)')
        plt.xlabel('Period')  # unit is s
        plt.xticks([3600, 6 * 3600, 12 * 3600, 86400, 7 * 86400, 30.5 * 86400, 92 * 86400],
                   ['hourly', '6-hourly', '12-hourly', 'daily', 'weekly', 'monthly', '3-monthly'],
                   rotation=45)
        # plt.show()

        plot_name = col + '_ts_psd' + plot_file_type
        plot_name = os.path.join(output_path, plot_name)
        plt.savefig(plot_name)
        plt.clf()
        plt.close('all')
    print("Finished!")


def read_gw_data(read_df):
    '''
    Reads gw level data from a csv or a pickled dataframe.
   :param read_df: False: csv will be read and pickled to dataframe, True: pickle will be read
   :return: dataframe
   '''

    import os
    import pandas as pd
    import datetime as dt
    import numpy as np

    #print(os.getcwd())
    work_dir = os.getcwd()
    # input_csv and plots_dir should be in work dir. Plots dir will be made if not already there.
    # work_dir = 'E:/PythonProjects/pythonProject_Tauranga_Simon/files'
    input_csv = r'files/20210916 GWData_partiallyNZSTcorrected_NaNremoved_cleaned_STATS.csv'
    time_col = 'Time'

    dataframe = os.path.join(work_dir, r'files/dataframe')
    if not os.path.exists(dataframe):
        os.mkdir(dataframe)

    df_path = os.path.join(dataframe, 'gw_data_df')

    # Either read dataframe or read csv and write dataframe.
    if read_df:
        print('Reading dataframe object from ' + df_path)
        data_df = pd.read_pickle(df_path)
    else:
        print('Reading csv file')
        data_df = pd.read_csv(os.path.join(work_dir, input_csv), encoding='utf-8-sig')
        # convert time to timestamp
        data_df['Time'] = pd.to_datetime(data_df['Time'], dayfirst=True)
        # set the index to timestamp
        data_df.set_index('Time', inplace=True)
        print('Saving dataframe object')
        data_df.to_pickle(df_path)

    return data_df


def read_rain_data(read_df):
    '''
    Reads rain data from a csv or a pickled dataframe.
    :param read_df: False: csv will be read and pickled to dataframe, True: pickle will be read
    :return: dataframe
    '''

    import os
    import pandas as pd
    import datetime as dt
    import numpy as np

    # input_csv and plots_dir should be in work dir. Plots dir will be made if not already there.
    work_dir = os.getcwd()
    input_csv = r'files/rain_tauranga.csv'
    time_col = 'Time'

    dataframe = os.path.join(work_dir, r'files/dataframe')
    if not os.path.exists(dataframe):
        os.mkdir(dataframe)

    df_path = os.path.join(dataframe, 'rain_data_df')

    if read_df:
        print('Reading dataframe object from ' + df_path)
        data_df = pd.read_pickle(df_path)
    else:
        print('Reading csv file')
        data_df = pd.read_csv(os.path.join(work_dir, input_csv), encoding='utf-8-sig')
        # convert time to timestamp
        data_df[time_col] = pd.to_datetime(data_df[time_col], dayfirst=True)
        # set the index to timestamp
        data_df.set_index(time_col, inplace=True)
        print('Saving dataframe object')
        data_df.to_pickle(df_path)

    return data_df


def read_rain_locations(read_df):
    '''
    Reads rain location data from a csv or a pickled dataframe.
    :param read_df: False: csv will be read and pickled to dataframe, True: pickle will be read
    :return: dataframe
    '''

    import os
    import pandas as pd
    import datetime as dt
    import numpy as np

    # input_csv and plots_dir should be in work dir. Plots dir will be made if not already there.
    work_dir = os.getcwd()
    input_csv = r'files/TCC_logger_info.csv'
    time_col = 'Time'

    dataframe = os.path.join(work_dir, r'files/dataframe')
    if not os.path.exists(dataframe):
        os.mkdir(dataframe)

    df_path = os.path.join(dataframe, 'rain_location_data_df')

    if read_df:
        print('Reading dataframe object from ' + df_path)
        data_df = pd.read_pickle(df_path)
    else:
        print('Reading csv file')
        data_df = pd.read_csv(os.path.join(work_dir, input_csv), encoding='utf-8-sig')
        print('Saving dataframe object')
        data_df.to_pickle(df_path)
    return data_df


def merge_dataframes(df1, df2, save_to_pickle):
    import pandas as pd
    import os

    work_dir = os.getcwd()

    dataframe = os.path.join(work_dir, r'files/dataframe')
    if not os.path.exists(dataframe):
        os.mkdir(dataframe)

    print('Merging dataframe objects')
    merged_df = pd.merge_asof(df2, df1, on='Time')
    merged_df['Time'] = pd.to_datetime(merged_df['Time'], dayfirst=True)
    # set the index to timestamp
    merged_df.set_index('Time', inplace=True)

    if save_to_pickle:
        df_path = os.path.join(dataframe, 'merged_data_df')
        merged_df.to_pickle(df_path)

    return merged_df


def make_coherence_spectra(df_rain,df_gw):
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.mlab as mlab

    nfft_power = 14
    # print('nfft_value =' + str(2**nfft_power))
    work_dir = os.getcwd()
    plot_size = (10, 8)
    plot_file_type = '.png'
    plots_dir = r'files/plots_coherence'
    time_col = 'Time'

    # set and make folder for outputs (if it doesn't exist already)
    output_path = os.path.join(work_dir, plots_dir)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    merged_df = merge_dataframes(df_gw,df_rain,False)

    print('Calculating coherence...')
    # df_rain_mean = df_rain.mean(axis = 1)
    # for each gwcol for col in col list blabla
    # print(df_rain_mean)
    gw_list = [col for col in df_gw]
    rain_list = [col for col in df_rain]

    df_coherence = pd.DataFrame()
    df_coherence['Name'] = ['Cxy h-6h', 'Cxy 6h-daily', 'Cxy daily-weekly']
    for col in gw_list:
    # col = gw_list[0]
        plot_list = rain_list + [col]

        df_tmp = merged_df[plot_list].dropna()
            # [Pxx, freqs] = mlab.psd(df_tmp[rain_list].mean(axis=1), # power spectrum of a timeseries
            #                     NFFT=2**nfft_power,
            #                     noverlap=2048,
            #                     Fs=1 / (15 * 60),  # 15-min data
            #                     scale_by_freq=False,
            #                     detrend='linear'
            #                     )
            #
            # [Pyy, freqs] = mlab.psd(df_tmp[col], # power spectrum of a timeseries
            #                     NFFT=2**nfft_power,
            #                     noverlap=2048,
            #                     Fs=1 / (15 * 60),  # 15-min data
            #                     scale_by_freq=False,
            #                     detrend='linear'
            #                     )
            #
            # [Pxy, freqs] = mlab.csd(df_tmp[rain_list].mean(axis=1), df_tmp[col], # cross-spectral density power spectrum of two timeseries
            #                         NFFT=2**nfft_power,
            #                         noverlap=2048,
            #                         Fs=1 / (15 * 60),  # 15-min data
            #                         scale_by_freq=False,
            #                         detrend='linear'
            #                         )

        # coherence (normalised Pxy). Basically, does all of the above (Pxx, Pyy and Pxy) in one and then normalises.
        [Cxy, freqs] = mlab.cohere(df_tmp[rain_list].mean(axis=1), df_tmp[col],
                            NFFT=2**nfft_power,
                            noverlap=2048,
                            Fs=1 / (15 * 60),  # 15-min data
                            scale_by_freq=True,
                            detrend='linear'
                            )
        Cxy = abs(Cxy) # because Cxy is complex

        plt.subplot(311)
        col_plot = df_tmp[col].plot(title=col, figsize=plot_size)
        col_plot.set_ylabel("gw elevation (to NZVD2016 datum)")
        col_plot.set_xlabel("")
        plt.xticks(rotation=0, ha='center')
        plt.grid('on')

        plt.subplot(312)
        rain_plot = df_tmp[rain_list].mean(axis=1).plot(title='', figsize=plot_size)
        rain_plot.set_ylabel("rain (mm per time unit)")
        rain_plot.set_xlabel("")
        plt.xticks(rotation=0, ha='center')
        plt.grid('on')

        # plt.subplot(313) # I left this in for checking if needed
        # plt.semilogx(1 / freqs[1:], 10 * np.log10(abs(Pxy[1:])))
        # plt.grid('on')
        # plt.ylabel('CSD (dB)')
        # plt.xlabel('')  # Period, unit is s
        # plt.xticks([3600, 6 * 3600, 12 * 3600, 86400, 7 * 86400, 30.5 * 86400, 92 * 86400],
        #            ['hourly', '6-hourly', '12-hourly', 'daily', 'weekly', 'monthly', '3-monthly'],
        #            rotation=45)

        plt.subplot(313)
        plt.semilogx(1 / freqs[1:], Cxy[1:])
        Cxy_av = movingaverage(Cxy, 50) # a moving average looks a bit more comprehensible.
        plt.semilogx(1 / freqs[1:], Cxy_av[1:],"r")
        plt.grid('on')
        plt.ylabel('Coherence')
        plt.xlabel('')  # Period, unit is s
        plt.xticks([3600, 6 * 3600, 12 * 3600, 86400, 7 * 86400, 30.5 * 86400, 92 * 86400],
                    ['hourly', '6-hourly', '12-hourly', 'daily', 'weekly', 'monthly', '3-monthly'],
                    rotation=45)
        plt.tight_layout()
        # plt.show()

        # find interval ranges
        idx1 = find_nearest_index(1 / freqs[1:], 3600) # value1 = find_nearest_value(1/freqs, 3600) # check for debug
        idx2 = find_nearest_index(1 / freqs[1:], 6*3600) # value2 = find_nearest_value(1 / freqs, 6*3600)  # check for debug
        h_6h = np.nanmedian(Cxy[idx2:idx1]) # median Cxy, hourly to 6-hourty

        idx1 = find_nearest_index(1 / freqs[1:], 6 * 3600)  # value1 = find_nearest_value(1 / freqs, 6 * 3600) # check for debug
        idx2 = find_nearest_index(1 / freqs[1:], 86400)  # value2 = find_nearest_value(1 / freqs, 86400)  # check for debug
        sixh_d = np.nanmedian(Cxy[idx2:idx1])  # median Cxy, 6-hourly to daily

        idx1 = find_nearest_index(1 / freqs[1:], 86400)  # value1 = find_nearest_value(1 / freqs, 86400) # check for debug
        idx2 = find_nearest_index(1 / freqs[1:], 7*86400)  # value2 = find_nearest_value(1 / freqs, 7 * 86400)  # check for debug
        d_w = np.nanmedian(Cxy[idx2:idx1])  # median Cxy, daily to weekly

        df_coherence[col] = [h_6h, sixh_d, d_w]

        plot_name = col + '_ts_coherence' + plot_file_type
        plot_name = os.path.join(output_path, plot_name)
        plt.savefig(plot_name)
        plt.clf()
        plt.close('all')

    return df_coherence

def lucas_script_tmp_4_copy():
    script_name = '2_plot_time_series_v3'
    # by Luke Easterbrook-Clarke 3/2022
    # reads a time series csv and creates plots for selected
    # or all columns over specified time period
    # produces a runtime error if saving png. but still works.

    import os
    import pandas as pd
    import datetime as dt
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import math

    ##USER INPUTS##

    # input_csv and plots_dir should be in work dir. Plots dir will be made if not already there.
    work_dir = 'E:/PythonProjects/pythonProject_Tauranga_Simon/files'
    input_csv = '20210916 GWData_partiallyNZSTcorrected_NaNremoved_cleaned_STATS.csv'
    time_col = 'Time'
    DOY_col = 'Day of Year'
    dataframe = os.path.join(work_dir, 'dataframe')

    # set to True to read a dataframe (df). Can be used after the first time you import a csv to save time.
    # set to False to read the csv and save a dataframe object.
    read_df = False

    # set start and end time in this format '2019-01-01'
    date_start = '2016-01-01'
    date_mid = '2020-01-01'  # mid used to highlight period in doy scatter (ignored for time series)
    date_end = '2021-01-01'

    # set this to True if only subset of plots is required.
    # enter name in plot_list as it appears in column header
    col_subset = False
    plot_list = ['GW1A']

    # comment out either below to select type
    # plot_type = 'time_series'
    plot_type = 'doy_scatter'

    # set y range limit on plots
    y_range = 4

    # plots
    plots_dir = 'plots4'
    # set plot size in inches (can upgrade to mm).
    plot_size = (8, 4)

    # Define the date format for plot (not used)
    date_form = mdates.DateFormatter("%Y-%m")

    # set type (good options png, pdf)
    plot_file_type = '.png'

    ##SCRIPT##
    # don't modify anything after here
    # set and make dataframe folder and path to file
    if not os.path.exists(dataframe):
        os.mkdir(dataframe)
    df_path = os.path.join(dataframe, 'data_df')

    # Either read dataframe or read csv and write dataframe.
    if read_df == True:
        print('Reading dataframe object')
        data_df = pd.read_pickle(df_path)
    else:
        print('Reading csv file')
        data_df = pd.read_csv(os.path.join(work_dir, input_csv), encoding='utf-8-sig')
        # convert time to timestamp
        data_df['Time'] = pd.to_datetime(data_df['Time'], dayfirst=True)
        # set the index to timestamp
        data_df.set_index('Time', inplace=True)
        print('Saving dataframe object')
        data_df.to_pickle(df_path)

    # make plot_list all of the if not using a subset.
    if col_subset == False:
        plot_list = [col for col in data_df]

    # add day of year column if making doy scatter plot
    if plot_type == 'doy_scatter':
        data_df[DOY_col] = data_df.index.dayofyear

    # set and make folder for outputs (if it doesn't exist already)
    output_path = os.path.join(work_dir, plots_dir)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    ##MAIN LOOP##
    # loop through the plot list
    print('Making plots')
    for col in plot_list:

        # get data for column in time period
        col_df = data_df.loc[date_start:date_end][col]

        # get min rounded down to nearest 0.5 increment
        col_min = math.floor(col_df.min() * 2.0) * 0.5

        # check which type of plot to do
        if plot_type == 'time_series':

            # produce the plot, set y limit ,remove Time label.
            col_plot = col_df.plot(title=col, figsize=plot_size)
            col_plot.set_ylim(col_min, col_min + y_range)
            col_plot.set_xlabel("")
            plt.xticks(rotation=0, ha='center')

            # set date format - not working -need to fix
            # col_plot.xaxis.set_major_formatter(date_form)
        elif plot_type == 'doy_scatter':
            ax1 = data_df.loc[date_start:date_mid].plot(kind='scatter', x=DOY_col, y=col,
                                                        label=date_start + ' to ' + date_mid, edgecolors='none', s=6)
            ax2 = data_df.loc[date_mid:date_end].plot(kind='scatter', x=DOY_col, y=col, color='orange', ax=ax1,
                                                      label=date_mid + ' to ' + date_end, legend=True, title=col,
                                                      figsize=plot_size, edgecolors='none', s=6)
            ax2.set_ylabel("")
            ax2.set_ylim(col_min, col_min + y_range)
            ax2.set_xlim(0, 366)
            plt.tight_layout()

        # set file name and save figure out
        plot_name = col + '_' + date_start + '_' + date_end + plot_file_type
        plt.savefig(os.path.join(output_path, plot_name))
        plt.clf()
        plt.close('all')
    print('Finished')
