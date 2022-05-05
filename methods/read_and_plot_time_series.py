def lucas_script():
    script_name = '2_plot_time_series_v3'
    #by Luke Easterbrook-Clarke 3/2022
    #reads a time series csv and creates plots for selected
    #or all columns over specified time period
    #produces a runtime error if saving png. but still works.

    import os
    import pandas as pd
    import datetime as dt
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import math

    ##USER INPUTS##

    #input_csv and plots_dir should be in work dir. Plots dir will be made if not already there.
    work_dir = 'E:/PythonProjects/pythonProject_Tauranga_Simon/files'
    input_csv = '20210916 GWData_partiallyNZSTcorrected_NaNremoved_cleaned_STATS.csv'
    time_col = 'Time'
    DOY_col = 'Day of Year'
    dataframe = os.path.join(work_dir,'dataframe')

    #set to True to read a dataframe (df). Can be used after the first time you import a csv to save time.
    #set to False to read the csv and save a dataframe object.
    read_df = True

    #set start and end time in this format '2019-01-01'
    date_start = '2016-01-01'
    date_mid = '2020-01-01'#mid used to highlight period in doy scatter (ignored for time series)
    date_end = '2021-01-01'

    #set this to True if only subset of plots is required.
    #enter name in plot_list as it appears in column header
    col_subset = False
    plot_list = ['GW1A']

    #comment out either below to select type
    plot_type = 'time_series'
    # plot_type = 'doy_scatter'

    #set y range limit on plots
    y_range = 4

    #plots
    plots_dir = 'plots5'
    #set plot size in inches (can upgrade to mm).
    plot_size = (8,4)

    # Define the date format for plot (not used)
    date_form = mdates.DateFormatter("%Y-%m")

    #set type (good options png, pdf)
    plot_file_type = '.png'

    ##SCRIPT##
    #don't modify anything after here
    #set and make dataframe folder and path to file
    if not os.path.exists(dataframe):
        os.mkdir(dataframe)
    df_path = os.path.join(dataframe, 'data_df')

    #Either read dataframe or read csv and write dataframe.
    if read_df == True:
        print('Reading dataframe object')
        data_df = pd.read_pickle(df_path)
    else:
        print('Reading csv file')
        data_df = pd.read_csv(os.path.join(work_dir,input_csv),encoding='utf-8-sig')
        #convert time to timestamp
        data_df['Time'] = pd.to_datetime(data_df['Time'],dayfirst=True)
        #set the index to timestamp
        data_df.set_index('Time',inplace=True)
        print('Saving dataframe object')
        data_df.to_pickle(df_path)


    #make plot_list all of the if not using a subset.
    if col_subset == False:
        plot_list = [col for col in data_df]

    #add day of year column if making doy scatter plot
    if plot_type == 'doy_scatter':
        data_df[DOY_col] = data_df.index.dayofyear

    #set and make folder for outputs (if it doesn't exist already)
    output_path = os.path.join(work_dir, plots_dir)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    ##MAIN LOOP##
    #loop through the plot list
    plot_my_files = True
    if plot_my_files:
        print('Making plots')
        for col in plot_list:

            #get data for column in time period
            col_df = data_df.loc[date_start:date_end][col]

            #get min rounded down to nearest 0.5 increment
            col_min = math.floor(col_df.min()*2.0)*0.5


            #check which type of plot to do
            if plot_type == 'time_series':

                #produce the plot, set y limit ,remove Time label.
                col_plot = col_df.plot(title = col, figsize = plot_size)
                col_plot.set_ylim(col_min, col_min + y_range)
                col_plot.set_xlabel("")
                plt.xticks(rotation=0,ha='center')

                #set date format - not working -need to fix
                #col_plot.xaxis.set_major_formatter(date_form)
            elif plot_type == 'doy_scatter':
                ax1 = data_df.loc[date_start:date_mid].plot(kind ='scatter', x=DOY_col,y=col, label =date_start+' to '+date_mid,edgecolors='none',s=6)
                ax2 = data_df.loc[date_mid:date_end].plot(kind ='scatter', x=DOY_col,y=col,color='orange',ax=ax1, label =date_mid+' to '+date_end,legend=True,title = col,figsize = plot_size, edgecolors='none',s=6)
                ax2.set_ylabel("")
                ax2.set_ylim(col_min, col_min + y_range)
                ax2.set_xlim(0,366)
                plt.tight_layout()

            #set file name and save figure out
            plot_name = col+'_'+date_start+'_'+date_end+plot_file_type
            plt.savefig(os.path.join(output_path,plot_name))
            plt.clf()
            plt.close('all')
        print('Finished')

