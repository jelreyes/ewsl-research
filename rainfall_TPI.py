# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:57:25 2024

@author: JSR
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import numpy as np
import os
import sql_connector as db

def get_rain_gauge(sql=True, to_csv=False):
    if sql:
        query_rg = "SELECT gauge_name, date_activated, date_deactivated "
        query_rg += "FROM analysis_db.rainfall_gauges "
        query_rg += "WHERE data_source = 'senslope' " #get senslope-installed rain gauges only 
        rg = db.df_read(query_rg)
        if to_csv:
            rg.to_csv(output_path + 'raingauge.csv')
    else:
        rg = pd.read_csv(output_path + 'raingauge.csv')
    return rg

def get_threshold(sql=True, to_csv=False):
    if sql:
        query_threshold = "SELECT site_code, threshold_value  "
        query_threshold += "FROM analysis_db.rainfall_thresholds "
        query_threshold += "LEFT JOIN commons_db.sites USING (site_id) "
        threshold = db.df_read(query_threshold)
        if to_csv:
            threshold.to_csv(output_path + 'threshold.csv')
    else:
        threshold = pd.read_csv(output_path + 'threshold.csv')
    return threshold

def get_rain_data(start, end, sql=True, to_csv=False):
    """Gets rainfall data from all identified ARGs in get_rain_gauge(), resamples
        every 30 mins, and computes one-day and three-day cumuulative rainfall
        values
    
        Args:
            end (datetime): for rainfall data query
                
        Returns:
            all_rain (dataframe): summary of rainfall data for all sites; contains ts, data_id, rain, source, OneDayCumulative,
                   ThreeDayCumulative, site_code, threshold_value
    """
    
    rain_data = [] #for resampled rain data
    true_rain_data = [] #for counting true/raw data
    
    rg = get_rain_gauge()
    
    #get rainfall datafrom gauge name in rg, resample every 30 mins, fill NaN values with 0, append to empty df
    for index, row in rg.iterrows():
        gauge_name = row['gauge_name']
        date_activated = row['date_activated']
        date_deactivated = row['date_deactivated']
        
        if sql:
            query_rain = "SELECT data_id, ts, rain "
            query_rain += "FROM analysis_db.rain_{} ".format(gauge_name)
            query_rain += "WHERE ts NOT IN ( "
            query_rain += "SELECT ts FROM analysis_db.rain_{} ".format(gauge_name)
            query_rain += "GROUP BY ts "
            query_rain += "HAVING count(*)>1) "
            query_rain += "AND rain >= 0 "   
            # Check if date_deactivated is not null (active)
            if pd.notna(date_deactivated):
                query_rain += " AND ts BETWEEN '{date_activated}' AND '{date_deactivated}'".format(date_activated=date_activated, date_deactivated=date_deactivated)
            else:
                query_rain += " AND ts BETWEEN '{date_activated}' AND '{end}'".format(date_activated=date_activated, end=end)
            rain = db.df_read(query_rain)
            if to_csv:
                rain.to_csv(output_path + f"rain_{gauge_name}.csv")
        else:
            rain = pd.read_csv(output_path + f"rain_{gauge_name}.csv")
        rain['source'] = gauge_name
        rain['ts'] = pd.to_datetime(rain['ts'])
        
        true_rain_data.append(rain)
        
        ############################ resampled rain ###############################################
        resampled_rain = rain.set_index('ts').resample('30T').sum()
        resampled_rain['rain'].fillna(0, inplace=True)
        # cap to 30mm (torrential)
        resampled_rain.loc[resampled_rain['rain'] > 30, 'rain'] = 30
        
        ############################ cumulative value #############################################
        resampled_rain['OneDayCumulative'] = resampled_rain['rain'].rolling('1D').sum()
        resampled_rain['ThreeDayCumulative'] = resampled_rain['rain'].rolling('3D').sum()
    
        #Fill source
        for index, row in resampled_rain.iterrows():
            if row['source'] == 0:
                prev_index = resampled_rain.index.get_loc(index) - 1
                next_index = resampled_rain.index.get_loc(index) + 1
        
                prev_value = None
                next_value = None
        
                if prev_index >= 0:
                    prev_value = resampled_rain.at[resampled_rain.index[prev_index], 'source']
        
                if next_index < len(resampled_rain):
                    next_value = resampled_rain.at[resampled_rain.index[next_index], 'source']
        
                if prev_value is not None and prev_value != 0:
                    resampled_rain.at[index, 'source'] = prev_value
                elif next_value is not None and next_value != 0:
                    resampled_rain.at[index, 'source'] = next_value
    
        rain_data.append(resampled_rain)
    
    all_rain = pd.concat(rain_data, ignore_index=False)
    all_rain.sort_values(['source','ts'], inplace = True)
    all_rain['source'] = all_rain['source'].replace({0: np.nan}).fillna(method='ffill')
    
    #get site_code since each rain data table is not connected to site table
    all_rain['site_code'] = all_rain['source'].str[:3]
    all_rain = all_rain.reset_index()
    all_rain = all_rain.drop_duplicates(subset=['ts', 'site_code'])
    all_rain.sort_values(['source','ts'], inplace = True)
    
    threshold = get_threshold()    
    all_rain = all_rain.join(threshold.set_index('site_code'), on='site_code')
    
    #true data vs resampled
    true_rain_data = pd.concat(true_rain_data, ignore_index=False)
    true_rain_data['site_code'] = true_rain_data['source'].str[:3]
    true_rain_data = true_rain_data.reset_index()
    true_rain_data = true_rain_data.drop_duplicates(subset=['ts', 'site_code'])
    true_rain_data.sort_values(['source','ts'], inplace = True)
    resampled_data = all_rain.groupby('site_code').size().reset_index(name='resampled_count')
    true_data = true_rain_data.groupby('site_code').size().reset_index(name='true_count')
    percent_true = pd.merge(true_data, resampled_data, on='site_code')
    percent_true['true_data_percentage'] = percent_true['true_count'] / percent_true['resampled_count']
    percent_true.to_csv(output_path + 'percent_true.csv')

    return all_rain

def round_up_to_interval(ts):
    
    """round ts up to 4/8/12 AM/PM interval (4-hourly release of early warning information)"""
    
    hour = ts.hour
    rounded_up = ts + pd.to_timedelta(4 - (hour % 4), unit='h')
    
    return rounded_up.replace(minute=0, second=0, microsecond=0)

def rainfall_alert(start, end=None, to_csv=False):
    
    """Get rainfall alert based on rainfall threshold exceedance

        Returns:
            rainfall_alerts (dataframe): rainfall data exceeding threshold for all sites;
                    contains ts, data_id, rain, source, OneDayCumulative,
                   ThreeDayCumulative, site_code, threshold_value
    """
    
    all_rain = get_rain_data(start, end, to_csv, )
    
    rainfall_alerts = all_rain[
        (all_rain['OneDayCumulative'] > ((all_rain['threshold_value']/2))) 
        | (all_rain['ThreeDayCumulative'] > (all_rain['threshold_value']))]
    rainfall_alerts = rainfall_alerts.reset_index()
    
    return rainfall_alerts

def get_nonexceeding(start, end=None, to_csv=False):
    
    """Get instances when rainfall value is more than 75% but less than 100% of
        rainfall threshold 
    
        Returns:
            nonexceeding (dataframe):  rainfall data with values between 75% - 100% of the
                threshold for all sites; contains ts, data_id, rain, source,
                OneDayCumulative, ThreeDayCumulative, site_code, threshold_value
    """
    
    all_rain = get_rain_data(start, end, to_csv)

    nonexceeding = all_rain[(all_rain['OneDayCumulative'].between((((all_rain['threshold_value'])/2)*0.75), ((all_rain['threshold_value'])/2))) 
                  | (all_rain['ThreeDayCumulative'].between(((all_rain['threshold_value'])*0.75), ((all_rain['threshold_value']))))]
    nonexceeding = nonexceeding.reset_index() 
    
    return nonexceeding


def rainfall_event(start, end=None, to_csv=False):    
    
    """Create rainfall event based on current protocol
    
        Returns:
            rainfall_events (dataframe): site_code, event_id, start, end
    """
    
    rainfall_alerts = rainfall_alert(start, end, to_csv)
    rainfall_alerts = rainfall_alerts.groupby('site_code')
    
    rainfall_events = {'site_code': [], 'event_id': [], 'start': [], 'end': []}
    event_id = 1
    
    for site_code, site_alerts in rainfall_alerts:
        event_start = None
        event_end = None
        for index, row in site_alerts.iterrows():
            if event_start is None:
                event_start = row['ts']
                event_end = round_up_to_interval(row['ts']) + timedelta(days=1)
            elif row['ts'] > event_end:
                rainfall_events['site_code'].append(site_code)
                rainfall_events['event_id'].append(event_id)
                rainfall_events['start'].append(event_start)
                rainfall_events['end'].append(event_end)
                event_id += 1
                event_start = row['ts']
                event_end = round_up_to_interval(row['ts']) + timedelta(days=1)
            else:
                event_end = max(event_end, round_up_to_interval(row['ts']) + timedelta(days=1))
    
        if event_start is not None and event_end is not None:
            rainfall_events['site_code'].append(site_code)
            rainfall_events['event_id'].append(event_id)
            rainfall_events['start'].append(event_start)
            rainfall_events['end'].append(event_end)
            event_id += 1
    
    rainfall_events = pd.DataFrame(rainfall_events)

    return rainfall_events

def ground_movement(start, end, sql=True, to_csv=False):
    if sql:
        query_movement = "SELECT event_id, site_code, ANY_VALUE(trigger_list) as triggers, "
        query_movement += "ANY_VALUE(ts) AS trigger_ts, event_start, validity "
        query_movement += "FROM ewi_db.monitoring_releases "
        query_movement += "LEFT JOIN ewi_db.monitoring_triggers USING (release_id) "
        query_movement += "LEFT JOIN ewi_db.monitoring_event_alerts using (event_alert_id) "
        query_movement += "LEFT JOIN ewi_db.monitoring_events using (event_id) "
        query_movement += "LEFT JOIN commons_db.sites using (site_id) "
        query_movement += "WHERE trigger_list REGEXP 's|g|m' "
        query_movement += " AND event_start > '{start}' ".format(start=start)
        query_movement += " AND validity <'{end}'".format(end=end)
        query_movement += "GROUP BY event_id "
        movement = db.df_read(query_movement)
        if to_csv:
            movement.to_csv(output_path + 'movement.csv')
    else:
        movement = pd.read_csv(output_path + 'movement.csv')

    return movement

def confusion_matrix(start, end=None, to_csv=False):
    
    """rainfall data-based confusion matrix
    
        Returns:
            TP_site, FP_site, TN_site, FN_site (series): confusion matrix for each site
            
    """
    movement = ground_movement(start, end)
    
    rainfall_events = rainfall_event(start, end, to_csv)
    rainfall_nonexceeding = get_nonexceeding(start, end, to_csv)
    events = pd.merge(rainfall_events, movement, on='site_code', suffixes=('_rainfall_events', '_movement'))
    non_events = pd.merge(rainfall_nonexceeding, movement, on='site_code', suffixes=('_rainfall_rx', '_movement'))
    
    #True positive - with valid rainfall trigger and ground movement    
    TP = events[events.apply(lambda row: row['trigger_ts'] >= row['start'] and row['trigger_ts'] <= row['end'], axis=1)]   
    TP_site = TP.groupby('site_code').size()
    
    #False positive - with valid rainfall trigger without ground movement
    FP = events[~events.apply(lambda row: row['trigger_ts'] >= row['start'] and row['trigger_ts'] <= row['end'], axis=1)]
    TPs = TP.reset_index(drop=True)
    FP = (FP[~FP['event_id_rainfall_events'].isin(TPs['event_id_rainfall_events'])]
        .groupby(['site_code', 'event_id_rainfall_events', 'start', 'end'])
        .size())
    FP_site = FP.groupby('site_code').size()
    
    #True negative - with rainfall but not triggering (>75%, <100%), and with no ground movement        
    TN = non_events[~non_events.apply(lambda row: row['ts'] >= row['event_start'] and row['ts'] <= row['validity'], axis=1)]
    TN = TN.sort_values(by=['site_code', 'ts'])
    # Calculate the time difference between consecutive rows within each 'site_code' group
    TN['time_diff'] = TN.groupby('site_code')['ts'].diff()
    TN['time_diff'] = TN['time_diff'].fillna(pd.to_timedelta('0 days 00:00:00'))
    # Identify consecutive instances with less than or equal to 30 minutes interval
    TN['less_30'] = ~((TN['time_diff'] > pd.Timedelta('30 minutes')) | (TN['time_diff'].isnull()))
    TN['is_consecutive'] = ~TN['less_30'].shift(fill_value=False)
    TN.loc[TN['less_30'] == False, 'is_consecutive'] = True
    TN_site = TN[TN['is_consecutive']].groupby('site_code').size().reset_index(name='count_true')
    TN_site = TN_site.set_index('site_code')['count_true']

    #False negative - ground movement without rainfall trigger - g|s|m
    FN = events[events.apply(lambda row: row['trigger_ts'] < row['start'] or row['trigger_ts'] > row['end'], axis=1)]    
    TPs = TP.reset_index(drop=True)
    FN = (FN[~FN['event_id_movement'].isin(TPs['event_id_movement'])]
        .groupby(['site_code', 'event_id_movement', 'triggers', 'trigger_ts'])
        .size())  
    FN_site = FN.groupby('site_code').size()
    
    return TP_site, FP_site, TN_site, FN_site
    
def ROC(start, end=None, sql=True to_csv=False):   
    """Summary of confusion matrix and derived skills scores for each site
            
        Returns:
            Matrix (dataframe): TP, FP, TN, FN, TPR, FPR, FNR, TNR, Prevalence,
            Precision, NPV, F1_score          
    """
    
    TP_site, FP_site, TN_site, FN_site = confusion_matrix(start, end, to_csv)
    
    matrix = pd.concat([TP_site, FP_site, TN_site, FN_site], axis=1, keys=['TP','FP','TN','FN'])
    matrix = matrix.fillna(0).astype(int)
    
    # Calculate TPR and FPR using vectorized operations
    matrix['TPR'] = matrix['TP'] / (matrix['TP'] + matrix['FN'])
    matrix['FPR'] = matrix['FP'] / (matrix['FP'] + matrix['TN'])
    matrix['FNR'] = matrix['FN'] / (matrix['FN'] + matrix['TP'])
    matrix['TNR'] = matrix['TN'] / (matrix['TN'] + matrix['FP'])
    matrix['Prevalence'] = (matrix['TP'] + matrix['FN']) / (matrix['TP'] + matrix['TN'] + matrix['FP'] + matrix['FN'])
    matrix['Precision'] = matrix['TP'] / (matrix['TP'] + matrix['FP'])
    matrix['NPV'] = matrix['TN'] / (matrix['TN'] + matrix['FN'])
    matrix['F1_score'] = (2 * matrix['Precision'] * matrix['TPR']) / (matrix['Precision'] + matrix['TPR'])     
    
    matrix = matrix.sort_values(by='site_code')
    matrix.to_csv(output_path + 'prc_curve.png')
    
    return matrix

def ROC_plot(start, end=None, to_csv=False):

    matrix = ROC(start, end, to_csv)
    print(matrix)
    
    x = matrix['TPR']
    y = matrix['FPR']
    
    ROC_percentage = ""

    above_line_count = sum(y < x)
    below_line_count = sum(y > x)
    on_line_count = sum(x == y)
    total_points = len(x)
    
    above_line_percentage = (above_line_count / total_points) * 100
    below_line_percentage = (below_line_count / total_points) * 100
    on_line_percentage = (on_line_count / total_points) * 100
        
    print('---------------ROC---------------')
    print("Percentage above the line: {:.2f}%".format(above_line_percentage))
    print("Percentage below the line: {:.2f}%".format(below_line_percentage))
    print("Percentage on the line: {:.2f}%".format(on_line_percentage))
    
    plt.figure(figsize=(7, 5))
    plt.scatter(matrix['FPR'], matrix['TPR'], c=(0.5607843137254902, 0.6666666666666666, 0.8627450980392157),
                edgecolors='darkblue', linewidths=0.7, alpha=0.8, zorder=2)
    
    max_x = 1
    max_y = 1
    plt.xlim(0, max_x)
    plt.ylim(0, max_y)
    
    #45-degree line
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', zorder=1)
    
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Threshold ROC')
    
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['', 0.2, 0.4, 0.6, 0.8, 1])
    
    ROC_percentage += "Above the line: {:.2f}% \n".format(above_line_percentage)
    ROC_percentage += "Below the line: {:.2f}%".format(below_line_percentage)
    
    text = "{ROC_percentage}".format(ROC_percentage=ROC_percentage)
    plt.text(0.67, 0.07, text, fontsize=10, bbox=dict(facecolor='white', edgecolor='none'))
    
    plt.savefig(output_path + 'roc_curve.png', dpi=400, bbox_inches='tight')
    plt.show()
    
def precision_recall(start, end=None, to_csv=False):
    
    matrix = ROC(start, end, to_csv)
    
    plt.figure(figsize=(7, 5))
    plt.scatter(matrix['TPR'], matrix['Precision'], c=(0.5607843137254902, 0.6666666666666666, 0.8627450980392157),
                edgecolors='darkblue', linewidths=0.7, alpha=0.8, zorder=2)
    
    max_x = 1
    max_y = 1
    plt.xlim(0, max_x)
    plt.ylim(0, max_y)
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Threshold Precision-Recall')
    
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['', 0.2, 0.4, 0.6, 0.8, 1])
    
    #baseline
    precision_ave = matrix.loc[:, 'Precision'].mean()
    pr_results = (len(matrix[matrix['Precision'] > precision_ave]) / len(matrix)) * 100
    text = "Above the line: {:.2f}%".format(pr_results)
    plt.text(0.68, 0.86, text, fontsize=10, bbox=dict(facecolor='white', edgecolor='none'))
    plt.axhline(y=0.12, color='red', linestyle='--', label='baseline {:.2f}%'.format(precision_ave), linewidth=0.7, alpha= 0.8, zorder=1)
    plt.legend(frameon=False)

    print('-------------Precision-recall-------------')
    print("Percentage above the line: {:.2f}%".format(pr_results))    

    plt.savefig(output + 'prc_curve.png', dpi=400, bbox_inches='tight')
    plt.show()
    
    matrix = ROC()
    
    plt.figure(figsize=(7, 5))
    plt.scatter(matrix['TPR'], matrix['Precision'], c=(0.5607843137254902, 0.6666666666666666, 0.8627450980392157),
                edgecolors='darkblue', linewidths=0.7, alpha=0.8, zorder=2)
    
    max_x = 1
    max_y = 1
    plt.xlim(0, max_x)
    plt.ylim(0, max_y)
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Threshold Precision-Recall')
    
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['', 0.2, 0.4, 0.6, 0.8, 1])
    
    #baseline
    precision_ave = matrix.loc[:, 'Precision'].mean()
    pr_results = (len(matrix[matrix['Precision'] > precision_ave]) / len(matrix)) * 100
    text = "Above the line: {:.2f}%".format(pr_results)
    plt.text(0.68, 0.86, text, fontsize=10, bbox=dict(facecolor='white', edgecolor='none'))
    plt.axhline(y=precision_ave, color='red', linestyle='--', label='baseline {:.2f}%'.format(precision_ave), linewidth=0.7, alpha= 0.8, zorder=1)
    plt.legend(frameon=False)

    print('-------------Precision-recall-------------')
    print("Percentage above the line: {:.2f}%".format(pr_results))    

    plt.savefig(output_path + 'prc_curve.png', dpi=400, bbox_inches='tight')
    plt.show()
    
#################################################################################################

if __name__ == '__main__':
    run_start = datetime.now()

    start = pd.to_datetime('2016-09-16 00:00:00')
    end = pd.to_datetime('2022-07-14 00:00:00')
    to_csv = False
    sql = True
    output_path = os.path.dirname(os.path.abspath(__file__))
        
    ROC(start, end, sql, to_csv)
    ROC_plot()
    precision_recall()
    
    runtime = datetime.now() - run_start
    print("runtime = {}".format(runtime))