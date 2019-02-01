import os
import string
import time
import random

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from pandas import DataFrame, Series
from pandas_summary import DataFrameSummary
from clickhouse_driver.client import Client
from IPython.display import display

ch_fields = [
'SessionId',
'CampaignId',
'BannerId',
'AdWidth',
'AdHeight',
'VisitorId',
'ThirdPartyVisitorId',
'AdvertiserId',
'LandingDomain',
'BannerType',
'ConnectionType',
'PlacementCategories',
'PlacementViewability',
'TimezoneOffset',
'ScreenWidth',
'ScreenHeight',
'AppCategory',
'Language',
'PublisherId',
'SiteId',
'PlacementId',
'Domain',
'AppId',
'PlacementPosition',
'AdvertisedAppId',
'AdvertisedAppCategory',
'VisitorTotalImpressionsCount',
'VisitorTotalClicksCount',
'Date',
'DeviceType',
'DeviceCarrier',
'DeviceModel',
'Os',
'OsVersion',
'Browser',
'BrowserVersion',
'Country',
'Region',
'Sity',
'Gender',
'SspId']


def make_dataframe(views_data, click_data):
    ch_data_clicks_test = pd.DataFrame(click_data, columns =['SessionId'] )
    ch_data_views_test = pd.DataFrame(views_data, columns=ch_fields)
    ch_data_clicks_test['click'] = 1
    full_df_test = pd.merge(ch_data_views_test, ch_data_clicks_test, on="SessionId", how="left")
    full_df_test['click'] = np.where(full_df_test['click'] == 1, 1, 0)
    return full_df_test

def iter_download_from_clickhouse_to_csv(time_str, days_to_train, days_to_test):
    clickhouse_hosts = ['db101', 'db102', 'db103', 'db104', 'db105']
    cc = Client(random.choice(clickhouse_hosts), compression='lz4')

    # дату кликов смотрим дальше, чтобы догнать поздние клики
    clickhouse_query_template_clicks = """
    SELECT
    SessionId
    FROM statistic.DistributedSessions
    PREWHERE (Date >= '{0}') AND (Date < '{1}') AND (ClicksCount = 1) AND (SessionId != '')
    FORMAT CSV
    """

    clickhouse_query_template_views = """
    SELECT
    {0}
    FROM statistic.DistributedSessions
    PREWHERE (Date >= '{1}') AND (Date < '{2}') AND (ImpressionsCount = 1) AND (SessionId != '')
    FORMAT CSV
    """

    for i in range(0, days_to_train):
        if i==0:
            train_datetime_from = datetime(*time.strptime(time_str, '%Y-%m-%d %H:%M:%S')[:6])
        train_datetime_till = train_datetime_from + timedelta(days=1)
        train_datetime_till_clicks = train_datetime_till + timedelta(hours=2)
        views_train = cc.execute(clickhouse_query_template_views.format(', '.join(ch_fields), train_datetime_from, train_datetime_till))
        clicks_train = cc.execute(clickhouse_query_template_clicks.format(train_datetime_from, train_datetime_till_clicks))
        raw_train_df = make_dataframe(views_train, clicks_train)
        #raw_train_df = df_features(raw_train_df)
        if not os.path.isfile('train.csv'):
            raw_train_df.to_csv('train.csv', header='column_names')
        else: # else it exists so append without writing the header
            raw_train_df.to_csv('train.csv', mode='a', header=False)
        time_str = train_datetime_till

    for i in range(0, days_to_test):
        if i==0:
            test_datetime_from = datetime(*time.strptime(time_str, '%Y-%m-%d %H:%M:%S')[:6]) - timedelta(days=days_to_test)
        test_datetime_till = test_datetime_from + timedelta(days=1)
        test_datetime_till_clicks = test_datetime_till + timedelta(hours=2)
        views_test = cc.execute(clickhouse_query_template_views.format(', '.join(ch_fields), test_datetime_from, test_datetime_till))
        clicks_test = cc.execute(clickhouse_query_template_clicks.format(test_datetime_from, test_datetime_till_clicks))
        raw_test_df = make_dataframe(views_test, clicks_test)
        #raw_test_df = df_features(raw_test_df)
        if not os.path.isfile('test.csv'):
            raw_test_df.to_csv('test.csv', header='column_names')
        else: # else it exists so append without writing the header
            raw_test_df.to_csv('test.csv', mode='a', header=False)
        time_str = test_datetime_till
    cc.disconnect()

days_to_train = 1
days_to_test = 1
time_str = '2018-12-1 00:00:00'

iter_download_from_clickhouse_to_csv(time_str, days_to_train, days_to_test)
