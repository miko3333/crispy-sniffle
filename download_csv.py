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


dima_fields = [
"AdId",
"AdvertisedAppId",
"AdvertiserCategories",
"AppCategory",
"AppId",
"Browser",
"Device",
"DeviceType",
"AdvertiserId",
"VisitorDayOfWeek",
"OperatingSystemName",
"PageDomain",
"PageDomainAdvertiserId",
"PlacementCategories",
"PlacementPosition",
"PlacementSize",
"ScreenHeight",
"ScreenSize",
"ScreenWidth",
"ServerHour",
"Ssp",
"VisitorLanguage",
"VisitorRemarketingSegments"
]


fields = [
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

import os
import string
import json
from datetime import datetime, timedelta
import time
import random

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from clickhouse_driver.client import Client

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics



days_to_train = 14
hours_to_train = 0
days_to_test = 2
hours_to_test = 0


test_time_str = '2018-12-10 00:00:00'

test_datetime_from = datetime(*time.strptime( test_time_str, '%Y-%m-%d %H:%M:%S')[:6])
td = timedelta(days=days_to_test) if days_to_test else timedelta(hours=hours_to_test)
test_datetime_till = test_datetime_from + td
test_datetime_till_clicks = test_datetime_till + timedelta(hours=2)
print("Test")
print(f"""impressions:\nfrom  {test_datetime_from}\nuntil {test_datetime_till}\n
clicks:\nfrom  {test_datetime_from}\nuntil {test_datetime_till_clicks}\n""")

train_datetime_from = test_datetime_from - timedelta(days=days_to_train) if days_to_train else test_datetime_from - timedelta(hours=hours_to_train)
td = timedelta(days=days_to_train) if days_to_train else timedelta(hours=hours_to_train)
train_datetime_till = train_datetime_from + td
train_datetime_till_clicks = train_datetime_till + timedelta(hours=2)
print("Train")
print(f"""impressions:\nfrom  {train_datetime_from}\nuntil {train_datetime_till}\n
clicks:\nfrom  {train_datetime_from}\nuntil {train_datetime_till_clicks}""")



# дату кликов смотрим дальше, чтобы догнать позние клики
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




%%time
clickhouse_hosts = ['db101', 'db102', 'db103', 'db104', 'db105']

cc = Client(random.choice(clickhouse_hosts), compression='lz4')

views_train = cc.execute(clickhouse_query_template_views.format(', '.join(ch_fields), train_datetime_from, train_datetime_till))
clicks_train = cc.execute(clickhouse_query_template_clicks.format(train_datetime_from, train_datetime_till_clicks))

views_test = cc.execute(clickhouse_query_template_views.format(', '.join(ch_fields), test_datetime_from, test_datetime_till))
clicks_test = cc.execute(clickhouse_query_template_clicks.format(test_datetime_from, test_datetime_till_clicks))

cc.disconnect()



len(views_train), len(views_test)
if views_train < views_test:
    views_train, views_test = views_test, views_train




def make_dataframe(views_data, click_data):
    ch_data_clicks_test = pd.DataFrame(click_data, columns =['SessionId'] )
    ch_data_views_test = pd.DataFrame(views_data, columns=ch_fields)
    ch_data_clicks_test['click'] = 1
    full_df_test = pd.merge(ch_data_views_test, ch_data_clicks_test, on="SessionId", how="left")
    full_df_test['click'] = np.where(full_df_test['click'] == 1, 1, 0)
    return full_df_test


raw_train_df = make_dataframe(views_train, clicks_train)
raw_test_df = make_dataframe(views_test, clicks_test)
print(f'train shape {raw_train_df.shape}, clicks = {raw_train_df.click.sum()}, ctr = {(raw_train_df.click.sum() * 100 / raw_train_df.shape[0]):.3f}%')
print(f'test shape {raw_test_df.shape}, clicks = {raw_test_df.click.sum()}, ctr = {(raw_test_df.click.sum() * 100 / raw_test_df.shape[0]):.3f}%')


def df_features(data):
    data.rename(columns={"Sity": "City"})
    data.loc[:,'BrowserInfo'] = data['Browser'] + data['BrowserVersion']
    data.loc[:,'OsInfo'] = data['Os'] + data['OsVersion']
    data.loc[:,'AdSize'] = data['AdWidth'].apply(str) + 'x' + data['AdHeight'].apply(str)
    data.loc[:,'ScreenSize'] = data['ScreenWidth'].apply(str) + 'x' + data['ScreenHeight'].apply(str)
    data.loc[:,'Device'] = data['DeviceCarrier'] + data['DeviceModel']
    data['TimezoneOffset'].fillna(0, inplace=True)
    data.loc[:,'Date'] = pd.to_datetime(data['Date'])
    data.loc[:,'UserDate'] = data['Date'] - pd.to_timedelta(data['TimezoneOffset'], 'h')
    data.loc[:,'UserHour'] = data['UserDate'].dt.hour
    data.loc[:,'UserDayOfWeek'] = data['UserDate'].dt.dayofweek
    data.drop(['Browser', 'BrowserVersion', 'Os', 'OsVersion', 'Date', 'Os',
               'OsVersion', 'AdWidth', 'AdHeight', 'ScreenWidth', 'ScreenHeight',
               'DeviceCarrier', 'DeviceModel', 'TimezoneOffset', 'Date', 'UserDate'], axis=1, inplace=True)
    cols = list(data.columns.values)
    cols.pop(cols.index('click'))
    data = data[cols[1:]+['click']]
    return data

raw_train_df = df_features(raw_train_df)
raw_test_df = df_features(raw_test_df)


def pd_to_format(format_name, input_df, name):
    pd.options.mode.chained_assignment = None

    input_df.replace('https://', '', inplace = True)
    input_df.replace('http://', '', inplace = True)
    input_df.replace(':', '.', inplace = True)

    if format_name == 'vw':
        input_df.replace({'click': {0: -1}}, inplace=True)
        input_df.replace('', '__null__', inplace=True)

        input_df.loc[:,'VW'] = input_df['click'].astype(str)

        for i, j in zip(list(input_df.columns), list(string.ascii_letters)):
            if i == 'click':
                pass
            else:
                input_df['VW'] += ' |' + j + ' ' + input_df.loc[:,i].astype(str)

        with open('click-dumps/' + name + '.vw', 'a') as tfile:
            tfile.write(input_df['VW'].to_string(header=False, index=False))
            tfile.close()

        return True

    if format_name == 'json': #разобраться с неймспейсами

        data = input_df.rename(columns={'click':'_label'})
        data.replace({'_label': {0: -1}}, inplace=True)
        out = data.to_json(orient='records')

        with open('click-dumps/' + name + '.json', 'w') as outfile:
            json.dump(out, outfile)

        return True

    if format_name == 'csv':
        input_df.to_csv('click-dumps/' + name + '.csv', index=False)

        return True

    else:
        print('Wrong format')



pd_to_format('csv', raw_train_df, 'train')
pd_to_format('csv', raw_test_df, 'test')
