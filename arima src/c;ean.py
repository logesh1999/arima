import pandas as pd

#idx = pd.date_range('2014-01-01', '2017-12-31')
df=pd.read_csv('normal.csv')


df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

#df['Order Date'] = pd.to_datetime(df['Order Date'], format = '%d-%m-%Y')
#df['Order Date'] = pd.to_datetime(df['Order Date'], format='%Y-%m-%d')
print(df)

#df.set_index(['Order Date'], inplace = True)

#df = df.reindex(idx,fill_value=0)
#df.to_csv('Actual.csv')