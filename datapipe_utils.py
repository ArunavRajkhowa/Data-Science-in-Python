#!/usr/bin/env python

''' template for the classes 
class SomeDataProcess(BaseEstimator, TransformerMixin) : 
    
    def __init__(self, other_inputs ...):
        
        # func body 
        
    def fit(self , x , y=None):
        
        # here the learning happens 
        
    def transform ( self , x , y= None ):
        
        # here the data transformation happens 
        
    def get_feature_names(self):
        
        # this returns the column headers for transformed data
'''

# var selector
# convert to numeric 
# custom var
# create dummies 
# missing values

import pandas as pd 
import numpy as np

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion

class VarSelector(BaseEstimator, TransformerMixin):

    def __init__(self,feature_names):

        self.feature_names=feature_names


    def fit(self,x,y=None):

        return self

    def transform(self,x):

        return x[self.feature_names]

    def get_feature_names(self):

        return self.feature_names


class CustomVar(BaseEstimator,TransformerMixin):

    def __init__(self,func,var):

        self.func=func
        self.feature_names=[var]
        self.var=var

    def fit(self,x,y=None):

        return self

    def transform(self,x):
        
        return pd.DataFrame({self.var:self.func(x[self.var])})

    def get_feature_names(self):

        return self.feature_names


class ConvertNumeric(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.feature_names=[]

    def fit(self,x,y=None):
        self.feature_names=x.columns
        return self

    def transform(self,x):
        for col in x.columns:
            x.loc[:,col]=pd.to_numeric(x[col],errors='coerce')
        return x
    def get_feature_names(self):
        return self.feature_names


class CreateDummies(BaseEstimator, TransformerMixin):

    def __init__(self,freq_cutoff=0):

        self.freq_cutoff=freq_cutoff
        self.var_cat_dict={}
        self.feature_names=[]

    def fit(self,x,y=None):

        data_cols=x.columns

        for col in data_cols:

            k=x[col].value_counts()

            if (k<=self.freq_cutoff).sum()==0:
                cats=k.index[:-1]

            else:
                cats=k.index[k>self.freq_cutoff]

            self.var_cat_dict[col]=cats

        for col in self.var_cat_dict.keys():
            for cat in self.var_cat_dict[col]:
                self.feature_names.append(col+'_'+cat)
        return self

    def transform(self,x,y=None):
        dummy_data=x.copy()

        for col in self.var_cat_dict.keys():
            for cat in self.var_cat_dict[col]:
                name=col+'_'+cat
                dummy_data[name]=(dummy_data[col]==cat).astype(int)

            del dummy_data[col]
        return dummy_data

    def get_feature_names(self):

        return self.feature_names

class DataFrameImputer(BaseEstimator,TransformerMixin):

    def __init__(self):

        self.impute_dict={}
        self.feature_names=[]

    def fit(self, x, y=None):

        self.feature_names=x.columns

        for col in x.columns:
            if x[col].dtype=='O':
                self.impute_dict[col]='missing'
            else:
                self.impute_dict[col]=x[col].median()
        return self

    def transform(self, x, y=None):
        return x.fillna(self.impute_dict)

    def get_feature_names(self):

        return self.feature_names

class CyclicFeatures(BaseEstimator,TransformerMixin):

    def __init__(self):

        self.feature_names=[]
        self.week_freq=7
        self.month_freq=12
        self.month_day_freq=31

    def fit(self,x,y=None):

        for col in x.columns:

            for kind in ['week','month','month_day']:

                self.feature_names.extend([col + '_'+kind+temp for temp in ['_sin','_cos']])

        return self 

    def transform(self,x):

        for col in x.columns:

            x[col]=pd.to_datetime(x[col])
            
            wdays=x[col].dt.dayofweek
            month=x[col].dt.month
            day=x[col].dt.day

            x[col+'_'+'week_sin']=np.sin(2*np.pi*wdays/self.week_freq)
            x[col+'_'+'week_cos']=np.cos(2*np.pi*wdays/self.week_freq)

            x[col+'_'+'month_sin']=np.sin(2*np.pi*month/self.month_freq)
            x[col+'_'+'month_cos']=np.cos(2*np.pi*month/self.month_freq)

            x[col+'_'+'month_day_sin']=np.sin(2*np.pi*day/self.month_day_freq)
            x[col+'_'+'month_day_cos']=np.cos(2*np.pi*day/self.month_day_freq)

            del x[col]

        return x

    def get_feature_names(self):

        return self.feature_names

class DateDiffs(BaseEstimator,TransformerMixin):

    def __init__(self):

        self.feature_names=[]

    def fit(self,x,y=None):

        num_cols=len(x.columns)

        for i in range(num_cols-1):

            for j in range(i+1,num_cols):

                name=x.columns[i]+'_diff_with_'+x.columns[j]

                self.feature_names.append(name)

        return self


    def transform(self,x):

        cols=x.columns
        num_cols=len(cols)

        for col in cols:

            x[col]=pd.to_datetime(x[col])

        for i in range(num_cols-1):

            for j in range(i+1,num_cols):

                name=x.columns[i]+'_diff_with_'+x.columns[j]

                x[name]=(x[cols[i]]-x[cols[j]]).dt.days

        for col in cols:

            del x[col]

        return x


    def get_feature_names(self):

        return self.feature_names

class TextFeatures(BaseEstimator,TransformerMixin):

    def __init__(self):

        self.feature_names=[]
        self.tfidfs={}

    def fit(self,x,y=None):

        for col in x.columns:

            self.tfidfs[col]=TfidfVectorizer(analyzer='word',stop_words='english',
                token_pattern=r'(?u)\b[A-Za-z]+\b',min_df=0.01,max_df=0.8,max_features=200)
            self.tfidfs[col].fit(x[col])
            self.feature_names.extend([col+'_'+_ for _ in list(self.tfidfs[col].get_feature_names_out())])

        return self


    def transform(self,x):

        datasets={}

        for col in x.columns:

            datasets[col]=pd.DataFrame(data=self.tfidfs[col].transform(x[col]).toarray(),
                                        columns=[col+'_'+_ for _ in list(self.tfidfs[col].get_feature_names_out())])

        return pd.concat(datasets.values(),axis=1)

    def get_feature_names(self):

        return self.feature_names

class SummaryFeatures:

    def __init__(self,data,summarize_at=None,cat_counts=None,num_unique=None,top_cat=None,age=None,
                    last_interaction=None,intervals_summary=None,num_summary=None,
                    ref_date_age=None,ref_date_last_interaction=None):


        self.summarize_at=summarize_at
        self.cat_counts=cat_counts
        self.num_unique=num_unique
        self.top_cat=top_cat
        self.age=age
        self.last_interaction=last_interaction
        self.ref_date_age=ref_date_age
        self.ref_date_last_interaction=ref_date_last_interaction
        self.intervals_summary=intervals_summary
        self.num_summary=num_summary
        self.data=data

        self._base_data=data[self.summarize_at].value_counts().reset_index()
        self._base_data.columns=[self.summarize_at,self.summarize_at+'_count']

    ''' 
        cat_counts : columns containing number of occurences of unique values of columns in cat_counts
                     across different values of summarise_at
        num_unique : number of unique values of a column within distinct values of summarise_at
        top_cat : top category by frequency for cols of top_cat across distinct values of summarise_at
        age : summarize_at age as per the date column given for age argument , if ref_date_age is None then current date
             is used to take difference 
        last_interaction : time since last interaction as per the date columns in last_interaction, if ref_date_LI is None 
                then current date is used to take difference
        intervals_summary : summary of intervals between dates as per date columns in intervals_summary . 
                        mean , std , min ,q1,q2,q3,max
        num_summary : mean , std , min ,q1,q2,q3,max for cols in num_summary across summarize_at


    '''

    def _num_unique(self):

        bd=self._base_data

        for col in self.num_unique:

            temp=self.data.groupby([self.summarize_at])[col].nunique().reset_index()
            bd=pd.merge(bd,temp,on=self.summarize_at,how='left')
            del temp
            bd.rename(columns={col:'unique_'+col+'_count'},inplace=True)

        print('done building num unique features')
        return bd

    def _cat_counts(self):

        bd=self._base_data

        for col in self.cat_counts:

            temp=pd.crosstab(self.data[self.summarize_at],self.data[col])
            temp.columns=[str(_)+'_cat_count_'+col for _ in list(temp.columns)]
            temp.reset_index(inplace=True)
            bd=pd.merge(bd,temp,on=self.summarize_at,how='left')
            del temp
        print('done building cat counts features')
        return bd

    def _top_cat(self):

        bd=self._base_data

        for col in self.top_cat:

            temp=pd.DataFrame(self.data.groupby([self.summarize_at])[col].value_counts())
            temp.columns=['max_freq_'+col+'_cat']
            temp.reset_index(inplace=True)
            temp=temp.groupby([self.summarize_at])[col].nth(0).reset_index()
            temp.columns=[self.summarize_at,'max_freq_'+col+'_cat']
            bd=pd.merge(bd,temp,on=self.summarize_at,how='left')
        
            del temp

        print('done building top cat features')

        return bd

    def _age(self):

        bd=self._base_data

        temp=self.data[[self.summarize_at,self.age]].copy()
        temp[self.age]=pd.to_datetime(temp[self.age])
        temp.sort_values([self.summarize_at,self.age],inplace=True)

        ref_date=self.ref_date_age
        if self.ref_date_age is None:
            ref_date=pd.to_datetime('today')
        
        temp=pd.DataFrame((ref_date-temp.groupby([self.summarize_at])[self.age].nth(0)).dt.days)
        temp.reset_index(inplace=True)

        temp.columns=[self.summarize_at,self.summarize_at+'_age']
        bd=pd.merge(bd,temp,on=self.summarize_at,how='left')
        del temp

        print('done building age feature')

        return bd

    def _last_interaction(self):

        bd=self._base_data

        temp=self.data[[self.summarize_at,self.last_interaction]].copy()
        temp[self.last_interaction]=pd.to_datetime(temp[self.last_interaction])
        temp.sort_values([self.summarize_at,self.last_interaction],inplace=True)

        ref_date=self.ref_date_last_interaction
        if self.ref_date_last_interaction is None:
            ref_date=pd.to_datetime('today')
        
        temp=pd.DataFrame((ref_date-temp.groupby([self.summarize_at])[self.last_interaction].nth(-1)).dt.days)
        temp.reset_index(inplace=True)

        temp.columns=[self.summarize_at,self.summarize_at+'_days_since_last_interaction']
        bd=pd.merge(bd,temp,on=self.summarize_at,how='left')
        del temp

        print('done building days since last interaction feature')

        return bd

    def _intervals_summary(self):

        bd=self._base_data

        for col in self.intervals_summary:

            temp=self.data[[self.summarize_at,col]].copy()
            temp[col]=pd.to_datetime(temp[col])
            temp.sort_values([self.summarize_at,col],inplace=True)
            temp.drop_duplicates(inplace=True)
            temp['lag_date']=temp.groupby([self.summarize_at])[col].shift()
            temp['diff']=(temp[col]-temp['lag_date']).dt.days
            temp.dropna(inplace=True)
            temp=temp.groupby(self.summarize_at)['diff'].describe()
            del temp['count']
            temp.columns=['intervals_'+x+'_'+col for x in temp.columns]
            bd=pd.merge(bd,temp,on=self.summarize_at,how='left')
            del temp
        
        print('done building intervals summary features')
        return bd

    def _num_summary(self):

        bd=self._base_data

        for col in self.num_summary:
            temp=self.data.groupby([self.summarize_at])[col].describe()
            del temp['count']
            temp.columns=[col+'_'+x for x in temp.columns]
            temp.reset_index(inplace=True)
            bd=pd.merge(bd,temp,on=self.summarize_at,how='left')
            del temp

        print('done building numeric summary features')

        return bd


    def build(self):

        bd=self._base_data

        for func in [self._num_unique,self._cat_counts,self._top_cat,
                     self._age,self._last_interaction,self._intervals_summary,self._num_summary]:

            bd=pd.merge(bd.drop([self.summarize_at+'_count'],1),func(),on=self.summarize_at,how='left')

        return bd

class pdPipeline(Pipeline):

    def get_feature_names(self):

        last_step = self.steps[-1][-1]

        return last_step.get_feature_names()

class DataPipe:

    def __init__(self,cat_to_dummies=None,
                        cat_to_numeric=None,
                        simple_numeric=None,
                        custom_var_dict=None,
                        date_diffs=None,
                        text_feat=None,
                        cyclic_feat=None,
                        for_catboost=False
                        ):

        self.cat_to_dummies=cat_to_dummies
        self.cat_to_numeric=cat_to_numeric
        self.custom_var_dict=custom_var_dict
        self.simple_numeric=simple_numeric
        self.date_diffs=date_diffs
        self.text_feat=text_feat
        self.cyclic_feat=cyclic_feat
        self.this_pipe=None
        self.for_catboost=for_catboost

    def fit(self,X):

        pipelines={}
        i=1

        if (self.cat_to_dummies is not None):

            if self.for_catboost:

                pipelines['p'+str(i)]=pdPipeline([
                                        ('var_select',VarSelector(self.cat_to_dummies)),
                                        ('missing_trt',DataFrameImputer())
                                        ])

            else: 

                freq_cutoff=int(X.shape[0]*0.01)

                pipelines['p'+str(i)]=pdPipeline([
                                        ('var_select',VarSelector(self.cat_to_dummies)),
                                        ('missing_trt',DataFrameImputer()),
                                        ('create_dummies',CreateDummies(freq_cutoff))
                                        ])
            i+=1
        if self.simple_numeric is not None:

            pipelines['p'+str(i)]=pdPipeline([
                                    ('var_select',VarSelector(self.simple_numeric)),
                                    ('missing_trt',DataFrameImputer())
                                    ])
            i+=1

        if self.cat_to_numeric is not None:

            pipelines['p'+str(i)]=pdPipeline([
                                    ('var_select',VarSelector(self.cat_to_numeric)),
                                    ('convert_to_numeric',ConvertNumeric()),
                                    ('missing_trt',DataFrameImputer())
                                    ])
            i+=1


        if self.custom_var_dict is not None:

            for var in self.custom_var_dict.keys():

                func=self.custom_var_dict[var]
                pipelines['p'+str(i)]=pdPipeline([
                                        ('custom_'+var,CustomVar(func,var)),
                                        ('missing_trt',DataFrameImputer())
                                        ])

                i+=1

        if self.date_diffs is not None:

            pipelines['p'+str(i)]=pdPipeline([
                                    ('var_select',VarSelector(self.date_diffs)),
                                    ('date_diffs',DateDiffs()),
                                    ('missing_trt',DataFrameImputer())
                                    ])

            i+=1

        if self.cyclic_feat is not None:

            pipelines['p'+str(i)]=pdPipeline([
                                    ('var_select',VarSelector(self.cyclic_feat)),
                                    ('cyclic_feat',CyclicFeatures())
                                    ])

            i+=1

        if self.text_feat is not None:

            pipelines['p'+str(i)]=pdPipeline([
                                    ('var_select',VarSelector(self.text_feat)),
                                    ('missing_trt',DataFrameImputer()),
                                    ('text_feat',TextFeatures())
                                    ])

            i+=1

        feature_union_pipes=list(pipelines.items())

        self.this_pipe=FeatureUnion(feature_union_pipes)
        self._all_feature_names=[]

        

        self.this_pipe.fit(X)

        for _ in list(pipelines.items()):
            self._all_feature_names.extend(_[1].get_feature_names())

        return self

    def transform(self,X):


        mydata=pd.DataFrame(
                        data=self.this_pipe.transform(X),
                        columns=self._all_feature_names
                     )

        return mydata