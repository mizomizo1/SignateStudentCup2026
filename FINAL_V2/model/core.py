from __future__ import annotations
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold

C=1.5
POS_WEIGHT=2.5
N_SPLITS=5

def add_final_features(df):
    x=df.copy()
    def num(c): return pd.to_numeric(x[c],errors='coerce').fillna(0)
    emp=num('従業員数'); le=np.log1p(emp.clip(lower=0))
    x['log_employees']=le; x['log_sales_abs']=np.log1p(num('売上').abs()); x['log_assets_abs']=np.log1p(num('総資産').abs())
    prof=x[['営業利益率','経常利益率','純利益率','ROA','ROE']].apply(pd.to_numeric,errors='coerce').mean(axis=1)
    cf=x[['営業CFマージン','フリーCFマージン','利益CF比率']].apply(pd.to_numeric,errors='coerce').mean(axis=1)
    sv5=num('アンケート５'); sites=np.log1p(num('事業所数').clip(lower=0))
    x['profitability_mean']=prof; x['cashflow_mean']=cf; x['enterprise_x_survey5']=le*sv5; x['enterprise_x_profitability']=le*prof; x['enterprise_x_cashflow']=le*cf; x['enterprise_x_sites']=le*sites
    bucket=pd.cut(emp,[-np.inf,299,999,4999,np.inf],labels=['S','M','L','XL']).astype(str)
    x['size_bucket']=bucket; x['industry_size_bucket']=x['業界'].fillna('__NA__').astype(str)+'__'+bucket; x['industry_dxdept']=x['業界'].fillna('__NA__').astype(str)+'__DX'+x['DX部署あり'].fillna(0).astype(str); x['listing_size_bucket']=x['上場種別'].fillna('__NA__').astype(str)+'__'+bucket
    sales=num('売上').abs(); assets=num('総資産').abs(); equity=num('自己資本').abs(); ocf=num('営業CF'); factories=num('工場数').clip(lower=0)
    ls=np.log1p(sales); la=np.log1p(assets); lq=np.log1p(equity); lf=np.log1p(factories); locf=np.sign(ocf)*np.log1p(np.abs(ocf))
    band=lambda s,e,l:pd.cut(s,e,labels=l,include_lowest=True).astype(str)
    size=band(emp,[-np.inf,299,999,4999,np.inf],['S','M','L','XL']); sb=band(ls,[-np.inf,10,12,14,np.inf],['s0','s1','s2','s3']); ab=band(la,[-np.inf,10,12,14,np.inf],['a0','a1','a2','a3']); qb=band(lq,[-np.inf,9,11,13,np.inf],['q0','q1','q2','q3']); fb=band(lf,[-np.inf,.1,1.1,2.4,np.inf],['f0','f1','f2','f3']); cb=band(locf,[-np.inf,-10,0,10,12,np.inf],['c0','c1','c2','c3','c4'])
    ind=x['業界'].fillna('__NA__').astype(str); listing=x['上場種別'].fillna('__NA__').astype(str); dx=(num('DX部署あり')>0).astype(int).astype(str)
    x['industry_size2']=ind+'__'+size; x['industry_sales_band']=ind+'__'+sb; x['industry_asset_band']=ind+'__'+ab; x['industry_equity_band']=ind+'__'+qb; x['industry_ocf_band']=ind+'__'+cb; x['industry_factory_band']=ind+'__'+fb
    x['industry_dx_size2']=ind+'__DX'+dx+'__'+size; x['industry_dx_sales']=ind+'__DX'+dx+'__'+sb; x['industry_dx_assets']=ind+'__DX'+dx+'__'+ab
    x['listing_size2']=listing+'__'+size; x['listing_sales_band2']=listing+'__'+sb; x['listing_asset_band2']=listing+'__'+ab; x['listing_equity_band']=listing+'__'+qb
    x['industry_scale_capacity_num']=le*(ls+la+lq)/3.; x['dx_scale_capacity_num']=dx.astype(int)*le*(ls+la)/2.
    def slog(s):return np.sign(s)*np.log1p(np.abs(s))
    fcf=num('簡易フリーCF'); capex=num('設備投資額'); sw=num('ソフトウェア投資額'); opm=num('営業利益率'); npm=num('純利益率'); roa=num('ROA'); roe=num('ROE')
    capacity=(slog(emp)+slog(sales)+slog(assets)+slog(equity))/4.; quality=(opm+npm+roa+roe)/4.; cash=(slog(ocf)+slog(fcf))/2.; investment=(slog(capex.abs())+slog(sw.abs()))/2.
    atoms={'capacity':capacity,'quality':quality,'cash':cash,'investment':investment,'cap_x_quality':capacity*quality,'cap_x_cash':capacity*cash,'cap_x_investment':capacity*investment,'cap_x_q7_low':capacity*(num('アンケート７')<=2).astype(float)}
    for k,v in atoms.items():x['s30_'+k]=v
    return x

def make_pipe(X):
    nums=[c for c in X.columns if is_numeric_dtype(X[c])]; cats=[c for c in X.columns if c not in nums]; trans=[]
    if nums: trans.append(('num',Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]),nums))
    if cats: trans.append(('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore',drop='first'))]),cats))
    return Pipeline([('pre',ColumnTransformer(trans)),('m',LogisticRegression(penalty='l1',C=C,class_weight={0:1.,1:POS_WEIGHT},solver='liblinear',max_iter=5000,random_state=42))])

def folds_for(seed,y):
    sk=StratifiedKFold(N_SPLITS,shuffle=True,random_state=seed); f=np.empty(len(y),int)
    for i,(_,va) in enumerate(sk.split(np.zeros(len(y)),y)):f[va]=i
    return f

def oof_test(X,y,fold,Xt=None):
    o=np.zeros(len(y)); t=np.zeros(len(Xt)) if Xt is not None else None; fs=sorted(np.unique(fold))
    for k in fs:
        tr=fold!=k; va=fold==k; m=make_pipe(X); m.fit(X.loc[tr],y[tr]); o[va]=m.predict_proba(X.loc[va])[:,1]
        if Xt is not None:t+=m.predict_proba(Xt)[:,1]/len(fs)
    return o,t

def best_threshold(y,p):
    y=np.asarray(y,int); p=np.asarray(p,float); o=np.argsort(p)[::-1]; ys=y[o]; tp=np.cumsum(ys); fp=np.cumsum(1-ys); fn=y.sum()-tp; d=2*tp+fp+fn; s=np.divide(2*tp,d,out=np.zeros_like(tp,dtype=float),where=d!=0); i=int(np.argmax(s)); return float(p[o[i]]),float(s[i])

def topk_labels(p,k):
    p=np.asarray(p,float); z=np.zeros(len(p),int); z[np.argsort(p)[-int(k):]]=1; return z
