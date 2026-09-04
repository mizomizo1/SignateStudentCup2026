from __future__ import annotations
import math, re, unicodedata
import numpy as np
import pandas as pd

EMPLOYEE_PATTERNS=(
 re.compile(r"(?:従業員(?:数)?|社員(?:数)?|職員(?:数)?|スタッフ(?:数)?|人員(?:数)?)[^0-9]{0,16}([0-9][0-9,]*(?:\.[0-9]+)?)\s*(万)?\s*(?:人|名)"),
 re.compile(r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(万)?\s*(?:人|名)[^。\n]{0,16}(?:従業員(?:数)?|社員(?:数)?|職員(?:数)?|スタッフ(?:数)?|人員(?:数)?)"),
)
COUNT_PATTERNS={
 'subsidiary':(re.compile(r"(?:連結)?子会社[^0-9]{0,8}([0-9][0-9,]*)\s*社"),re.compile(r"([0-9][0-9,]*)\s*社[^。\n]{0,8}(?:の)?(?:連結)?子会社")),
 'affiliate':(re.compile(r"関連会社[^0-9]{0,8}([0-9][0-9,]*)\s*社"),re.compile(r"([0-9][0-9,]*)\s*社[^。\n]{0,8}(?:の)?関連会社")),
}

def normalize(v):
    if pd.isna(v): return ''
    return unicodedata.normalize('NFKC',str(v)).replace('，',',')

def positive_number(v):
    try: x=float(normalize(v).replace(',',''))
    except ValueError: return float('nan')
    return x if x>0 else float('nan')

def extract_employee(text,structured):
    s=normalize(text); vals=[]
    for pat in EMPLOYEE_PATTERNS:
        for m in pat.finditer(s):
            v=float(m.group(1).replace(',',''))*(10000 if m.group(2) else 1)
            if 1<=v<=2_000_000: vals.append(v)
    if not vals:return float('nan')
    tab=positive_number(structured)
    return min(vals,key=lambda v:abs(math.log(v/tab))) if np.isfinite(tab) else vals[0]

def extract_company_count(text,patterns):
    vals=[]
    for pat in patterns: vals.extend(float(m.group(1).replace(',','')) for m in pat.finditer(text))
    return max(vals) if vals else float('nan')

def duplicate_lines(text):
    lines=[re.sub(r'^[\s├└│─]+','',line).strip() for line in text.splitlines()]
    lines=[line for line in lines if len(line)>=3]
    return len(lines)-len(set(lines))

def requested_features(raw):
    overview=raw['企業概要'].map(normalize); org=raw['組織図'].map(normalize); joined=overview+'\n'+org
    tab=pd.to_numeric(raw['従業員数'],errors='coerce').astype(float)
    text_emp=pd.Series([extract_employee(t,v) for t,v in zip(overview,tab)],index=raw.index,dtype=float)
    ratio=text_emp/tab.replace(0,np.nan); sym=np.maximum(ratio,1/ratio)
    out=pd.DataFrame(index=raw.index)
    out['employee_text_value']=text_emp; out['employee_text_found']=text_emp.notna().astype(float)
    out['employee_mismatch_20pct']=((ratio>1.2)|(ratio<1/1.2)).fillna(False).astype(float)
    out['employee_mismatch_3x']=sym.ge(3).fillna(False).astype(float)
    out['employee_near_10x']=sym.between(8,12).fillna(False).astype(float)
    out['overview_overseas']=overview.str.contains(r'海外|国外|国際|グローバル').astype(float); out['org_overseas']=org.str.contains(r'海外|国外|国際|グローバル').astype(float)
    out['both_overseas']=((out.overview_overseas==1)&(out.org_overseas==1)).astype(float); out['overview_global_literal']=overview.str.contains(r'グローバル').astype(float); out['org_global_literal']=org.str.contains(r'グローバル').astype(float)
    out['overview_subsidiary']=overview.str.contains(r'子会社|連結子会社').astype(float); out['org_subsidiary']=org.str.contains(r'子会社|連結子会社').astype(float); out['overview_affiliate']=overview.str.contains(r'関連会社').astype(float); out['org_affiliate']=org.str.contains(r'関連会社').astype(float); out['has_group_structure']=joined.str.contains(r'子会社|関連会社|グループ会社|持株会社|グループ体制').astype(float)
    for source,text in (('overview',overview),('org',org),('all',joined)):
        for word,label in (('子会社','subsidiary'),('関連会社','affiliate')):
            count=pd.Series([extract_company_count(v,COUNT_PATTERNS[label]) for v in text],index=raw.index,dtype=float)
            out[f'{source}_{label}_count_found']=count.notna().astype(float); out[f'{source}_{label}_count_log']=np.log1p(count.fillna(0).clip(lower=0,upper=500))
    software_pattern=r"ソフトウェア.{0,12}(?:開発|設計|提供|販売)|アプリ(?:ケーション)?.{0,12}(?:開発|設計|提供)|システム.{0,12}(?:開発|構築|受託)|IT.{0,10}(?:サービス|ソリューション|開発)"
    out['overview_software_dev']=overview.str.contains(software_pattern).astype(float)
    out['org_software_dev']=org.str.contains(r'ソフトウェア|アプリ(?:ケーション)?|システム開発|ITサービス|ITソリューション').astype(float)
    out['any_software_dev']=((out.overview_software_dev==1)|(out.org_software_dev==1)).astype(float); out['both_software_dev']=((out.overview_software_dev==1)&(out.org_software_dev==1)).astype(float)
    out['org_word_count']=org.str.count(r'組織図').astype(float); out['org_word_twice']=out.org_word_count.ge(2).astype(float)
    out['org_root_repeat']=((org.str.count(r'経営層')>=2)|(org.str.count(r'代表取締役')>=3)).astype(float)
    out['org_duplicate_lines']=org.map(duplicate_lines).astype(float); out['org_duplicate_lines_any']=out.org_duplicate_lines.gt(0).astype(float)
    return out.replace([np.inf,-np.inf],np.nan)

def interaction_features(raw,feature,emp_median,cf_median):
    emp=np.log1p(pd.to_numeric(raw['従業員数'],errors='coerce').clip(lower=0)); cf=pd.to_numeric(raw['営業CF'],errors='coerce')/pd.to_numeric(raw['売上'],errors='coerce').replace(0,np.nan)
    large=emp.ge(emp_median).fillna(False).astype(float); high_cf=cf.ge(cf_median).fillna(False).astype(float); high_q9=pd.to_numeric(raw['アンケート９'],errors='coerce').ge(4).fillna(False).astype(float)
    mismatch10=feature['employee_near_10x'].astype(float); software=feature['overview_software_dev'].astype(float); global_=feature['overview_global_literal'].astype(float); group=feature['has_group_structure'].astype(float)
    out=pd.DataFrame(index=raw.index)
    out['mismatch10_x_large_employee']=mismatch10*large; out['mismatch10_x_high_cf_margin']=mismatch10*high_cf; out['software_x_large_employee']=software*large; out['software_x_high_q9']=software*high_q9; out['global_x_large_employee']=global_*large; out['group_x_large_employee']=group*large
    return out

def make_stage53_interactions(train_raw,test_raw):
    ftr,fte=requested_features(train_raw),requested_features(test_raw)
    emp_med=float(np.log1p(pd.to_numeric(train_raw['従業員数'],errors='coerce').clip(lower=0)).median())
    cf=pd.to_numeric(train_raw['営業CF'],errors='coerce')/pd.to_numeric(train_raw['売上'],errors='coerce').replace(0,np.nan); cf_med=float(cf.median())
    itr=interaction_features(train_raw,ftr,emp_med,cf_med); ite=interaction_features(test_raw,fte,emp_med,cf_med)
    return ftr,fte,itr,ite,{'log_employee_median':emp_med,'cf_margin_median':cf_med}
