from __future__ import annotations
import hashlib, json, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from FINAL_V2.preprocess.contracts import ID,TARGET,STAGE48_MODEL_FEATURES
from FINAL_V2.preprocess.stage48 import build_stage48_model_inputs
from FINAL_V2.preprocess.stage51_53 import make_stage53_interactions
from FINAL_V2.preprocess.stage60 import make_stage60_features
from FINAL_V2.model.core import add_final_features, folds_for, oof_test, best_threshold, topk_labels

SEEDS=[3,7,11,17,21,31,42,53,67,73,89,101,127,149,173]
EXPECTED_RAW={'train.csv':'48792d669790f8b25050d60d6a90a924a9e7fd36081d42eb3ede610708c79425','test.csv':'764633807020d59d53986ad5b4f64297892facc639727e9ecf3a361634b10dcd'}
EXPECTED_STAGE48_MODEL_INPUT={'train.csv':'d3ae8f3b180af1c6ad67c4a68487bde9916494e394f120e94ccb796993567a6b','test.csv':'7c8b24a204be0a47ab7d0251744c7205ba252fd8aad0fd212e6c351988acb298'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def write_submit(path,ids,labels):
    pd.DataFrame({ID:ids,TARGET:np.asarray(labels,int)}).to_csv(path,index=False,header=False,encoding='utf-8')

def ensure_data():
    d=ROOT/'data'; d.mkdir(exist_ok=True)
    repo=ROOT.parent/'data'
    for name in ('train.csv','test.csv'):
        dst=d/name
        if not dst.exists(): shutil.copy2(repo/name,dst)
        got=sha(dst); assert got==EXPECTED_RAW[name],(name,got)

def build_feature_sets(tr_raw,te_raw,tr48,te48):
    base_cols=STAGE48_MODEL_FEATURES
    xb=add_final_features(tr48[base_cols]); xtb=add_final_features(te48[base_cols])
    ftr,fte,itr,ite,thresholds=make_stage53_interactions(tr_raw,te_raw)
    core_cols=['mismatch10_x_large_employee','mismatch10_x_high_cf_margin','software_x_large_employee','software_x_high_q9']
    xc=xb.copy(); xtc=xtb.copy()
    for c in core_cols:
        xc['s53_'+c]=itr[c].to_numpy(float); xtc['s53_'+c]=ite[c].to_numpy(float)
    s60tr,s60te=make_stage60_features(tr_raw),make_stage60_features(te_raw)
    xn=xc.copy(); xtn=xtc.copy()
    for c in ('education_need','education_need_x_capacity'):
        xn['s60_'+c]=s60tr[c].to_numpy(float); xtn['s60_'+c]=s60te[c].to_numpy(float)
    return xb,xtb,xc,xtc,xn,xtn,ftr,fte,itr,ite,s60tr,s60te,thresholds

def main():
    ensure_data(); out=ROOT/'outputs'; mi=ROOT/'model_input'; shutil.rmtree(out,ignore_errors=True); shutil.rmtree(mi,ignore_errors=True); out.mkdir(); mi.mkdir()
    tr0=pd.read_csv(ROOT/'data/train.csv'); te0=pd.read_csv(ROOT/'data/test.csv'); manual=json.loads((ROOT/'assets/train_annotations.json').read_text(encoding='utf-8'))
    tr48,te48,atr,ate=build_stage48_model_inputs(tr0,te0,manual)
    trp=mi/'stage48_train.csv'; tep=mi/'stage48_test.csv'; tr48.to_csv(trp,index=False,encoding='utf-8-sig'); te48.to_csv(tep,index=False,encoding='utf-8-sig')
    input_hash={'train.csv':sha(trp),'test.csv':sha(tep)}; assert input_hash==EXPECTED_STAGE48_MODEL_INPUT,input_hash
    # Historical path reload is intentional: Stage48/54/60 descendants consumed persisted floats.
    tr48=pd.read_csv(trp); te48=pd.read_csv(tep); y=tr48[TARGET].to_numpy(int)
    xb,xtb,xc,xtc,xn,xtn,ftr,fte,itr,ite,s60tr,s60te,cut=build_feature_sets(tr0,te0,tr48,te48)
    assert xb.shape==(742,113) and xtb.shape==(800,113)
    assert xc.shape[1]==117 and xn.shape[1]==119
    rows=[]; base_oof=[]; base_test=[]; core_oof=[]; core_test=[]; need_oof=[]; need_test=[]; seed_base_threshold=[]
    for seed in SEEDS:
        fold=folds_for(seed,y)
        pb,pbt=oof_test(xb,y,fold,xtb); pc,pct=oof_test(xc,y,fold,xtc); pn,pnt=oof_test(xn,y,fold,xtn)
        thb,f1b=best_threshold(y,pb); seed_base_threshold.append(thb)
        p54=pb+.50*(pc-pb); t54=pbt+.50*(pct-pbt)
        p60=p54+.25*(pn-pc); t60=t54+.25*(pnt-pct)
        th54,f154=best_threshold(y,p54); th60,f160=best_threshold(y,p60)
        rows += [{'seed':seed,'variant':'stage48','threshold':thb,'f1':f1b},{'seed':seed,'variant':'stage54','threshold':th54,'f1':f154},{'seed':seed,'variant':'stage60','threshold':th60,'f1':f160}]
        base_oof.append(pb); base_test.append(pbt); core_oof.append(p54); core_test.append(t54); need_oof.append(p60); need_test.append(t60)
    p48=np.mean(base_oof,axis=0); t48=np.mean(base_test,axis=0); p54=np.mean(core_oof,axis=0); t54=np.mean(core_test,axis=0); p60=np.mean(need_oof,axis=0); t60=np.mean(need_test,axis=0)
    th48,f148=best_threshold(y,p48); th54,f154=best_threshold(y,p54); th60,f160=best_threshold(y,p60)
    # Exact old 205 hedge: Stage48 median seed-threshold -> positive rate -> top-k test.
    med48=float(np.median(seed_base_threshold)); rate48=float((p48>=med48).mean()); k205=int(round(len(t48)*rate48)); old205=topk_labels(t48,k205)
    labels190=(t60>=th60).astype(int); labels192=(t54>=th54).astype(int); labels193=topk_labels(t60,193); labels185=topk_labels(t60,185)
    counts={'190_stage60':int(labels190.sum()),'205_stage48_hedge':int(old205.sum()),'192_stage54':int(labels192.sum()),'193_stage60_topk':int(labels193.sum()),'185_stage60_topk':int(labels185.sum())}
    assert counts['190_stage60']==190,counts
    assert counts['205_stage48_hedge']==205,counts
    assert counts['192_stage54']==192,counts
    assert counts['193_stage60_topk']==193 and counts['185_stage60_topk']==185
    files={
      '190_stage60':'SUBMIT_190_STAGE60.csv',
      '205_stage48_hedge':'SUBMIT_205_STAGE48_HEDGE.csv',
      '192_stage54':'SUBMIT_192_STAGE54.csv',
      '193_stage60_topk':'SUBMIT_193_STAGE60_TOPK.csv',
      '185_stage60_topk':'SUBMIT_185_STAGE60_TOPK.csv'}
    labels={'190_stage60':labels190,'205_stage48_hedge':old205,'192_stage54':labels192,'193_stage60_topk':labels193,'185_stage60_topk':labels185}
    for k,n in files.items():write_submit(out/n,te0[ID],labels[k])
    hashes={k:sha(out/n) for k,n in files.items()}
    # audits
    pd.DataFrame(rows).to_csv(out/'seed_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame({ID:te0[ID],'stage48_prob':t48,'stage54_prob':t54,'stage60_prob':t60,**{k:v for k,v in labels.items()}}).to_csv(out/'test_probability_and_labels.csv',index=False,encoding='utf-8-sig')
    ftr.to_csv(mi/'stage51_train_features.csv',index=False,encoding='utf-8-sig'); fte.to_csv(mi/'stage51_test_features.csv',index=False,encoding='utf-8-sig'); itr.to_csv(mi/'stage53_train_interactions.csv',index=False,encoding='utf-8-sig'); ite.to_csv(mi/'stage53_test_interactions.csv',index=False,encoding='utf-8-sig'); s60tr.to_csv(mi/'stage60_train_features.csv',index=False,encoding='utf-8-sig'); s60te.to_csv(mi/'stage60_test_features.csv',index=False,encoding='utf-8-sig'); atr.to_csv(mi/'train_sentence_annotations.csv',index=False,encoding='utf-8-sig'); ate.to_csv(mi/'test_sentence_annotations.csv',index=False,encoding='utf-8-sig')
    manifest={'status':'VERIFIED','purpose':'five final public-LB candidates with fully frozen local reproduction','seeds':SEEDS,'models_per_variant':75,'feature_shapes':{'stage48':list(xb.shape),'stage54_core':list(xc.shape),'stage60_need_capacity':list(xn.shape)},'train_thresholds':{'stage48':th48,'stage54':th54,'stage60':th60},'oof_f1':{'stage48':f148,'stage54':f154,'stage60':f160},'stage48_median_seed_threshold':med48,'stage48_median_rate':rate48,'stage53_train_cuts':cut,'counts':counts,'files':files,'sha256':hashes,'raw_sha256':EXPECTED_RAW,'stage48_model_input_sha256':input_hash,'contracts':{'190':'Stage60 DEV-average OOF best threshold transported to test','192':'Stage54 DEV-average OOF best threshold transported to test','193':'Stage60 test probability top-k=193 diagnostic hedge','185':'Stage60 test probability top-k=185 precision hedge','205':'historical Stage48 median-seed-threshold positive-rate top-k, exactly the old 205 hedge'}}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
