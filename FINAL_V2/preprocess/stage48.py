from __future__ import annotations
import re, math
from collections import Counter
import numpy as np
import pandas as pd
from .contracts import ID, TARGET, STAGE48_MODEL_FEATURES

CONTRAST_PATTERNS=('しかしながら','しかし','ただし','一方で','その一方で','一方','もっとも','とはいえ','ところが','けれども','ですが','だが')
FUTURE_CUES={'今後':2.8,'将来':2.5,'これから':2.2,'中長期':1.7,'次期':1.7,'予定':2.2,'計画':1.8,'方針':1.4,'目標':1.5,'構想':1.4,'目指':2.2,'見込':1.8,'実現に向け':1.8,'引き続き':1.4,'していき':2.5,'してまいり':2.5,'進めていき':2.5,'取り組んでいき':2.5,'推進していき':2.5,'強化していき':2.5,'拡大していき':2.5,'展開していき':2.5,'図っていき':2.3,'進めます':2.0,'取り組みます':2.0,'推進します':2.0,'強化します':2.0,'拡大します':2.0,'展開します':2.0,'図ります':2.0,'目指します':2.2,'予定です':2.3}
PAST_CUES={'これまで':2.3,'現在':1.9,'現時点':1.8,'すでに':2.2,'既に':2.2,'してきた':2.3,'してきました':2.5,'実施した':2.3,'実施しました':2.4,'導入した':2.3,'導入しました':2.4,'開始した':2.2,'開始しました':2.3,'構築した':2.2,'構築しました':2.3,'開発した':2.1,'開発しました':2.2,'実現した':2.3,'実現しました':2.4,'達成した':2.3,'達成しました':2.4,'発表した':2.1,'締結した':2.1,'活用している':1.9,'運用している':1.9,'導入している':1.9,'利用している':1.8,'実施している':1.8,'行っている':1.6,'取り組んでいる':1.5,'進めている':1.5,'推進している':1.5,'構築している':1.5}
LEX={
'future_aggressive':['全社','全社員','加速','拡大','拡充','深化','高度化','意思決定','次世代','イノベーション','本格','迅速','推進'],
'future_cautious':['慎重','限定','最小限','最低限','段階的','小規模','範囲','とどめ','リスク','見極め','抑制','既存'],
'past_achievement':['成果','実装','着実','部門横断','業務プロセス','連携','導入','推進','改善','確か'],
'past_opportunity':['改善余地','余地','伸びしろ','さらなる','スピード感'],
'past_unfinished':['至ってい','限定的','とどま','十分では','未導入','実情','小規模','従来']}

def normalize_text(v): return '' if v is None else str(v).replace('\r\n','\n').replace('\r','\n').strip()
def split_paragraphs(t): return [p.strip() for p in re.split(r'\n\s*\n',normalize_text(t)) if p.strip()]
def split_sentences(t): return [s.strip() for s in re.split(r'(?<=[。！？!?])',normalize_text(t)) if s.strip()]
def company_sentences(text):
    out=[]; idx=0
    for pi,p in enumerate(split_paragraphs(text)):
        for s in split_sentences(p): out.append({'index':idx,'paragraph_index':pi,'text':s}); idx+=1
    return out

def char_ngrams(text):
    compact=re.sub(r'\s+','',text); grams=Counter()
    for n in (2,3,4):
        for i in range(max(len(compact)-n+1,0)): grams[compact[i:i+n]]+=1
    return grams

class Classifier:
    def __init__(self):
        self.doc_counts=Counter(); self.feature_counts={l:Counter() for l in ('past','future')}; self.feature_totals=Counter(); self.vocab=set()
    def add(self,text,label):
        if label not in ('past','future'): return
        g=char_ngrams(text); self.doc_counts[label]+=1; self.feature_counts[label].update(g); self.feature_totals[label]+=sum(g.values()); self.vocab.update(g)
    def margin(self,text):
        g=char_ngrams(text)
        if not g or not self.vocab:return 0.0
        alpha=.4; v=max(len(self.vocab),1); scores={}
        for l in ('past','future'):
            denom=self.feature_totals[l]+alpha*v; total=0.; weight=0; counts=self.feature_counts[l]
            for gram,freq in g.items():
                if gram not in self.vocab: continue
                total += freq*math.log((counts.get(gram,0)+alpha)/denom); weight+=freq
            scores[l]=total/max(weight,1)
        return (scores['future']-scores['past'])*5.
    def predict(self,sentence,pi):
        score=self.margin(sentence)+(-.65 if pi==0 else .55)
        for cue,w in FUTURE_CUES.items():
            if cue in sentence: score+=w
        for cue,w in PAST_CUES.items():
            if cue in sentence: score-=w
        if re.search(r'(?:していく|してまいります|していきます|を目指す|を目指します)[。！？!?]?$',sentence): score+=1.8
        if re.search(r'(?:した|しました|してきた|してきました)[。！？!?]?$',sentence): score-=1.8
        return 'future' if score>=0 else 'past'

def resolve_manual(a,si,pi):
    labels=a.get('labels',{}) if isinstance(a,dict) else {}; pars=a.get('paragraph_labels',{}) if isinstance(a,dict) else {}
    if str(si) in labels and labels[str(si)] in ('past','future'):return labels[str(si)]
    if str(pi) in pars and pars[str(pi)] in ('past','future'):return pars[str(pi)]
    return 'past' if pi==0 else 'future'

def manual_contrast(a,si,s):
    flags=a.get('contrast_flags',{}) if isinstance(a,dict) else {}
    return int(bool(flags[str(si)])) if str(si) in flags else int(any(p in s for p in CONTRAST_PATTERNS))

def train_classifier(train,manual):
    cl=Classifier()
    for _,r in train.iterrows():
        a=manual.get(str(r[ID]).strip())
        if not isinstance(a,dict):continue
        for s in company_sentences(r['今後のDX展望']): cl.add(s['text'],resolve_manual(a,s['index'],s['paragraph_index']))
    return cl

def make_annotations(df,cl,manual=None):
    rows=[]
    for _,r in df.iterrows():
        a=manual.get(str(r[ID]).strip()) if manual else None
        for s in company_sentences(r['今後のDX展望']):
            label=resolve_manual(a,s['index'],s['paragraph_index']) if isinstance(a,dict) else cl.predict(s['text'],s['paragraph_index'])
            contrast=manual_contrast(a,s['index'],s['text']) if isinstance(a,dict) else int(any(p in s['text'] for p in CONTRAST_PATTERNS))
            rows.append({ID:r[ID],'sentence':s['text'],'label':label,'contrast_flag':contrast})
    return pd.DataFrame(rows)

def safe_div(a,b): return a/b.replace(0,np.nan)

def base_features(df):
    o=df.copy(); org=o['組織図'].fillna('').astype(str); o['DX部署あり']=org.str.contains(r'DX|ＤＸ|デジタル推進|デジタル戦略|デジタル変革',case=False,regex=True).astype(int)
    specs=[('自己資本比率','自己資本','総資産'),('負債比率','負債','総資産'),('純資産比率','純資産','総資産'),('流動資産比率','流動資産','総資産'),('固定資産比率','固定資産','総資産')]
    for n,a,b in specs:o[n]=safe_div(o[a],o[b])
    o['総借入金']=o['短期借入金']+o['長期借入金']; o['借入金自己資本比率']=safe_div(o['総借入金'],o['自己資本']); o['短期借入金比率']=safe_div(o['短期借入金'],o['総資産']); o['長期借入金比率']=safe_div(o['長期借入金'],o['総資産'])
    for n,a,b in [('営業利益率','営業利益','売上'),('経常利益率','経常利益','売上'),('純利益率','当期純利益','売上'),('ROA','当期純利益','総資産'),('ROE','当期純利益','自己資本'),('総資産回転率','売上','総資産')]:o[n]=safe_div(o[a],o[b])
    o['簡易フリーCF']=o['営業CF']+o['投資CF']
    for n,a in [('営業CFマージン','営業CF'),('投資CFマージン','投資CF'),('フリーCFマージン','簡易フリーCF')]:o[n]=safe_div(o[a],o['売上'])
    o['利益CF比率']=safe_div(o['営業CF'],o['当期純利益']); o['投資額']=-o['投資CF']; o['設備投資額']=-o['有形固定資産変動']; o['ソフトウェア投資額']=-o['無形固定資産変動(ソフトウェア関連)']
    for n,a in [('設備投資売上比率','設備投資額'),('減価償却売上比率','減価償却費'),('運転資本変動売上比率','運転資本変動'),('ソフトウェア投資売上比率','ソフトウェア投資額')]:o[n]=safe_div(o[a],o['売上'])
    o['ソフトウェア投資構成比']=safe_div(o['ソフトウェア投資額'],o['ソフトウェア投資額']+o['設備投資額']); o['DX部署_ソフトウェア投資']=o['DX部署あり']*o['ソフトウェア投資売上比率']
    return o.replace([np.inf,-np.inf],np.nan)

def _count(s,words): return s.astype(str).str.count('|'.join(re.escape(w) for w in words)).astype('float32')

def semantic(ann,ids):
    z=pd.DataFrame({ID:pd.Series(ids).drop_duplicates()})
    masks={'future_all':ann.label.eq('future'),'past_all':ann.label.eq('past'),'future_c1':ann.label.eq('future')&ann.contrast_flag.eq(1),'past_c1':ann.label.eq('past')&ann.contrast_flag.eq(1)}
    for name,m in masks.items():
        g=ann.loc[m].groupby(ID).agg(**{name+'_n':('sentence','size'),name+'_text':('sentence',lambda x:' '.join(x.astype(str)))}).reset_index(); z=z.merge(g,on=ID,how='left')
    for c in [c for c in z if c.endswith('_text')]:z[c]=z[c].fillna('')
    for c in [c for c in z if c.endswith('_n')]:z[c]=z[c].fillna(0)
    z['future_contrast_ratio']=(z.future_c1_n/z.future_all_n.replace(0,np.nan)).fillna(0); z['past_contrast_ratio']=(z.past_c1_n/z.past_all_n.replace(0,np.nan)).fillna(0)
    for name,words in LEX.items():
        ctx='future' if name.startswith('future') else 'past'; z[name]=_count(z[ctx+'_all_text'],words); z[name+'_c1']=_count(z[ctx+'_c1_text'],words)
    z['dx_momentum']=z.past_achievement+z.future_aggressive-z.future_cautious; z['opportunity_to_expansion']=z.past_opportunity+z.future_aggressive; z['stagnation_to_caution']=z.past_unfinished+z.future_cautious; z['past_contrast_quality']=z.past_opportunity_c1-z.past_unfinished_c1; z['future_contrast_brake']=z.future_cautious_c1-z.future_aggressive_c1
    return z

def add_rates(o):
    o=o.copy(); fa=o.future_all_n.clip(lower=1); pa=o.past_all_n.clip(lower=1); fc=o.future_c1_n.clip(lower=1); pc=o.past_c1_n.clip(lower=1)
    o['future_aggressive_rate']=o.future_aggressive/fa; o['future_cautious_rate']=o.future_cautious/fa; o['future_net_intent_rate']=(o.future_aggressive-o.future_cautious)/fa; o['future_c1_aggressive_rate']=o.future_aggressive_c1/fc; o['future_c1_cautious_rate']=o.future_cautious_c1/fc; o['future_c1_net_rate']=(o.future_aggressive_c1-o.future_cautious_c1)/fc
    o['past_achievement_rate']=o.past_achievement/pa; o['past_opportunity_rate']=o.past_opportunity/pa; o['past_unfinished_rate']=o.past_unfinished/pa; o['past_progress_rate']=(o.past_achievement+o.past_opportunity-o.past_unfinished)/pa; o['past_c1_opportunity_rate']=o.past_opportunity_c1/pc; o['past_c1_unfinished_rate']=o.past_unfinished_c1/pc; o['past_c1_quality_rate']=(o.past_opportunity_c1-o.past_unfinished_c1)/pc
    o['transition_score']=o.past_progress_rate+o.future_net_intent_rate; o['expansion_after_success']=o.past_achievement_rate*o.future_aggressive_rate; o['caution_after_stagnation']=o.past_unfinished_rate*o.future_cautious_rate; o['contrast_brake_strength']=o.future_contrast_ratio*o.future_c1_cautious_rate; o['contrast_opportunity_strength']=o.past_contrast_ratio*o.past_c1_opportunity_rate
    return o.replace([np.inf,-np.inf],np.nan)

def build_stage48_model_inputs(train_raw,test_raw,manual):
    cl=train_classifier(train_raw,manual); atr=make_annotations(train_raw,cl,manual); ate=make_annotations(test_raw,cl,None)
    def one(raw,ann,train):
        b=base_features(raw); s=semantic(ann,b[ID]); dense=[c for c in s if c!=ID and not c.endswith('_text')]; o=b.merge(s[[ID]+dense],on=ID,how='left'); o[dense]=o[dense].fillna(0); o=add_rates(o); cols=[ID]+STAGE48_MODEL_FEATURES+([TARGET] if train else []); return o[cols].copy()
    return one(train_raw,atr,True),one(test_raw,ate,False),atr,ate
