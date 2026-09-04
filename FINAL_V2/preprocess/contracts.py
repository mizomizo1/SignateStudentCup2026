"""Frozen preprocessing contracts shared by FINAL_V2 modules."""
ID = "企業ID"
TARGET = "購入フラグ"

STAGE48_MODEL_FEATURES = [
    "従業員数","業界","上場種別","組織図","DX部署あり","事業所数","工場数","店舗数","資本金","総資産","流動資産","固定資産","負債","短期借入金","長期借入金","純資産","自己資本","売上","営業利益","経常利益","当期純利益","営業CF","減価償却費","運転資本変動","投資CF","有形固定資産変動","無形固定資産変動(ソフトウェア関連)",
    "アンケート１","アンケート２","アンケート３","アンケート４","アンケート５","アンケート６","アンケート７","アンケート８","アンケート９","アンケート１０","アンケート１１",
    "自己資本比率","負債比率","純資産比率","流動資産比率","固定資産比率","総借入金","借入金自己資本比率","短期借入金比率","長期借入金比率","営業利益率","経常利益率","純利益率","ROA","ROE","総資産回転率","簡易フリーCF","営業CFマージン","投資CFマージン","フリーCFマージン","利益CF比率","設備投資額","ソフトウェア投資額","設備投資売上比率","減価償却売上比率","運転資本変動売上比率","ソフトウェア投資売上比率","ソフトウェア投資構成比","DX部署_ソフトウェア投資",
    "future_aggressive_rate","future_cautious_rate","future_net_intent_rate","future_c1_aggressive_rate","future_c1_cautious_rate","future_c1_net_rate","transition_score","expansion_after_success","caution_after_stagnation","contrast_brake_strength","contrast_opportunity_strength",
]
assert len(STAGE48_MODEL_FEATURES) == 77

STAGE60_FEATURES = [
    "education_need","internalize","outsource","internalize_net","ambition",
    "capability_constraint","ambition_gap","education_need_x_capacity",
    "internalize_need","gap_x_capacity",
]
