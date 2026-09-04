"""Frozen Stage60 first-principles feature preprocessing.

This module is copied from the accepted Stage60 mechanism definition and has no
runtime dependency on emsamble/make_model/**.  It only consumes raw train/test
frames and returns the ten Stage60 mechanism features used by need_capacity25.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

EDU_NEED = (
    r"人材.{0,8}(?:不足|育成|教育|研修|強化)|"
    r"(?:DX|デジタル|IT|AI|データ).{0,10}(?:人材|教育|研修|スキル|リテラシ)|"
    r"(?:教育|研修|人材育成|リスキリング|学び直し|スキルアップ|リテラシー).{0,10}(?:DX|デジタル|IT|AI|データ)?|"
    r"(?:スキル|知識|ノウハウ|リテラシー).{0,8}(?:不足|向上|強化|習得)"
)
INTERNALIZE = (
    r"内製化|内製|自社(?:内)?(?:で|開発|運用)|自走|自律(?:的)?に|"
    r"ノウハウ.{0,8}(?:蓄積|内製|社内)|知見.{0,8}(?:蓄積|社内)|"
    r"社内.{0,8}(?:人材|育成|教育|研修|スキル)"
)
OUTSOURCE = r"外注|外部委託|業務委託|ベンダー.{0,8}(?:活用|委託|依存)|外部.{0,8}(?:パートナー|専門家|コンサル)"
AMBITION = r"DX.{0,10}(?:推進|強化|加速|拡大)|デジタル.{0,10}(?:推進|強化|活用)|AI.{0,10}(?:導入|活用|推進)|データ.{0,10}(?:活用|高度化|分析)|(?:導入|活用|展開).{0,8}(?:拡大|強化|加速)"
CAPABILITY_CONSTRAINT = (
    r"(?:人材|スキル|知識|ノウハウ|リテラシー|体制).{0,8}(?:不足|課題|未整備|弱い|不十分)|"
    r"(?:不足|課題).{0,8}(?:人材|スキル|知識|ノウハウ|体制)|"
    r"(?:社内|全社|現場).{0,8}(?:浸透|定着).{0,6}(?:課題|不足|遅れ)|"
    r"(?:DX|デジタル化).{0,10}(?:遅れ|停滞|未整備)"
)

FEATURE_COLUMNS = [
    "education_need",
    "internalize",
    "outsource",
    "internalize_net",
    "ambition",
    "capability_constraint",
    "ambition_gap",
    "education_need_x_capacity",
    "internalize_need",
    "gap_x_capacity",
]


def norm(value: object) -> str:
    if pd.isna(value):
        return ""
    return unicodedata.normalize("NFKC", str(value)).replace("\u3000", " ")


def slog(x: pd.Series) -> pd.Series:
    z = pd.to_numeric(x, errors="coerce").astype(float)
    return np.sign(z) * np.log1p(np.abs(z))


def capacity(raw: pd.DataFrame) -> pd.Series:
    """Exact Stage60 signed-log capacity proxy."""
    parts = [
        slog(pd.to_numeric(raw["従業員数"], errors="coerce").clip(lower=0)),
        slog(pd.to_numeric(raw["売上"], errors="coerce").abs()),
        slog(pd.to_numeric(raw["総資産"], errors="coerce").abs()),
        slog(pd.to_numeric(raw["自己資本"], errors="coerce").abs()),
    ]
    return pd.concat(parts, axis=1).mean(axis=1)


def make_stage60_features(raw: pd.DataFrame) -> pd.DataFrame:
    future = raw["今後のDX展望"].map(norm)
    overview = raw["企業概要"].map(norm)
    characteristic = raw["特徴"].map(norm)
    org = raw["組織図"].map(norm)
    all_text = future + "\n" + overview + "\n" + characteristic + "\n" + org

    out = pd.DataFrame(index=raw.index)
    out["education_need"] = all_text.str.contains(EDU_NEED, regex=True, na=False).astype(float)
    out["internalize"] = all_text.str.contains(INTERNALIZE, regex=True, na=False).astype(float)
    out["outsource"] = all_text.str.contains(OUTSOURCE, regex=True, na=False).astype(float)
    out["internalize_net"] = out["internalize"] - out["outsource"]
    out["ambition"] = future.str.contains(AMBITION, regex=True, na=False).astype(float)
    out["capability_constraint"] = all_text.str.contains(CAPABILITY_CONSTRAINT, regex=True, na=False).astype(float)
    out["ambition_gap"] = out["ambition"] * out["capability_constraint"]
    cap = capacity(raw)
    out["education_need_x_capacity"] = out["education_need"] * cap
    out["internalize_need"] = out["internalize"] * out["education_need"]
    out["gap_x_capacity"] = out["ambition_gap"] * cap
    return out[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
