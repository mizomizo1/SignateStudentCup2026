# SignateStudentCup2026

SIGNATE × TECH OCEAN **Student Cup 2026** の最終解法を、元リポジトリの探索コードから切り離して再現できるように保存した公開リポジトリです。

このリポジトリは [mizomizo1/signate の `FINAL_V2`](https://github.com/mizomizo1/signate/tree/main/FINAL_V2) を再現用に凍結したものです。最終候補5本について、前処理、手動アノテーション、特徴量生成、CV、threshold / top-k 契約、出力ファイルの SHA256 検証まで含みます。

## Competition

- Event: **SIGNATE × TECH OCEAN Student Cup 2026 夏**
- Task: 企業の基本情報・財務・アンケート・テキスト情報から、DX教育商材の購入有無を予測する二値分類
- Metric: F1
- Event information: https://techportal.jp/areas/domains/01KB1HTG1529RQ4XZA5NVBP1QW/contents/events/01KT2ZFW3J2YZXH2T7ZT0NWT27
- SIGNATE Competition: https://competition-content.signate.jp/

> コンペ配布データそのものはこの公開リポジトリには含めません。参加者はSIGNATE上でコンペに参加し、公式の「データ」タブから取得してください。

## Reproducibility

再現に必要な独自成果物はリポジトリ内に含めています。

- `FINAL_V2/assets/train_annotations.json`: Stage48で使用する22社分の凍結manual annotation
- `FINAL_V2/preprocess/`: Stage48 → Stage54 → Stage60 の前処理・特徴量
- `FINAL_V2/model/`: L1 Logistic Regression / CV / threshold / top-k ロジック
- `FINAL_V2/frozen_manifest.json`: 期待するshape・threshold・F1・positive count・SHA256
- `FINAL_V2/run_final_v2.py`: 5候補を一括再生成
- `FINAL_V2/verify_outputs.py`: 出力CSVのpositive countとSHA256を検証

探索用の `emsamble/make_model/**` 等には依存しません。

## Data setup

SIGNATEから取得した元データを、次のどちらかに配置します。

```text
data/
├── train.csv
└── test.csv
```

または直接、

```text
FINAL_V2/data/
├── train.csv
└── test.csv
```

に置くこともできます。

`run_final_v2.py` は実行前に raw CSV の SHA256 を確認します。想定と異なるデータの場合は学習前に停止します。

Expected SHA256:

```text
train.csv  48792d669790f8b25050d60d6a90a924a9e7fd36081d42eb3ede610708c79425
test.csv   764633807020d59d53986ad5b4f64297892facc639727e9ecf3a361634b10dcd
```

## Run

Python 3 の環境で repository root から実行します。

```bash
git clone https://github.com/mizomizo1/SignateStudentCup2026.git
cd SignateStudentCup2026

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r FINAL_V2/requirements.txt

python FINAL_V2/run_final_v2.py
python FINAL_V2/verify_outputs.py
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r FINAL_V2/requirements.txt
python FINAL_V2/run_final_v2.py
python FINAL_V2/verify_outputs.py
```

## Frozen final candidates

| File | Positives | Contract |
|---|---:|---|
| `SUBMIT_190_STAGE60.csv` | 190 | Stage60 DEV-average OOF best threshold transported to test |
| `SUBMIT_205_STAGE48_HEDGE.csv` | 205 | Stage48 median-seed-threshold positive-rate top-k hedge |
| `SUBMIT_192_STAGE54.csv` | 192 | Stage54 DEV-average OOF best threshold transported to test |
| `SUBMIT_193_STAGE60_TOPK.csv` | 193 | Stage60 probability top-k=193 |
| `SUBMIT_185_STAGE60_TOPK.csv` | 185 | Stage60 probability top-k=185 |

Main model:

- Stage48: 113 features
- Stage54: 117 features
- Stage60: 119 features
- 15 seeds × 5 folds = 75 models per variant
- Stage60 frozen OOF threshold: `0.5586184921185782`
- Stage60 OOF F1: `0.9016393442622951`

生成物は `FINAL_V2/outputs/`、監査用中間ファイルは `FINAL_V2/model_input/` に作成されます。これらは再生成可能なのでGit管理対象外です。

## Manual annotations

`FINAL_V2/assets/train_annotations.json` は、22社の「今後のDX展望」について sentence / paragraph の past・future 判定と contrast flag を凍結したファイルです。

このアノテーションを含めることで、元の探索ディレクトリや作業履歴がなくてもStage48 model-inputを同一SHA256まで再生成できます。

## Verification contract

正常な再現では以下を検証します。

1. raw train/test SHA256
2. Stage48 model-input SHA256
3. Stage48 / Stage54 / Stage60 feature shape
4. 5 submission のpositive count
5. 5 submission の最終SHA256

1 byteでも最終CSVが変化した場合、`verify_outputs.py` は失敗します。

## Repository layout

```text
.
├── README.md
├── .gitignore
└── FINAL_V2/
    ├── LINEAGE.md
    ├── README.md
    ├── frozen_manifest.json
    ├── requirements.txt
    ├── run_final_v2.py
    ├── verify_outputs.py
    ├── assets/
    │   └── train_annotations.json
    ├── data/
    │   ├── README.md
    │   └── sync_data.py
    ├── preprocess/
    └── model/
```

詳細なモデルの系譜は [FINAL_V2/LINEAGE.md](FINAL_V2/LINEAGE.md) を参照してください。

## Data and competition terms

本リポジトリは解法コードと作者が作成した再現用アノテーションを公開するもので、SIGNATEから配布された `train.csv` / `test.csv` は再配布しません。

データを利用する場合は、SIGNATEの当該コンペへの参加・利用条件を確認し、公式経路から各自取得してください。SIGNATEの案内では、参加後にコンペページの「データ」タブから配布ファイルをダウンロードできます。

## Provenance

Frozen source:

- https://github.com/mizomizo1/signate/tree/main/FINAL_V2

このリポジトリでは、最終提出候補の**再現に必要な確定ロジックだけ**を保存しています。探索Stageや採用されなかった多数の実験は含めていません。
