# FINAL_V2 — 5本の最終候補完全再現パッケージ

今回の最終提出候補 **190 / 205 / 192 / 193 / 185 positives** を、`emsamble/make_model/**` の探索コードへ依存せず、凍結したロジックだけで再生成するパッケージです。

このディレクトリの目的は、単にsubmission CSVを置くことではなく、

- どの入力データを使ったか
- どの前処理・特徴量を使ったか
- どのモデル・seed・foldで学習したか
- どのthreshold / top-k契約で0/1にしたか
- 最後にどのSHA256のCSVが生成されたか

までを一気通貫で再現・監査できる状態にすることです。

---

## 1. 最終候補5本と使い分け

| 優先 | positives | file | モデル/契約 | 役割 |
|---:|---:|---|---|---|
| **1** | **190** | `SUBMIT_190_STAGE60.csv` | Stage60 DEV平均OOFのbest thresholdをtestへ固定輸送 | **本命** |
| **2** | **205** | `SUBMIT_205_STAGE48_HEDGE.csv` | Stage48 backup contract / recall-side top-k | **大きく離したPrivate hedge** |
| 3 | 193 | `SUBMIT_193_STAGE60_TOPK.csv` | Stage60 probability rankingのtop-k=193 | prevalence hedge |
| 4 | 192 | `SUBMIT_192_STAGE54.csv` | Stage54 DEV平均OOF best thresholdをtestへ固定輸送 | model-structure hedge |
| 5 | 185 | `SUBMIT_185_STAGE60_TOPK.csv` | Stage60 probability rankingのtop-k=185 | precision hedge |

### 現時点の最終提出方針

Private leaderboardに2本しか残せない場合の第一案は、

1. **`SUBMIT_190_STAGE60.csv`** — 本命
2. **`SUBMIT_205_STAGE48_HEDGE.csv`** — recall側に十分離したヘッジ

です。

190 / 193 / 185 は同じStage60 probability rankingを使うため、最終2枠に複数残すと予測の相関が高くなります。192はStage54なので構造は少し異なりますが、190に比較的近い候補です。205はStage48系で陽性数も離れており、2枠目の分散確保という意味があります。

Public leaderboardは診断材料として使いますが、Publicの順位だけで190本命を捨てる前提にはしません。

---

## 2. Stage48 → Stage54 → Stage60で何が違うか

FINAL_V2の主要モデルは、Stage48を土台に少数の特徴量・probability blendを段階的に追加しています。

### Stage48 — baseline / 113 features

既存 `FINAL/` で完全再現していた土台です。

主な特徴量カテゴリ:

- 企業規模
  - 従業員数
  - 売上
  - 総資産
  - 自己資本
- 財務
  - signed-log変換
  - 財務ratio / margin系
  - cash / investment / capacity系
- アンケート・設問
  - Q系の数値・カテゴリ情報
  - 大企業・能力・余力とのinteraction
- テキスト意味特徴
  - 企業概要
  - 組織図
  - 今後のDX展望
  - future / past sentence分類
  - DX導入意向、教育・人材、能力、実績等のsemantic rate
- Stage14 / Stage18 / Stage30で確定したfeature engineering

前処理後のStage48 final feature shapeは **742 x 113**。

モデルはL1 Logistic Regressionです。

- `C = 1.5`
- positive class weight = `2.5`
- numeric: median imputation + scaling
- categorical: mode imputation + one-hot encoding
- solver: liblinear / L1
- 15 seeds × 5 folds = **75 models**

### Stage54 — Stage48 + core-four / 117 features

Stage48へ、企業概要・組織図などから得られる「構造化値と会社説明のズレ」「software developmentらしさ」を中心に4つのinteractionを追加します。

追加するcore-four:

1. **10x employee mismatch × large employee**
2. **10x employee mismatch × high operating-CF margin**
3. **software development × large employee**
4. **software development × high Q9**

feature shapeは **742 x 117**。

Stage54 probabilityは単独モデルに置換するのではなく、Stage48とcore modelをblendします。

```text
P54 = P48 + 0.50 * (Pcore - P48)
```

`SUBMIT_192_STAGE54.csv` は、このStage54のDEV平均OOFから決めたbest thresholdをtestへ固定輸送したものです。

### Stage60 — Stage54 + education need × capacity / 119 features

Stage60では、「教材を必要としていそうか」だけでなく、「その必要性を実際に導入へつなげられる余力があるか」を弱く加えます。

追加する主特徴:

- `education_need`
- `education_need_x_capacity`

`education_need` は、企業テキスト内の広い教育・人材文脈をまとめた特徴です。例として、

- 人材不足
- 教育 / 研修
- リスキリング
- スキル / リテラシー
- ノウハウ不足
- DX / AI / データ人材

などを扱います。

`capacity` は、

- 従業員数
- 売上
- 総資産
- 自己資本

のsigned-logを平均した導入余力proxyです。

feature shapeは **742 x 119**。

Stage60 probabilityは、Stage54へneed-capacity modelの差分を25%だけ加えます。

```text
P60 = P54 + 0.25 * (Pneed_capacity - Pcore)
```

このStage60が現在の本命モデルです。

- `SUBMIT_190_STAGE60.csv` — DEV threshold固定輸送
- `SUBMIT_193_STAGE60_TOPK.csv` — 同じStage60 probabilityで上位193件
- `SUBMIT_185_STAGE60_TOPK.csv` — 同じStage60 probabilityで上位185件

つまり190 / 193 / 185は**モデル自体は同じ**で、0/1への切り方だけが違います。

---

## 3. なぜStage61以降をFINAL_V2へ入れていないか

Stage61–67ではsemantic feature、ownership/readiness、複数interactionなどを追加検証しましたが、DEV / HOLD / untouched Fresh panelをまたいでStage60を安定して上回りませんでした。

Stage68–70の `triple05` local probability bumpは、複数Fresh panelでrobustness改善が再現しました。ただし、凍結したStage60 thresholdではtestの0/1ラベルを**1件も変更しません**でした。

そのため0/1 submission再現パッケージであるFINAL_V2には持ち込んでいません。

さらに最後の確認として、増えた特徴量を用いた

- soft voting
- cross-fit stacking ridge
- expanded feature blend

も試しましたが、robust scoreではStage60を安定して超えませんでした。したがってFINAL_V2では複雑化せず、Stage60を主モデルとして凍結しています。

---

## 4. 5本の契約をもう少し詳しく

### 190 — Stage60 transported threshold

DEVの15 seeds × 5 folds OOF予測を平均し、DEV上でF1が最大になるthresholdを1回だけ決定します。

```text
Stage60 threshold = 0.5586184921185782
```

このthresholdを変更せずtest probabilityへ適用した結果が190 positivesです。

### 192 — Stage54 transported threshold

同様にStage54のDEV平均OOFで決めたthresholdをtestへ固定輸送します。

```text
Stage54 threshold = 0.5653207714485066
```

結果は192 positivesです。

### 193 — Stage60 top-k hedge

Stage60 probabilityをそのまま順位付けし、上位193社を1にします。

thresholdをtest陽性数に合わせて探索しているのではなく、**top-k=193という契約を事前に固定したhedge**です。

### 185 — Stage60 precision-side hedge

同じStage60 rankingで上位185社のみを1にします。190よりprecision寄りの候補です。

### 205 — Stage48 recall-side hedge

205はStage60系ではありません。

15個のStage48 seed別OOF best thresholdの中央値からDEV側の陽性率を決め、その比率をtestへ写したbackup contractです。Stage60より陽性数を増やしたrecall側のヘッジです。

---

## 5. 凍結パイプライン

`python FINAL_V2/run_final_v2.py` は次を順番に実行します。

1. raw `train.csv` / `test.csv` のSHA256確認
2. Stage48の決定的な前処理
3. future/past文分類 + 凍結したmanual annotation
4. 77列Stage48 model-input再生成
5. 既知SHA256を確認してCSV保存・再読込
6. Stage48 final feature engineering → 113列
7. Stage51/53 core-four追加 → 117列
8. Stage60 need-capacity追加 → 119列
9. L1 Logistic Regressionを15 seeds × 5 foldsで学習
10. Stage48 / Stage54 / Stage60 probabilityを生成
11. threshold / top-k契約で5本の0/1ラベルを生成
12. positive countをassert
13. 出力CSVと監査情報を `outputs/` へ保存

---

## 6. ローカルでの実行方法

### 前提

- Python 3系
- Git
- repository rootで実行

### macOS / Linux

```bash
git clone <YOUR_REPOSITORY_URL>
cd signate

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r FINAL_V2/requirements.txt

python FINAL_V2/run_final_v2.py
python FINAL_V2/verify_outputs.py
```

### Windows PowerShell

```powershell
git clone <YOUR_REPOSITORY_URL>
cd signate

py -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r FINAL_V2/requirements.txt

python FINAL_V2/run_final_v2.py
python FINAL_V2/verify_outputs.py
```

PowerShellでvirtual environmentのactivateがexecution policyで止まる場合は、現在のPowerShellセッションだけ一時的に許可してからactivateできます。

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### すでにrepositoryをclone済みの場合

```bash
cd signate
git pull
python -m pip install -r FINAL_V2/requirements.txt
python FINAL_V2/run_final_v2.py
python FINAL_V2/verify_outputs.py
```

---

## 7. 正常終了時に確認するもの

submissionは次に出力されます。

```text
FINAL_V2/outputs/
├── SUBMIT_190_STAGE60.csv
├── SUBMIT_205_STAGE48_HEDGE.csv
├── SUBMIT_192_STAGE54.csv
├── SUBMIT_193_STAGE60_TOPK.csv
├── SUBMIT_185_STAGE60_TOPK.csv
└── ... audit files
```

`verify_outputs.py` は、各CSVについて

- ファイルが存在するか
- positive countが期待値と一致するか
- SHA256が `frozen_manifest.json` と一致するか

を検証します。

```bash
python FINAL_V2/verify_outputs.py
```

正常なら検証成功メッセージで終了します。1 byteでもsubmissionが変わればSHA256 mismatchで失敗します。

---

## 8. データの扱い

`FINAL_V2/data/train.csv` / `test.csv` が無い環境では、同じrepositoryのroot `data/` から同期します。

ただし単にコピーするのではなくraw SHA256を検証します。違うtrain/testを誤って配置した場合は、モデル学習へ進む前に停止します。

凍結raw SHA256:

```text
train.csv = 48792d669790f8b25050d60d6a90a924a9e7fd36081d42eb3ede610708c79425
test.csv  = 764633807020d59d53986ad5b4f64297892facc639727e9ecf3a361634b10dcd
```

---

## 9. 検証済み値

FINAL_V2で確認している主要値:

- Stage48 feature shape: `742 x 113`
- Stage54 core shape: `742 x 117`
- Stage60 need-capacity shape: `742 x 119`
- Stage48 OOF threshold: `0.4790796951438194`
- Stage54 OOF threshold: `0.5653207714485066`
- Stage60 OOF threshold: `0.5586184921185782`
- Stage48 OOF F1: `0.898936170212766`
- Stage54 OOF F1: `0.8986301369863013`
- Stage60 OOF F1: `0.9016393442622951`
- 出力陽性数: **190 / 205 / 192 / 193 / 185**

Stage48中間model-inputも既存FINALと一致します。

```text
train SHA256 = d3ae8f3b180af1c6ad67c4a68487bde9916494e394f120e94ccb796993567a6b
test SHA256  = 7c8b24a204be0a47ab7d0251744c7205ba252fd8aad0fd212e6c351988acb298
```

最終5 CSVの期待SHA256は `frozen_manifest.json` に固定しています。

---

## 10. ディレクトリ構成

```text
FINAL_V2/
├── README.md
├── LINEAGE.md
├── frozen_manifest.json
├── requirements.txt
├── run_final_v2.py
├── verify_outputs.py
├── data/
│   ├── sync_data.py
│   └── ... raw CSV
├── assets/
│   └── Stage48用の凍結manual annotation
├── preprocess/
│   └── Stage48 / Stage51-53 / Stage60前処理
├── model/
│   └── L1 model / CV / threshold / top-k logic
├── model_input/
│   └── 実行時に再生成する監査用中間CSV
└── outputs/
    └── submissions / probabilities / metrics / manifests
```

探索Stageから最終ロジックへの詳しい系譜は `LINEAGE.md` を参照してください。

---

## 11. 最終提出前チェックリスト

最終提出直前は次だけ確認すれば十分です。

- `git pull` 済み
- `python FINAL_V2/run_final_v2.py` が正常終了
- `python FINAL_V2/verify_outputs.py` が正常終了
- `SUBMIT_190_STAGE60.csv` のpositive countが190
- 2本目として選ぶhedgeのファイルを取り違えていない
- CSVをExcel等で開いて上書き保存していない

提出CSVは生成されたものをそのままアップロードしてください。
