# FINAL_V2 lineage

`FINAL_V2` は探索履歴を丸ごと保存するものではなく、今回提出する5本を再現するために必要な確定ロジックだけを残します。

## 1. Stage48 baseline

起点は既存 `FINAL/` が完全再現していたStage48です。

- deterministic raw preprocessing
- future/past sentence classification + frozen 22-company manual annotations
- financial ratios / semantic rate features
- Stage14/18/30由来の最終feature engineering
- L1 Logistic Regression, `C=1.5`, positive class weight `2.5`
- 15 seeds × 5 folds

`FINAL_V2` はStage48の77-column model-inputを独立に再生成し、既知SHA256一致を必須条件にしています。

## 2. Stage51 → Stage53 → Stage54

Stage51で企業概要・組織図からfold-independentなrequested featuresを抽出し、Stage53で次のcore-four interactionを固定しました。

1. 10x employee mismatch × large employee
2. 10x employee mismatch × high operating-CF margin
3. software development × large employee
4. software development × high Q9

Stage54 probabilityは

`P54 = P48 + 0.50 * (Pcore - P48)`

です。今回の192-positive候補はこのStage54 DEV-average OOF thresholdをそのままtestへ輸送したものです。

## 3. Stage60

Stage60はStage54へ、第一原理の広い教育・人材needと導入余力を追加したモデルです。

`need_capacity`で追加するのは

- `education_need`
- `education_need_x_capacity`

で、capacityは従業員数・売上・総資産・自己資本のsigned-log平均です。

Stage60 probabilityは

`P60 = P54 + 0.25 * (Pneed_capacity - Pcore)`

です。

今回の本命190はDEV-average OOFで決めたthresholdをtestへ固定輸送します。193と185はモデルを変更せず、同じStage60 probability rankingへ固定top-kを適用したprevalence/precision hedgeです。

## 4. 205 hedge

205はStage60系ではありません。既存FINALで使っていたStage48のbackup contractを残しています。

15個のStage48 seed別OOF best thresholdの中央値を取り、そのthresholdでDEV平均OOFが作る陽性率をtest件数へ写し、test probabilityのtop-kを陽性にします。この契約で205件になります。

## 5. 採用しなかった探索

Stage61–67のsemantic/interaction challengerは複数panelでStage60を安定して上回らなかったため持ち込みません。

Stage68–70の`triple05` local probability bumpは複数Fresh panelでrobustness改善が再現しましたが、凍結したStage60 thresholdではtestのbinary labelがStage60と完全一致しました。したがって0/1提出再現パッケージには不要です。

## 6. 5 submissions

| file | positives | origin |
|---|---:|---|
| `SUBMIT_190_STAGE60.csv` | 190 | Stage60 transported DEV threshold |
| `SUBMIT_205_STAGE48_HEDGE.csv` | 205 | historical Stage48 median-rate top-k |
| `SUBMIT_192_STAGE54.csv` | 192 | Stage54 transported DEV threshold |
| `SUBMIT_193_STAGE60_TOPK.csv` | 193 | Stage60 fixed top-k |
| `SUBMIT_185_STAGE60_TOPK.csv` | 185 | Stage60 fixed top-k |

すべて `run_final_v2.py` の一回の実行から生成され、陽性件数とSHA256をmanifestへ記録します。
