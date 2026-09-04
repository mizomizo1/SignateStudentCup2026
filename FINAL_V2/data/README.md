# FINAL_V2/data

`FINAL/` と同様に、最終再現では raw `train.csv` / `test.csv` を固定入力として扱います。

現在の正本は repository root の `data/` です。Git blob SHA は次の通りです。

- `train.csv`: `7e7ef33cca57f8e1564004267cbcde31181ee947` (742 rows)
- `test.csv`: `d6a352d79e211f3d2d1c0bd9c636d4a7af515474` (800 rows)
- target: `購入フラグ`

`sync_data.py` は root `data/train.csv` / `data/test.csv` をこのディレクトリへコピーし、Git blob SHA が上記契約と一致することを検証します。

最終版では `FINAL/` と同様、`FINAL_V2/data/train.csv` と `FINAL_V2/data/test.csv` が再現パッケージの固定入力になります。
