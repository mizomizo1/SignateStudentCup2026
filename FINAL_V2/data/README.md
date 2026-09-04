# FINAL_V2/data

この公開リポジトリには、SIGNATEから配布された raw `train.csv` / `test.csv` は含めません。

コンペ参加者がSIGNATEの公式「データ」タブから取得したファイルを、repository root の `data/` に配置してください。

```text
data/
├── train.csv
└── test.csv
```

`FINAL_V2/data/sync_data.py` は root `data/train.csv` / `data/test.csv` をこのディレクトリへコピーし、Git blob SHA が凍結契約と一致することを確認します。

Expected Git blob SHA:

- `train.csv`: `7e7ef33cca57f8e1564004267cbcde31181ee947` (742 rows)
- `test.csv`: `d6a352d79e211f3d2d1c0bd9c636d4a7af515474` (800 rows)
- target: `購入フラグ`

さらに `run_final_v2.py` 側でも raw CSV の SHA256 を検証してから処理を開始します。
