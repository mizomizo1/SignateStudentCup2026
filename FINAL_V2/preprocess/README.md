# FINAL_V2/preprocess

ここには最終候補 5 本を再生成するための前処理だけを凍結します。

方針は `FINAL/src/preprocess.py` を参考にしますが、そのままコピーはしません。今回の本命は Stage60 系なので、Stage48 baseline の前処理に加えて Stage54 / Stage60 で実際に使う追加特徴まで、過去 Stage ディレクトリに依存しない形へ整理します。

最終 contract は次の順で固定します。

1. raw CSV の型・列検証
2. Stage48 baseline feature reconstruction
3. Stage54 employee-mismatch / software core interactions
4. Stage60 education-need × capacity features
5. 5 candidate submissions すべてが同じ preprocess output から再現できることを検証

探索用の Stage61+ 特徴、Stage67-70 の診断専用特徴は、提出ラベル生成に必要でない限り入れません。
