# RinGo

*日本語 / [English README](README.en.md)*

[![CI](https://github.com/kanade73/RinGo/actions/workflows/ci.yml/badge.svg)](https://github.com/kanade73/RinGo/actions/workflows/ci.yml)

KataGo の囲碁エンジンを Apple Silicon 向けに Swift + [MLX](https://github.com/ml-explore/mlx-swift)
へ移植したものです。モデルの読み込み（v8–v17、nested-bottleneck および transformer 系ネットを含む）、
数値的に忠実な MLX 順伝播、V7 入力特徴、バッチ化・コンパイル済み評価器、PUCT MCTS、GTP フロントエンドを
備えています。

数値の忠実性は参照実装である C++ 版（Eigen バックエンド）に対して fp32 で検証しています。5 つの参照
テストモデルすべてで policy / value / score / ownership が ≤1e-4 の範囲で一致し、現行で最強の公開ネット
（`kata1-zhizi-b40c768nbt`、232M パラメータ）ではオラクルの出力表示と完全に一致します。

## このプロジェクトの位置づけ

RinGo は fork ではなく、独立した Swift + MLX 実装です。KataGo の C++ ソースは一切含みません。
ネットワークの重みも配布していません。KataGo 形式の `.bin.gz` ネットワークは
https://katagotraining.org/ から各自で用意してください。

## 動作要件

- Apple Silicon 上の macOS 14 以降。
- Swift 6.2 以降（Swift ツールチェーンまたは Xcode Command Line Tools）。
- Metal デバイス。テストスイートは実際の MLX カーネルを実行します。
- `make check` 用: `brew install swiftformat swiftlint`。エンジン自体のビルドと実行には不要です。
- 後述のオラクル系（`make oracle`、`make goldens`、`Scripts/check-rawnn-parity.sh`）でのみ必要:
  `cmake`、`brew install eigen libzip`、`python3`、および KataGo のチェックアウト。

## クイックスタート

```sh
make setup                      # SwiftPM の依存解決（mlx-swift は 0.31.4 に厳密固定）
Scripts/build-metallib.sh       # SwiftPM CLI 実行に一度だけ必要（MLX の基本カーネル）
make check                      # lint + ビルド + 全テスト（約 2 分）

# GTP で対局（gogui / Sabaki / lizgoban から GTP エンジンとして利用可能）:
swift run -c release ringo gtp -model <net.bin.gz> -visits 400

# SGF 局面の生 NN 評価（参照実装の `evalsgf -raw-nn` と diff 可能）:
swift run -c release ringo rawnn -model <net.bin.gz> -sgf game.sgf -move-num 30

# スループット計測:
swift run -c release ringo benchmark -model <net.bin.gz> -batch-sizes 1,8,32
```

https://katagotraining.org/ の実際の `kata1-*.bin.gz`（v8 以降）は既定の fp16 精度で動作します。
参照クローンの `cpp/tests/models/` 以下にある小型ネットはパリティ検証用で、fp16 では policy 出力が
非有限値になるものがあるため `-precision fp32` が必要です。

## 学習と評価（9×9）

**この版のスコープ。** 推論・探索・GTP 対局は任意の盤サイズで動作します（RinGo は KataGo のネットワークを
用いて 19×19 の大会に出場しています。後述）。**学習パイプラインは 9×9 専用**で、`ringo train` は
`nnLen != 9` のシャードを拒否します。本節の内容はすべて 9×9 が前提です。

独自ネットワークを作る正式な手順は、**ランダム初期化から始める KataGo 教師からの教師あり蒸留**です。
KataGo の重みを変換したり引き継いだりはしません。教師ネットワークは局面のラベル付けにのみ使い、
生徒側のパラメータはランダムに初期化され、本リポジトリ自前の MLX 学習ループ（`Sources/RinGoTrain`）で
学習されます。

### パイプライン

```sh
# 0. 最初に一度だけビルド。
make setup && Scripts/build-metallib.sh
swift build -c release --product ringo

# 1. 局面の生成。SGF ならどの入手元でも構いません。既存ネットとの自己対局だけで完結します。
.build/release/ringo selfplay -model <teacher.bin.gz> -games 6 \
    -out data/sgf -visits 60 -size 9 -komi 7 -precision fp32

# 2. ラベル付け。policy / value / score / ownership のターゲットを持つ RinGoData v2 シャード
#    （`.nngd`）と、ホールドアウト分割を書き出します。
.build/release/ringo makedata -sgf-dir data/sgf -out data/shards -nnlen 9 \
    -teacher-model <teacher.bin.gz> -teacher-precision fp32 \
    -teacher-visits 1 -teacher-target completed-q -val-ratio 0.2

# 3. ランダム初期化から学習。出力ディレクトリは事前に存在している必要があります。
mkdir -p runs/r1
.build/release/ringo train -data data/shards -out runs/r1 -arch b6c96 \
    -batch 32 -steps 40 -val-data data/shards -val-interval 20 -snapshot-interval 40

# 4. ホールドアウトシャードで評価。
.build/release/ringo evaluate -model runs/r1/model-best-val.bin.gz -data data/shards/val.nngd

# 5. 出来上がったネットで対局。
.build/release/ringo gtp -model runs/r1/model-best-val.bin.gz -visits 100 -nnlen 9 -precision fp32
```

`-arch` が受け付けるアーキテクチャ: `b6c96`、`b10c128`、`b15c192`、`b20c256`、`b24c320`、`b28c384`。

policy ターゲット: `-teacher-target visits` はルートの訪問回数分布を正規化したもの、`completed-q` は
ルートの事前分布と子ノードごとの Q 値から構成する Gumbel completed-Q 改良方策
（Danihelka et al., ICLR 2022）です。

### 学習実行の生成物

`runs/r1/` には `config.json`（全ハイパーパラメータ）、`training.log`、`metrics.csv`、
`validation_metrics.csv`、`checkpoint.safetensors`（再開可能なオプティマイザ状態）、
`model-final.bin.gz`、`model-best-val.bin.gz` が置かれます。エクスポートされるモデルは KataGo の
`.bin.gz` 形式なので、同じファイルを RinGo でも参照実装の KataGo でも読み込めます。

`data/shards/makedata-config.json` にはラベル付けの来歴（教師モデル、visits、ターゲット種別、ルール、
対称変換）が記録されます。ディスク上のシャード形式そのもの（`.nngd`、"RinGoData v2"）は
[`Scripts/ringo-data-format.md`](Scripts/ringo-data-format.md) に仕様があります。

### ホールドアウトの健全性

`makedata -val-ratio` はホールドアウト分割を、学習用シャードと**同じディレクトリに** `val.nngd` として
書き出します。`ringo train` は `-val-data` を渡したかどうかに関わらず、無条件にこのパスを学習集合から
除外します。したがって `-data` にそのディレクトリを指定してもホールドアウトが暗黙のうちに学習へ
混ざることはなく、後から同じファイルに対して `ringo evaluate` を実行すれば真のホールドアウト評価に
なります。

### 実行例（2026-07-27 に Apple M5 で確認）

自己対局 6 局 → 375 局面 → 学習 329 / ホールドアウト 46。`b6c96`（1,027,911 パラメータ）をランダム
初期化から 40 ステップ学習し、選択されたチェックポイントを評価した結果:

```
evaluate: samples=46 total=7.15756 pol=4.35212 val=0.83451 score=0.03208 own=0.60254
```

これは `validation_metrics.csv` のステップ 40（`total_loss=7.157561`）と一致しており、単体の評価器と
学習中の検証が整合していることを示します。得られたネットワークは `genmove` に応答します。

これはインターフェースを具体的に示すための最小規模の実行例であり、競技的に通用するネットワークに必要な
データ量・ステップ数には遠く及びません。

## 性能（Apple M5、32 GB、release、fp16）

| モデル | バッチ | evals/s | 参照実装（Eigen CPU、8 スレッド） |
|---|---|---|---|
| b6c96 (1.0M) | 32 | 約 3,000 | 約 1,365 |
| kata1-zhizi-b40c768nbt (232M) | 32 | 約 39 | 約 4.8 |

b6c96 での 19×19・400 visits の genmove: 約 0.17 秒/手。

参照実装の列は KataGo の Eigen **CPU** バックエンドであり、アクセラレータ同士の公平な比較ではありません。
ここで挙げているのは、これらの数値を検証した対象と同じビルドだからです。Mac 上での GPU 対 GPU の公平な
比較を求める場合は KataGo の OpenCL バックエンドと比較してください。その計測はここには含まれていません。

## 大会成績

- CGF Open 2026（日本）2026-07-25、9路: 16 チーム中 7 位（6 勝 5 敗 1 分）。RinGo 自身が学習した
  ネットワークを使用。
- 2026-07-26、19路: 12 チーム中 4 位（5 勝 2 敗）。公式 KataGo ネットワーク `kata1-b18c384nbt` を使用。
  RinGo は 19路用の自前ネットワークを持たないため、主催者の事前許可を得たうえでの使用です。
- 両日とも学生エントリー中の最上位。

## 構成

- `Sources/RinGoCore` — 盤面 / ルール / 履歴と V7 特徴（純 Swift、MLX 非依存。C++ から関数単位で移植し、
  参照テストベクタで検証済み）。
- `Sources/RinGoModel` — `.bin.gz` / `.txt.gz` モデルパーサと `KataGoNetwork` の MLX 順伝播。
- `Sources/RinGoEngine` — 後処理、SGF リーダ、バッチ化された `NNEvaluator` アクタ（MLX の実行を統制する
  ため、バケットごとにコンパイルしパイプライン化）、MCTS、GTP。
- `Sources/RinGoTrain` — 学習の損失関数、オプティマイザ、データ読み込み、トレーナ周辺。
- `Sources/ringo` — `ringo` コマンドライン実行ファイルとそのサブコマンド。
- `Scripts/oracle/` — 参照 KataGo ビルドから正解データを取り出すツール群（`dump_v7`、`dump_nn`、
  ブロック単位で二分探索する `dump_nn_debug`）。

アーキテクチャは純 Swift の囲碁ロジック、MLX でのモデル実行、探索、学習、CLI を分離しています。検証には
参照フィクスチャを用い、数値許容誤差と非目標を明示しています。コードからは形が読み取りにくい部分——MLX の
遅延評価をどう制御しているか、パリティが依存する数値上の規約、オプションの DAG 探索が何をするのか——は
[`docs/design-notes.md`](docs/design-notes.md) で扱っています。

`Scripts/run-matches.sh` は他の GTP プログラムと対戦させて棋力を測るスクリプトです。`gogui-twogtp` を
`../gogui/bin/gogui-twogtp` に想定して呼び出しますが、これは同梱していません。

## 検証の方針

正解は参照実装の C++ KataGo（読み取り専用クローン、Eigen CPU バックエンド）です。リポジトリに含む
ゴールデンフィクスチャが、特徴量（プレーン単位で完全一致）、ネットワーク出力（fp32 で ≤1e-4）、
エンドツーエンドの `rawnn` 出力（局面 × モデルの 18/18 組）をカバーします。`make oracle` が参照バイナリを
再ビルドし、`make goldens` / `Scripts/gen-*-goldens.sh` がフィクスチャを再生成、
`Scripts/check-rawnn-parity.sh` がその 18 組のエンドツーエンド比較を再実行し、実際に検査した組数を
報告します。

モデルを必要とするテストは `KATAGO_MODELS_DIR` から参照ネットを探し、既定値は
`../katago-origin/KataGo/cpp/tests/models`（本リポジトリと並置した KataGo チェックアウト）です。ネットが
無い場合、テストは失敗せず `XCTSkip` します。`make check` はどちらの環境でも 438 テストを実行し、ネットが
ある場合は 40 スキップ、無い新規クローンでは 92 スキップと報告します。

## ライセンス

MIT（[LICENSE](LICENSE) 参照）。サードパーティの帰属表示は
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) に記載しています。
