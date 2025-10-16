# NIR Face Landmark Detection Training

近赤外線（NIR）画像用の顔特徴点検出器の学習・評価環境

## 概要

このプロジェクトは、dlibライブラリを使用して近赤外線画像に特化した68点顔特徴点検出器を学習・評価するためのツール群です。

## プロジェクト構成

```
Landmark_Training/
├── README.md                   # このファイル
├── requirements.txt            # 必要なライブラリ
├── .gitignore                 # Git除外ファイル
│
├── create_dlib_xml.py         # dlib学習用XMLファイル作成スクリプト
├── train_dlib_detector.py     # dlib特徴点検出器学習スクリプト
├── predict_landmarks.py       # 特徴点予測スクリプト（一括処理）
├── visualize_dlib_data.py     # 学習データ可視化スクリプト
├── test_nir_detector.py       # 検出器テストスクリプト（単一画像）
├── test.ipynb                 # Jupyter Notebook（実験用）
│
├── nir_shape_predictor.dat    # 学習済みモデル
│
├── datasetA/                  # 学習用データセット
│   ├── images/               # 学習用画像（.png, .jpg, .bmp）
│   └── landmarks/            # ランドマークファイル（.npy形式）
│
├── datasetB/                  # テスト用データセット
│   ├── images/               # テスト用画像
│   └── landmarks/            # ランドマークファイル
│
├── datasetB_predictions/      # 予測結果
│   ├── detection_report.json # 検出レポート
│   ├── landmarks/            # 予測されたランドマーク（.npy）
│   └── visualizations/       # 可視化画像（.png）
│
├── dlib_data/                 # dlib形式の学習データ
│   ├── images/               # 画像ファイル（シンボリックリンクまたはコピー）
│   └── training_data.xml     # dlib用XMLファイル
│
├── dlib_data_2/               # 追加の学習データ
│   ├── images/
│   └── training_data.xml
│
└── s1/, s2/, s3/, s4/         # 被験者別の生データ
    └── 00001/, 00002/, ...    # セッション別データ
```

## 環境構築

### 必要要件

- Python 3.7以上
- dlib (コンパイル済み)
- OpenCV
- NumPy
- matplotlib

### インストール

```bash
# リポジトリのクローン
git clone <repository-url>
cd Landmark_Training

# 必要ライブラリのインストール
pip install -r requirements.txt

# dlibのインストール（事前コンパイル推奨）
pip install dlib
```

## 使用方法

### ワークフロー全体

```
[データセット] → [XML作成] → [可視化確認] → [モデル学習] → [テスト] → [一括予測]
  datasetA/      .xml         vis/          .dat        単一画像    複数画像
```

### 1. 学習用XMLファイルの作成

学習用データセット（`images/` + `landmarks/`）からdlib形式のXMLファイルを作成：

**推奨: 顔検出器を使用**
```bash
python create_dlib_xml.py --input datasetA --output dlib_data/training_data.xml --use-detector
```

**代替: ランドマークから自動計算**
```bash
python create_dlib_xml.py --input datasetA --output dlib_data/training_data.xml --margin 0.25
```

**オプション:**
- `--use-detector`: 顔検出器でバウンディングボックスを検出（推奨）
- `--margin 0.2`: ランドマークから計算時のマージン（デフォルト: 20%）
- `--samples 100`: サンプル数制限（テスト用）
- `--no-validate`: 妥当性検証をスキップ（高速化）

**バウンディングボックスについて:**
- 特徴点検出器の学習には、「どの領域で特徴点を学習するか」を指定するためのバウンディングボックスが必要です
- `--use-detector`を使うと、実際の推論時と同じ顔検出器を使用するため一貫性が保たれます
- NIR画像で顔検出が上手くいかない場合は、`--margin`オプションでランドマークから自動計算できます

### 2. 学習データの可視化

作成したXMLデータを確認：

```bash
# 全データを可視化して保存
python visualize_dlib_data.py --xml dlib_data/training_data.xml --output vis_results

# 最初の10枚のみ表示
python visualize_dlib_data.py --xml dlib_data/training_data.xml --show --samples 10

# OpenCVで可視化（軽量版）
python visualize_dlib_data.py --xml dlib_data/training_data.xml --output vis_results --opencv
```

### 3. 特徴点検出器の学習

dlib特徴点検出器（Shape Predictor）を学習：

```bash
# 基本的な学習
python train_dlib_detector.py --xml dlib_data/training_data.xml --output nir_shape_predictor.dat

# パラメータを調整した学習
python train_dlib_detector.py --xml dlib_data/training_data.xml --output model.dat \
    --tree-depth 4 --cascade-depth 15 --nu 0.1 --feature-pool-size 500
```

**学習パラメータの説明:**
- `--tree-depth`: 回帰木の深さ（2-5推奨、大きいほど複雑だが過学習リスク）
- `--cascade-depth`: カスケード数（10-20推奨、大きいほど精度向上だがモデルサイズ増加）
- `--nu`: 学習率（0.01-0.25、小さいほど慎重な学習）
- `--feature-pool-size`: 特徴プールサイズ（400-1000、大きいほど精度向上だが学習時間増加）
- `--oversampling`: データ増強倍数（デフォルト: 10）
- `--quiet`: 詳細ログを非表示

### 4. 単一画像でのテスト

学習済みモデルで個別の画像をテスト：

```bash
# 基本的なテスト
python test_nir_detector.py --model nir_shape_predictor.dat --image test_image.png

# 結果を保存して表示
python test_nir_detector.py --model nir_shape_predictor.dat --image test.png \
    --output result.png --show

# 特徴点番号を描画
python test_nir_detector.py --model nir_shape_predictor.dat --image test.png \
    --output result.png --draw-numbers
```

### 5. データセット全体の一括予測

学習済みモデルでテストデータセットを一括予測：

```bash
python predict_landmarks.py --model nir_shape_predictor.dat \
    --input datasetB/images --output datasetB_predictions
```

**出力内容:**
- `datasetB_predictions/landmarks/`: 予測座標（.npyファイル）
- `datasetB_predictions/visualizations/`: 可視化画像（.png）
- `datasetB_predictions/detection_report.json`: 検出統計レポート

### 6. 結果の確認

予測レポート（`detection_report.json`）の例：
```json
{
  "total_images": 100,
  "successful_detections": 95,
  "failed_detections": 5,
  "failed_files": ["path/to/failed_image.png"]
}
```

## 特徴点について

本プロジェクトは68点顔特徴点を検出します：

- **輪郭 (0-16)**: 顔の外輪郭
- **眉毛 (17-26)**: 左右の眉毛
- **鼻 (27-35)**: 鼻の形状
- **目 (36-47)**: 左右の目
- **口 (48-67)**: 口の形状

## データ形式

### 入力データセット形式

学習データセットは以下の構造が必要です：

```
datasetA/
├── images/
│   ├── 00001.png
│   ├── 00002.png
│   └── ...
└── landmarks/
    ├── 00001.npy
    ├── 00002.npy
    └── ...
```

- **画像**: PNG, JPG, JPEG, BMP形式
- **ランドマーク**: NumPy配列（.npy形式）、形状は `(N, 2)` で N は特徴点数（通常68点）

### 出力データ形式

- **XMLファイル**: dlib学習用XML（`training_data.xml`）
- **予測座標**: NumPy配列（.npy）、形状は `(68, 2)`
- **可視化画像**: PNG画像、特徴点とバウンディングボックスを描画
- **レポート**: JSON形式、検出統計情報を含む

## トラブルシューティング

### dlibのインストールエラー
```bash
# 方法1: コンパイル済みパッケージを使用（推奨）
conda install -c conda-forge dlib

# 方法2: バイナリパッケージ
pip install dlib-binary

# 方法3: ソースからビルド（CMake必要）
pip install dlib
```

### XMLファイル作成時の問題

**顔検出失敗が多い場合:**
```bash
# ランドマークから自動計算に切り替え
python create_dlib_xml.py --input datasetA --output dlib_data/training_data.xml --margin 0.3

# または、マージンを調整
python create_dlib_xml.py --input datasetA --output dlib_data/training_data.xml --margin 0.15
```

**画像とランドマークのペアが見つからない場合:**
- ファイル名が一致しているか確認（拡張子を除く）
- ファイルパスに日本語や特殊文字が含まれていないか確認

### 学習時の問題

**メモリ不足エラー:**
```bash
# パラメータを調整して学習
python train_dlib_detector.py --xml dlib_data/training_data.xml --output model.dat \
    --cascade-depth 8 --feature-pool-size 300
```

**学習に時間がかかりすぎる:**
- `--cascade-depth`を減らす（10 → 8）
- `--feature-pool-size`を減らす（400 → 300）
- `--oversampling`を減らす（10 → 5）

### 予測時の問題

**顔が検出されない:**
- 画像の明度・コントラストを調整
- 画像サイズが小さすぎないか確認（最低でも200x200ピクセル推奨）
- NIR画像の場合、前処理（ヒストグラム平坦化など）を検討

**精度が低い:**
- 学習データの品質を確認（正確にアノテーションされているか）
- 学習データの量を増やす
- 学習パラメータを調整（`--cascade-depth`を増やすなど）

## 開発情報

### スクリプト説明

| ファイル | 説明 | 主な機能 |
|---------|------|---------|
| `create_dlib_xml.py` | dlib学習用XMLファイル作成 | 画像+ランドマークデータからXML生成、バウンディングボックス自動計算 |
| `visualize_dlib_data.py` | 学習データ可視化 | XMLデータの内容確認、68点の色分け表示 |
| `train_dlib_detector.py` | 特徴点検出器学習 | Shape Predictorモデルの学習、パラメータカスタマイズ |
| `test_nir_detector.py` | 単一画像テスト | 学習済みモデルで個別画像をテスト |
| `predict_landmarks.py` | データセット一括予測 | 複数画像の特徴点を一括検出、レポート生成 |

### データフロー

```
1. 準備
   datasetA/images/ + landmarks/ → [create_dlib_xml.py] → training_data.xml

2. 確認
   training_data.xml → [visualize_dlib_data.py] → 可視化画像

3. 学習
   training_data.xml → [train_dlib_detector.py] → nir_shape_predictor.dat

4. テスト
   nir_shape_predictor.dat → [test_nir_detector.py] → 単一画像結果

5. 予測
   nir_shape_predictor.dat + datasetB/ → [predict_landmarks.py] → 一括予測結果
```

### 重要な技術ポイント

**1. バウンディングボックス**
- 特徴点検出器の学習には、顔の大まかな位置（バウンディングボックス）が必要
- これは推論時の顔検出器の出力を模倣するため
- 2つの方法: 顔検出器による自動検出 or ランドマークからの自動計算

**2. 学習パラメータ**
- `tree_depth`: 決定木の複雑さ（大きいほど複雑だが過学習のリスク）
- `cascade_depth`: 学習の反復回数（大きいほど精度向上だが時間増加）
- `nu`: 学習の慎重さ（小さいほど安定だが収束に時間）
- `feature_pool_size`: 特徴選択の候補数（大きいほど良い特徴を見つけやすい）

**3. NIR画像の特殊性**
- 近赤外線画像は可視光画像と異なる特性を持つ
- 標準の顔検出器が上手く動かない場合がある
- そのため、ランドマークからバウンディングボックスを計算する機能を用意

### 拡張可能性

- **特徴点数の変更**: 68点以外（例: 5点、194点など）にも対応可能
- **カスタム前処理**: ヒストグラム平坦化、ノイズ除去などの前処理を追加
- **異なる検出器**: CNNベースの検出器との性能比較
- **マルチモーダル**: 可視光 + NIR画像の組み合わせ

### ライセンスと引用

このプロジェクトで使用しているdlibについて：
- [dlib C++ Library](http://dlib.net/) - Davis King氏によるオープンソースライブラリ
- 顔特徴点検出のアルゴリズム: Ensemble of Regression Trees (ERT)

## 参考資料

- [dlib公式ドキュメント](http://dlib.net/)
- [顔特徴点検出について](http://dlib.net/face_landmark_detection.py.html)
- [dlib Shape Predictor Training](http://dlib.net/train_shape_predictor.py.html)
- [近赤外線画像処理 - OpenCV](https://docs.opencv.org/master/)
- [One Millisecond Face Alignment with an Ensemble of Regression Trees](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Kazemi_One_Millisecond_Face_2014_CVPR_paper.pdf) - ERT論文
