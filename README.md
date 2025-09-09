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
├── make_landmarks.py          # 学習データ作成スクリプト
├── train_dlib_detector.py     # dlib検出器学習スクリプト
├── predict_landmarks.py       # 特徴点予測スクリプト
├── visualize_dlib_data.py     # 学習データ可視化スクリプト
├── test_nir_detector.py       # 検出器テストスクリプト
├── test.ipynb                 # Jupyter Notebook（実験用）
│
├── nir_shape_predictor.dat    # 学習済みモデル
│
├── datasetA/                  # 学習用データセット
│   ├── images/               # 学習用画像
│   └── landmarks/            # ランドマークファイル
│
├── datasetB/                  # テスト用データセット
│   ├── images/               # テスト用画像
│   └── landmarks/            # ランドマークファイル
│
├── datasetB_predictions/      # 予測結果
│   ├── detection_report.json # 検出レポート
│   ├── landmarks/            # 予測されたランドマーク
│   └── visualizations/       # 可視化画像
│
├── dlib_data/                 # dlib形式の学習データ
│   ├── images/               # 画像ファイル
│   └── training_data.xml     # dlib用XMLファイル
│
├── dlib_data_2/               # 追加の学習データ
│   ├── images/
│   └── training_data.xml
│
└── s1/, s2/, s3/, s4/         # 被験者別データ
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

### 1. 学習データの作成

学習用データセットからdlib形式のXMLファイルを作成：

```bash
python make_landmarks.py --input datasetA --output dlib_data
```

### 2. 学習データの可視化

作成した学習データを確認：

```bash
# 全データを可視化して保存
python visualize_dlib_data.py --xml dlib_data/training_data.xml --output vis_results

# 最初の10枚のみ表示
python visualize_dlib_data.py --xml dlib_data/training_data.xml --show --samples 10
```

### 3. 特徴点検出器の学習

dlib検出器を学習：

```bash
# 基本的な学習
python train_dlib_detector.py --xml dlib_data/training_data.xml --output nir_shape_predictor.dat

# パラメータを調整した学習
python train_dlib_detector.py --xml dlib_data/training_data.xml --output model.dat \
    --tree-depth 4 --cascade-depth 15 --nu 0.1 --feature-pool-size 500
```

**学習パラメータの説明：**
- `tree-depth`: 回帰木の深さ（2-5推奨）
- `cascade-depth`: カスケード数（10-20推奨）
- `nu`: 学習率（0.01-0.25）
- `feature-pool-size`: 特徴プールサイズ（400-1000）

### 4. 特徴点の予測

学習済みモデルでテストデータを予測：

```bash
python predict_landmarks.py --model nir_shape_predictor.dat \
    --input datasetB/images --output datasetB_predictions
```

### 5. 結果の確認

予測結果は以下に保存されます：
- `datasetB_predictions/landmarks/`: 予測座標（.npyファイル）
- `datasetB_predictions/visualizations/`: 可視化画像
- `datasetB_predictions/detection_report.json`: 検出統計

### 6. 個別画像のテスト

```bash
python test_nir_detector.py --model nir_shape_predictor.dat --image test_image.png
```

## 特徴点について

本プロジェクトは68点顔特徴点を検出します：

- **輪郭 (0-16)**: 顔の外輪郭
- **眉毛 (17-26)**: 左右の眉毛
- **鼻 (27-35)**: 鼻の形状
- **目 (36-47)**: 左右の目
- **口 (48-67)**: 口の形状

## データ形式

### 入力データ形式
- 画像: PNG, JPG, BMP形式
- ランドマーク: テキストファイル（座標ペア）

### 出力データ形式
- 予測座標: NumPy配列（.npy）
- 可視化: PNG画像
- レポート: JSON形式

## トラブルシューティング

### dlibのインストールエラー
```bash
# コンパイル済みパッケージを使用
conda install -c conda-forge dlib
# または
pip install dlib-binary
```

### メモリ不足エラー
- 学習パラメータを調整（cascade-depthやfeature-pool-sizeを減らす）
- バッチサイズを小さくする

### 顔検出失敗
- 画像の前処理を確認
- 顔検出器のパラメータ調整を検討

## 開発情報

### ファイル説明

- `make_landmarks.py`: カスタム形式からdlib形式への変換
- `train_dlib_detector.py`: dlib学習メインスクリプト
- `predict_landmarks.py`: 学習済みモデルでの予測
- `visualize_dlib_data.py`: 学習データの可視化
- `test_nir_detector.py`: 個別画像のテスト

### 拡張可能性

- 特徴点数の変更（68点以外）
- カスタム前処理の追加
- 異なる検出アルゴリズムとの比較


## 参考資料

- [dlib公式ドキュメント](http://dlib.net/)
- [顔特徴点検出について](http://dlib.net/face_landmark_detection.py.html)
- [近赤外線画像処理](https://docs.opencv.org/master/)
