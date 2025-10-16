#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
datasetフォルダ（images + landmarks）からdlib特徴点検出器学習用XMLファイルを作成するスクリプト

ディレクトリ構造:
    dataset/
    ├── images/       - 画像ファイル (.png, .jpg, .bmp)
    └── landmarks/    - 特徴点座標ファイル (.npy)

使用方法:
    # 顔検出器を使用（推奨）
    python create_dlib_xml.py --input dataset --output dlib_data/training_data.xml --use-detector
    
    # ランドマークから自動計算
    python create_dlib_xml.py --input datasetA --output dlib_data/training_data.xml --margin 0.25
"""

import numpy as np
import cv2
import dlib
import argparse
import sys
import shutil
from pathlib import Path
from tqdm import tqdm
import xml.etree.ElementTree as ET
from xml.dom import minidom


def load_landmark_files(dataset_dir):
    """
    ランドマークファイルと対応する画像ファイルのペアを取得
    
    Args:
        dataset_dir (Path): データセットのルートディレクトリ
        
    Returns:
        list: (image_path, landmark_path)のタプルのリスト
    """
    images_dir = dataset_dir / 'images'
    landmarks_dir = dataset_dir / 'landmarks'
    
    if not images_dir.exists():
        raise FileNotFoundError(f"画像ディレクトリが見つかりません: {images_dir}")
    
    if not landmarks_dir.exists():
        raise FileNotFoundError(f"ランドマークディレクトリが見つかりません: {landmarks_dir}")
    
    # ランドマークファイルを取得
    landmark_files = list(landmarks_dir.glob('*.npy'))
    
    print(f"ランドマークファイル数: {len(landmark_files)}")
    
    # 対応する画像ファイルを探す
    pairs = []
    missing_images = []
    
    for landmark_path in landmark_files:
        stem = landmark_path.stem
        
        # 対応する画像を探す（複数の拡張子に対応）
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP']:
            candidate = images_dir / f'{stem}{ext}'
            if candidate.exists():
                image_path = candidate
                break
        
        if image_path:
            pairs.append((image_path, landmark_path))
        else:
            missing_images.append(stem)
    
    if missing_images:
        print(f"警告: {len(missing_images)}個のランドマークに対応する画像が見つかりません")
        if len(missing_images) <= 10:
            for stem in missing_images:
                print(f"  - {stem}")
    
    print(f"有効なペア数: {len(pairs)}")
    
    return pairs


def calculate_bounding_box(landmarks, margin=0.2):
    """
    ランドマーク座標からバウンディングボックスを計算
    
    Args:
        landmarks (np.array): (N, 2)の特徴点座標
        margin (float): マージンの割合（0.2 = 20%）
        
    Returns:
        tuple: (left, top, width, height)
    """
    # ランドマークの最小・最大座標を取得
    x_coords = landmarks[:, 0]
    y_coords = landmarks[:, 1]
    
    min_x = int(x_coords.min())
    max_x = int(x_coords.max())
    min_y = int(y_coords.min())
    max_y = int(y_coords.max())
    
    # 元の幅と高さ
    width = max_x - min_x
    height = max_y - min_y
    
    # マージンを追加
    margin_x = int(width * margin)
    margin_y = int(height * margin)
    
    left = max(0, min_x - margin_x)
    top = max(0, min_y - margin_y)
    width = width + 2 * margin_x
    height = height + 2 * margin_y
    
    return left, top, width, height


def detect_face_bounding_box(image, landmarks=None):
    """
    顔検出器を使ってバウンディングボックスを検出
    
    Args:
        image: 入力画像（OpenCV形式）
        landmarks (np.array): 特徴点座標（オプション、検証用）
        
    Returns:
        tuple: (left, top, width, height) または None（検出失敗時）
    """
    # dlib HOG顔検出器を使用
    detector = dlib.get_frontal_face_detector()
    
    # グレースケール変換
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 顔検出
    faces = detector(gray, 1)  # upsample=1
    
    if len(faces) == 0:
        return None
    
    # 複数の顔が検出された場合、ランドマークに最も近い顔を選択
    if len(faces) > 1 and landmarks is not None:
        # ランドマークの中心を計算
        landmark_center_x = landmarks[:, 0].mean()
        landmark_center_y = landmarks[:, 1].mean()
        
        # 最も近い顔を選択
        best_face = None
        min_distance = float('inf')
        
        for face in faces:
            face_center_x = (face.left() + face.right()) / 2
            face_center_y = (face.top() + face.bottom()) / 2
            
            distance = np.sqrt((face_center_x - landmark_center_x)**2 + 
                             (face_center_y - landmark_center_y)**2)
            
            if distance < min_distance:
                min_distance = distance
                best_face = face
        
        face = best_face
    else:
        # 最大の顔を選択
        face = max(faces, key=lambda rect: rect.width() * rect.height())
    
    # dlib rectangle から (left, top, width, height) に変換
    left = face.left()
    top = face.top()
    width = face.width()
    height = face.height()
    
    return left, top, width, height


def validate_landmarks(landmarks, image_shape):
    """
    ランドマークが画像範囲内にあるか確認
    
    Args:
        landmarks (np.array): (N, 2)の特徴点座標
        image_shape (tuple): (height, width)
        
    Returns:
        bool: すべての点が画像内ならTrue
    """
    height, width = image_shape[:2]
    
    x_coords = landmarks[:, 0]
    y_coords = landmarks[:, 1]
    
    # すべての点が画像範囲内か確認
    x_valid = np.all((x_coords >= 0) & (x_coords < width))
    y_valid = np.all((y_coords >= 0) & (y_coords < height))
    
    return x_valid and y_valid


def create_dlib_xml(pairs, output_path, dataset_root, margin=0.2, 
                   validate=True, max_samples=None, use_detector=False):
    """
    dlib特徴点検出器学習用XMLファイルを作成
    
    Args:
        pairs (list): (image_path, landmark_path)のリスト
        output_path (str): 出力XMLファイルパス
        dataset_root (Path): データセットのルートディレクトリ
        margin (float): バウンディングボックスのマージン（use_detector=False時のみ）
        validate (bool): ランドマークの妥当性を検証するか
        max_samples (int): 最大サンプル数（Noneなら全て）
        use_detector (bool): 顔検出器を使ってバウンディングボックスを検出するか
    """
    
    print("\n=== dlib 特徴点検出器学習用XML作成開始 ===")
    print(f"総ペア数: {len(pairs)}")
    if use_detector:
        print(f"バウンディングボックス: 顔検出器を使用")
    else:
        print(f"バウンディングボックス: ランドマークから計算（マージン {margin*100:.0f}%）")
    print(f"出力ファイル: {output_path}")
    
    # 出力ディレクトリを作成
    output_path = Path(output_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 画像出力ディレクトリを作成
    images_output_dir = output_dir / 'images'
    images_output_dir.mkdir(exist_ok=True)
    print(f"画像出力ディレクトリ: {images_output_dir}")
    
    # サンプル数制限
    if max_samples and max_samples < len(pairs):
        pairs = pairs[:max_samples]
        print(f"サンプル数を{max_samples}に制限")
    
    # XML構造を作成
    dataset_elem = ET.Element('dataset')
    name_elem = ET.SubElement(dataset_elem, 'name')
    name_elem.text = 'NIR Face Landmarks Training Dataset'
    
    comment_elem = ET.SubElement(dataset_elem, 'comment')
    comment_elem.text = 'Custom face landmark dataset for near-infrared images'
    
    images_elem = ET.SubElement(dataset_elem, 'images')
    
    # 統計情報
    stats = {
        'total': len(pairs),
        'success': 0,
        'failed': 0,
        'invalid_landmarks': 0,
        'load_errors': 0,
        'face_detection_failed': 0
    }
    
    # 各画像とランドマークを処理
    print("\nXMLデータを構築中...")
    for img_path, landmark_path in tqdm(pairs, desc='処理中'):
        try:
            # ランドマーク読み込み
            landmarks = np.load(landmark_path)
            
            # 形状チェック
            if landmarks.shape[1] != 2:
                print(f"\n警告: ランドマーク形状が不正: {landmark_path} ({landmarks.shape})")
                stats['invalid_landmarks'] += 1
                stats['failed'] += 1
                continue
            
            # 画像読み込み
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"\n警告: 画像の読み込みに失敗: {img_path}")
                stats['load_errors'] += 1
                stats['failed'] += 1
                continue
            
            # ランドマークの妥当性チェック
            if validate and not validate_landmarks(landmarks, image.shape):
                print(f"\n警告: ランドマークが画像範囲外: {img_path}")
                stats['invalid_landmarks'] += 1
                stats['failed'] += 1
                continue
            
            # バウンディングボックス取得
            if use_detector:
                # 顔検出器を使用
                bbox_result = detect_face_bounding_box(image, landmarks)
                if bbox_result is None:
                    print(f"\n警告: 顔が検出されませんでした: {img_path}")
                    stats['face_detection_failed'] += 1
                    stats['failed'] += 1
                    continue
                left, top, width, height = bbox_result
            else:
                # ランドマークから計算
                left, top, width, height = calculate_bounding_box(landmarks, margin)
            
            # 画像を出力ディレクトリにコピー
            image_filename = img_path.name
            output_image_path = images_output_dir / image_filename
            if not output_image_path.exists():
                shutil.copy2(img_path, output_image_path)
            
            # XMLに相対パスで画像を追加（XMLファイルの場所からの相対パス）
            # XMLファイルと同じディレクトリからの相対パスとして images/filename.ext の形式にする
            relative_path = f'images/{image_filename}'
            image_elem = ET.SubElement(images_elem, 'image', file=relative_path)
            
            # バウンディングボックスを追加
            box_elem = ET.SubElement(image_elem, 'box',
                                    top=str(top),
                                    left=str(left),
                                    width=str(width),
                                    height=str(height))
            
            # 各特徴点を追加
            for i, (x, y) in enumerate(landmarks):
                part_elem = ET.SubElement(box_elem, 'part',
                                         name=str(i),
                                         x=str(int(x)),
                                         y=str(int(y)))
            
            stats['success'] += 1
            
        except Exception as e:
            print(f"\nエラー: {img_path} - {e}")
            stats['failed'] += 1
            stats['load_errors'] += 1
            continue
    
    # XMLを整形して保存
    print("\nXMLファイルを保存中...")
    xml_str = ET.tostring(dataset_elem, encoding='unicode')
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent='  ')
    
    # ファイル保存
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)
    
    # 結果表示
    print("\n=== XML作成完了 ===")
    print(f"総ペア数: {stats['total']}")
    print(f"成功: {stats['success']}")
    print(f"失敗: {stats['failed']}")
    if stats['invalid_landmarks'] > 0:
        print(f"  - 不正なランドマーク: {stats['invalid_landmarks']}")
    if stats['load_errors'] > 0:
        print(f"  - 読み込みエラー: {stats['load_errors']}")
    if stats['face_detection_failed'] > 0:
        print(f"  - 顔検出失敗: {stats['face_detection_failed']}")
    print(f"成功率: {stats['success']/stats['total']*100:.1f}%")
    print(f"\n出力ファイル: {output_path}")
    print(f"画像ディレクトリ: {images_output_dir}")
    print(f"コピーされた画像: {len(list(images_output_dir.glob('*')))}枚")
    
    # ファイルサイズ表示
    file_size = output_path.stat().st_size / 1024 / 1024
    print(f"XMLファイルサイズ: {file_size:.2f} MB")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='datasetフォルダからdlib特徴点検出器学習用XMLファイルを作成',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ディレクトリ構造:
    dataset/
    ├── images/       - 画像ファイル (.png, .jpg, .bmp)
    └── landmarks/    - 特徴点座標ファイル (.npy)

使用例:
    # 顔検出器を使用（推奨：より正確なバウンディングボックス）
    python create_dlib_xml.py --input datasetA --output dlib_data/training_data.xml --use-detector
    
    # ランドマークから計算（顔検出器が上手く動かない場合）
    python create_dlib_xml.py --input datasetA --output dlib_data/training_data.xml --margin 0.25
    
    # サンプル数を制限（テスト用）
    python create_dlib_xml.py --input datasetA --output test.xml --use-detector --samples 100
    
    # 検証をスキップ（高速化）
    python create_dlib_xml.py --input datasetA --output train.xml --use-detector --no-validate

注意:
    このスクリプトは特徴点検出器（Shape Predictor）の学習用XMLを作成します。
    バウンディングボックスは学習時に「どの領域で特徴点を学習するか」を指定するためのものです。
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='入力データセットディレクトリ（images/とlandmarks/を含む）')
    parser.add_argument('--output', '-o', required=True,
                       help='出力XMLファイルパス')
    parser.add_argument('--use-detector', '-d', action='store_true',
                       help='顔検出器を使用してバウンディングボックスを検出（推奨）')
    parser.add_argument('--margin', '-m', type=float, default=0.2,
                       help='バウンディングボックスのマージン（--use-detectorなしの場合のみ、デフォルト: 0.2 = 20%%）')
    parser.add_argument('--samples', '-s', type=int,
                       help='最大サンプル数（指定しない場合は全て）')
    parser.add_argument('--no-validate', action='store_true',
                       help='ランドマークの妥当性検証をスキップ（高速化）')
    
    args = parser.parse_args()
    
    # パス処理
    dataset_root = Path(args.input).resolve()
    
    if not dataset_root.exists():
        print(f"エラー: データセットディレクトリが見つかりません: {dataset_root}")
        sys.exit(1)
    
    try:
        # ランドマークと画像のペアを取得
        print("=== データセット読み込み ===")
        print(f"データセット: {dataset_root}")
        pairs = load_landmark_files(dataset_root)
        
        if len(pairs) == 0:
            print("エラー: 有効な画像とランドマークのペアが見つかりませんでした")
            sys.exit(1)
        
        # XMLファイル作成
        stats = create_dlib_xml(
            pairs=pairs,
            output_path=args.output,
            dataset_root=dataset_root,
            margin=args.margin,
            validate=not args.no_validate,
            max_samples=args.samples,
            use_detector=args.use_detector
        )
        
        if stats['success'] == 0:
            print("\nエラー: 有効なデータが1つもありませんでした")
            sys.exit(1)
        
        print(f"\n次のコマンドでXMLを可視化できます:")
        print(f"python visualize_dlib_data.py --xml {args.output} --output vis_results --samples 10")
        
        print(f"\n次のコマンドでモデルを学習できます:")
        print(f"python train_dlib_detector.py --xml {args.output} --output trained_model.dat")
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

