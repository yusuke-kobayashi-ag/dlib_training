#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学習済み近赤外線特徴点検出器のテストスクリプト

学習したdlibモデルで近赤外線画像の特徴点検出をテストします。

使用方法:
    python test_nir_detector.py --model nir_shape_predictor.dat --image test_image.png
    python test_nir_detector.py --model model.dat --image test.png --output result.png
"""

import dlib
import cv2
import numpy as np
import argparse
import sys
import os
from pathlib import Path



def load_model_and_detector(model_path):
    """モデルと顔検出器を読み込み"""
    try:
        # 顔検出器（dlibの標準HOG検出器）
        face_detector = dlib.get_frontal_face_detector()
        
        # 学習済み特徴点検出器
        shape_predictor = dlib.shape_predictor(model_path)
        
        print(f"✓ モデル読み込み成功: {model_path}")
        return face_detector, shape_predictor
        
    except Exception as e:
        raise RuntimeError(f"モデル読み込みエラー: {e}")


def detect_landmarks(image, face_detector, shape_predictor):
    """
    画像から顔と特徴点を検出
    
    Args:
        image: 入力画像（グレースケール）
        face_detector: dlib顔検出器
        shape_predictor: dlib特徴点検出器
        
    Returns:
        list: 検出された顔と特徴点のリスト
    """
    
    # 顔検出
    faces = face_detector(image)
    print(f"検出された顔の数: {len(faces)}")
    
    results = []
    
    for i, face in enumerate(faces):
        print(f"顔 {i+1}: ({face.left()}, {face.top()}) - ({face.right()}, {face.bottom()})")
        
        # 特徴点検出
        landmarks = shape_predictor(image, face)
        
        # 特徴点座標を配列に変換
        points = []
        for j in range(landmarks.num_parts):
            x = landmarks.part(j).x
            y = landmarks.part(j).y
            points.append((x, y))
        
        results.append({
            'face_rect': face,
            'landmarks': landmarks,
            'points': points
        })
    
    return results


def draw_landmarks(image, results, draw_face_box=True, draw_point_numbers=False):
    """
    画像に検出結果を描画
    
    Args:
        image: 入力画像
        results: 検出結果
        draw_face_box: 顔のバウンディングボックスを描画するか
        draw_point_numbers: 特徴点番号を描画するか
        
    Returns:
        np.ndarray: 描画済み画像
    """
    
    # カラー画像に変換（グレースケールの場合）
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = image.copy()
    
    colors = [
        (0, 255, 0),    # 緑
        (255, 0, 0),    # 青  
        (0, 0, 255),    # 赤
        (255, 255, 0),  # シアン
        (255, 0, 255),  # マゼンタ
    ]
    
    for i, result in enumerate(results):
        color = colors[i % len(colors)]
        face = result['face_rect']
        points = result['points']
        
        # 顔のバウンディングボックス描画
        if draw_face_box:
            cv2.rectangle(vis_image, 
                         (face.left(), face.top()), 
                         (face.right(), face.bottom()), 
                         color, 2)
            cv2.putText(vis_image, f'Face {i+1}', 
                       (face.left(), face.top()-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 特徴点描画
        for j, (x, y) in enumerate(points):
            cv2.circle(vis_image, (x, y), 2, color, -1)
            
            # 特徴点番号描画
            if draw_point_numbers:
                cv2.putText(vis_image, str(j), 
                           (x+3, y-3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    return vis_image


def analyze_landmarks(results):
    """特徴点の分析情報を表示"""
    print("\n=== 特徴点分析 ===")
    
    for i, result in enumerate(results):
        points = np.array(result['points'])
        face = result['face_rect']
        
        print(f"\n--- 顔 {i+1} ---")
        print(f"顔サイズ: {face.width()} x {face.height()}")
        
        # 特徴点の範囲
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)
        print(f"特徴点範囲: ({min_x}, {min_y}) - ({max_x}, {max_y})")
        
        # 主要な特徴点（例）
        print(f"鼻先 (点33): ({points[33][0]}, {points[33][1]})")
        print(f"左目中心 (点36-41平均): ({points[36:42, 0].mean():.1f}, {points[36:42, 1].mean():.1f})")
        print(f"右目中心 (点42-47平均): ({points[42:48, 0].mean():.1f}, {points[42:48, 1].mean():.1f})")
        print(f"口中心 (点48-67平均): ({points[48:68, 0].mean():.1f}, {points[48:68, 1].mean():.1f})")


def main():
    parser = argparse.ArgumentParser(
        description='学習済みdlibモデルで近赤外線画像の特徴点検出をテスト',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    python test_nir_detector.py --model nir_shape_predictor.dat --image test.png
    python test_nir_detector.py --model model.dat --image test.png --output result.png --show
        """
    )
    
    parser.add_argument('--model', '-m', required=True,
                       help='学習済みdlibモデルファイル (.dat)')
    parser.add_argument('--image', '-i', required=True,
                       help='テスト用画像ファイル')
    parser.add_argument('--output', '-o',
                       help='結果画像の保存先（指定しない場合は保存しない）')
    parser.add_argument('--show', action='store_true',
                       help='結果をウィンドウで表示')
    parser.add_argument('--draw-numbers', action='store_true',
                       help='特徴点番号を描画')
    parser.add_argument('--no-face-box', action='store_true',
                       help='顔のバウンディングボックスを描画しない')
    
    args = parser.parse_args()
    
    # ファイル存在確認
    if not os.path.exists(args.model):
        print(f"エラー: モデルファイルが見つかりません: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.image):
        print(f"エラー: 画像ファイルが見つかりません: {args.image}")
        sys.exit(1)
    
    try:
        # モデル読み込み
        face_detector, shape_predictor = load_model_and_detector(args.model)
        
        # 画像読み込み
        print(f"画像読み込み: {args.image}")
        image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("画像の読み込みに失敗しました")
        
        print(f"画像サイズ: {image.shape[1]} x {image.shape[0]}")
        
        # 特徴点検出
        print("\n特徴点検出を実行中...")
        results = detect_landmarks(image, face_detector, shape_predictor)
        
        if len(results) == 0:
            print("⚠️  顔が検出されませんでした")
            print("ヒント: 画像の明度やコントラストを調整してみてください")
            sys.exit(0)
        
        # 分析情報表示
        analyze_landmarks(results)
        
        # 結果描画
        vis_image = draw_landmarks(
            image, 
            results, 
            draw_face_box=not args.no_face_box,
            draw_point_numbers=args.draw_numbers
        )
        
        # 結果保存
        if args.output:
            cv2.imwrite(args.output, vis_image)
            print(f"\n結果画像を保存: {args.output}")
        
        # 結果表示
        if args.show:
            cv2.imshow('NIR Landmark Detection Result', vis_image)
            print("\nESCキーまたは任意のキーで終了")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        print("\n特徴点検出テスト完了!")
        
    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
