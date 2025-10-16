"""
学習済みdlibモデルでdatasetBの画像に特徴点を推定するスクリプト

使用方法:
    python predict_landmarks.py --model nir_shape_predictor.dat --input datasetB/images --output datasetB_predictions
"""

import dlib
import cv2
import numpy as np
import os
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

def load_predictor_and_detector(model_path):
    """
    学習済みモとデル顔検出器を読み込み
    
    Args:
        model_path (str): 学習済み特徴点検出器のパス
        
    Returns:
        tuple: (face_detector, shape_predictor)
    """
    # HOG顔検出器（dlibの標準）
    face_detector = dlib.get_frontal_face_detector()
    
    # 学習済み特徴点検出器
    shape_predictor = dlib.shape_predictor(model_path)
    
    return face_detector, shape_predictor

def detect_landmarks(image_path, face_detector, shape_predictor):
    """
    画像から特徴点を検出
    
    Args:
        image_path (str): 画像ファイルパス
        face_detector: 顔検出器
        shape_predictor: 特徴点検出器
        
    Returns:
        tuple: (landmarks, face_rect, success)
    """
    try:
        # 画像読み込み
        image = cv2.imread(str(image_path))
        if image is None:
            return None, None, False
            
        # グレースケール変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 顔検出
        faces = face_detector(gray)
        
        if len(faces) == 0:
            print(f"顔が検出されませんでした: {image_path}")
            return None, None, False
        
        # 最大の顔を選択
        face_rect = max(faces, key=lambda rect: rect.width() * rect.height())
        
        # 特徴点検出
        landmarks = shape_predictor(gray, face_rect)
        
        # 座標を配列に変換
        points = np.array([[landmarks.part(i).x, landmarks.part(i).y] 
                          for i in range(landmarks.num_parts)])
        
        return points, face_rect, True
        
    except Exception as e:
        print(f"エラー: {image_path} - {e}")
        return None, None, False

def save_results(landmarks, image_path, output_dir, face_rect=None):
    """
    結果を保存
    
    Args:
        landmarks (np.array): 特徴点座標
        image_path (str): 元画像パス
        output_dir (Path): 出力ディレクトリ
        face_rect: 顔の矩形領域
    """
    image_name = Path(image_path).stem
    
    # ランドマーク座標をnpyで保存
    npy_path = output_dir / 'landmarks' / f'{image_name}.npy'
    np.save(npy_path, landmarks)
    
    # 可視化画像を保存
    img = Image.open(image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c='red', s=20)
    
    # 顔の矩形も描画
    if face_rect:
        rect = plt.Rectangle((face_rect.left(), face_rect.top()), 
                           face_rect.width(), face_rect.height(),
                           fill=False, color='blue', linewidth=2)
        plt.gca().add_patch(rect)
    
    plt.title(f'Predicted Landmarks: {image_name}')
    plt.axis('off')
    
    vis_path = output_dir / 'visualizations' / f'{image_name}_landmarks.png'
    plt.savefig(vis_path, bbox_inches='tight', dpi=150)
    plt.close()

def process_dataset(input_dir, model_path, output_dir):
    """
    データセット全体を処理
    
    Args:
        input_dir (str): 入力画像ディレクトリ
        model_path (str): 学習済みモデルパス
        output_dir (str): 出力ディレクトリ
    """
    
    print(f"学習済みモデル: {model_path}")
    print(f"入力ディレクトリ: {input_dir}")
    print(f"出力ディレクトリ: {output_dir}")
    
    # 出力ディレクトリ作成
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'landmarks').mkdir(exist_ok=True)
    (output_dir / 'visualizations').mkdir(exist_ok=True)
    
    # モデル読み込み
    print("モデルを読み込み中...")
    face_detector, shape_predictor = load_predictor_and_detector(model_path)
    
    # 画像ファイル検索
    input_dir = Path(input_dir)
    image_files = []
    for ext in ['*.png', '*.jpg', '*.bmp']:
        image_files.extend(input_dir.glob(ext))
    
    print(f"見つかった画像ファイル: {len(image_files)}枚")
    
    # 処理結果記録用
    results = {
        'total_images': len(image_files),
        'successful_detections': 0,
        'failed_detections': 0,
        'failed_files': []
    }
    
    # 各画像を処理
    print("特徴点検出を開始...")
    for image_path in tqdm(image_files, desc="特徴点検出中"):
        landmarks, face_rect, success = detect_landmarks(
            image_path, face_detector, shape_predictor
        )
        
        if success:
            save_results(landmarks, image_path, output_dir, face_rect)
            results['successful_detections'] += 1
        else:
            results['failed_detections'] += 1
            results['failed_files'].append(str(image_path))
    
    # 結果レポート保存
    report_path = output_dir / 'detection_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 結果表示
    print("\n=== 処理完了 ===")
    print(f"総画像数: {results['total_images']}")
    print(f"成功: {results['successful_detections']}枚")
    print(f"失敗: {results['failed_detections']}枚")
    print(f"成功率: {results['successful_detections']/results['total_images']*100:.1f}%")
    print(f"結果: {output_dir}")
    print(f"レポート: {report_path}")

def main():
    parser = argparse.ArgumentParser(
        description='学習済みdlibモデルで特徴点を検出',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    python predict_landmarks.py --model nir_shape_predictor.dat --input datasetB/images --output datasetB_predictions
        """
    )
    
    parser.add_argument('--model', '-m', required=True,
                       help='学習済み特徴点検出器のパス(.dat)')
    parser.add_argument('--input', '-i', required=True,
                       help='入力画像ディレクトリ')
    parser.add_argument('--output', '-o', required=True,
                       help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    # ファイル存在確認
    if not os.path.exists(args.model):
        print(f"エラー: モデルファイルが見つかりません: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.input):
        print(f"エラー: 入力ディレクトリが見つかりません: {args.input}")
        sys.exit(1)
    
    try:
        process_dataset(args.input, args.model, args.output)
        
    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()