#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dlib学習用データの可視化スクリプト

変換されたdlib形式のデータ（画像+XML）を読み込んで、
バウンディングボックスと68点特徴点を可視化します。

使用方法:
    python visualize_dlib_data.py --xml training_data.xml --output vis_results
    python visualize_dlib_data.py --xml training_data.xml --show --samples 10
"""

import cv2
import numpy as np
import xml.etree.ElementTree as ET
import argparse
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
try:
    from matplotlib.colors import get_cmap
except ImportError:
    # matplotlib 3.7以降の場合
    from matplotlib import colormaps
    def get_cmap(name):
        return colormaps[name]


def parse_dlib_xml(xml_path):
    """
    dlibのXMLファイルを解析
    
    Args:
        xml_path (str): XMLファイルパス
        
    Returns:
        list: 画像情報のリスト
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        images_data = []
        
        for image_elem in root.find('images').findall('image'):
            img_file = image_elem.get('file')
            
            # 各画像の顔情報を取得
            faces = []
            for box_elem in image_elem.findall('box'):
                box_info = {
                    'top': int(box_elem.get('top')),
                    'left': int(box_elem.get('left')),
                    'width': int(box_elem.get('width')),
                    'height': int(box_elem.get('height')),
                    'landmarks': []
                }
                
                # 特徴点を取得
                for part_elem in box_elem.findall('part'):
                    point = {
                        'name': part_elem.get('name'),
                        'x': int(part_elem.get('x')),
                        'y': int(part_elem.get('y'))
                    }
                    box_info['landmarks'].append(point)
                
                faces.append(box_info)
            
            images_data.append({
                'file': img_file,
                'faces': faces
            })
        
        return images_data
        
    except Exception as e:
        raise ValueError(f"XMLファイルの解析に失敗しました: {e}")


def draw_landmarks_opencv(image, face_data, colors=None):
    """
    OpenCVを使って特徴点とバウンディングボックスを描画
    
    Args:
        image: 入力画像
        face_data: 顔データ
        colors: 色リスト
        
    Returns:
        描画済み画像
    """
    if colors is None:
        colors = [
            (0, 255, 0),    # 緑
            (255, 0, 0),    # 青
            (0, 0, 255),    # 赤
            (255, 255, 0),  # シアン
            (255, 0, 255),  # マゼンタ
        ]
    
    # カラー画像に変換
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = image.copy()
    
    for i, face in enumerate(face_data['faces']):
        color = colors[i % len(colors)]
        
        # バウンディングボックス描画
        top, left = face['top'], face['left']
        width, height = face['width'], face['height']
        
        cv2.rectangle(vis_image, 
                     (left, top), 
                     (left + width, top + height), 
                     color, 2)
        
        cv2.putText(vis_image, f'Face {i+1}', 
                   (left, top - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 特徴点描画
        landmarks = face['landmarks']
        for j, point in enumerate(landmarks):
            x, y = point['x'], point['y']
            
            # 特徴点の種類に応じて色を変える
            if j < 17:  # 輪郭
                point_color = (0, 255, 255)  # イエロー
            elif j < 27:  # 眉毛
                point_color = (255, 0, 255)  # マゼンタ
            elif j < 36:  # 鼻
                point_color = (0, 255, 0)    # 緑
            elif j < 48:  # 目
                point_color = (255, 0, 0)    # 青
            else:  # 口
                point_color = (0, 0, 255)    # 赤
            
            cv2.circle(vis_image, (x, y), 2, point_color, -1)
            
            # 特徴点番号を描画（小さく）
            cv2.putText(vis_image, str(j), 
                       (x + 3, y - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, point_color, 1)
    
    return vis_image


def draw_landmarks_matplotlib(image, face_data, save_path=None, show=False):
    """
    Matplotlibを使って特徴点とバウンディングボックスを描画
    
    Args:
        image: 入力画像
        face_data: 顔データ
        save_path: 保存パス
        show: 表示するかどうか
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # 画像表示
    if len(image.shape) == 2:
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # 特徴点の色分け用
    colors = ['yellow', 'magenta', 'green', 'blue', 'red']
    point_colors = []
    
    for i in range(68):
        if i < 17:      # 輪郭
            point_colors.append('yellow')
        elif i < 27:    # 眉毛  
            point_colors.append('magenta')
        elif i < 36:    # 鼻
            point_colors.append('green')
        elif i < 48:    # 目
            point_colors.append('blue')
        else:           # 口
            point_colors.append('red')
    
    for i, face in enumerate(face_data['faces']):
        # バウンディングボックス描画
        top, left = face['top'], face['left']
        width, height = face['width'], face['height']
        
        rect = patches.Rectangle((left, top), width, height, 
                               linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        
        ax.text(left, top - 10, f'Face {i+1}', 
               color='lime', fontsize=12, fontweight='bold')
        
        # 特徴点描画
        landmarks = face['landmarks']
        for j, point in enumerate(landmarks):
            x, y = point['x'], point['y']
            
            ax.plot(x, y, 'o', color=point_colors[j], markersize=4)
            ax.text(x + 2, y - 2, str(j), fontsize=8, color=point_colors[j])
    
    ax.set_title(f'NIR Face Landmarks: {face_data["file"]}', fontsize=14)
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_legend_image():
    """特徴点の色分け凡例を作成"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    legend_data = [
        ('輪郭 (0-16)', 'yellow'),
        ('眉毛 (17-26)', 'magenta'), 
        ('鼻 (27-35)', 'green'),
        ('目 (36-47)', 'blue'),
        ('口 (48-67)', 'red')
    ]
    
    y_pos = 0.8
    for label, color in legend_data:
        ax.plot(0.1, y_pos, 'o', color=color, markersize=10)
        ax.text(0.2, y_pos, label, fontsize=14, va='center')
        y_pos -= 0.15
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('68点特徴点の色分け', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    return fig


def visualize_dlib_dataset(xml_path, output_dir=None, max_samples=None, 
                          show_images=False, use_matplotlib=True):
    """
    dlib学習用データセットを可視化
    
    Args:
        xml_path (str): XMLファイルパス
        output_dir (str): 出力ディレクトリ（None の場合は保存しない）
        max_samples (int): 最大サンプル数
        show_images (bool): 画像を表示するか
        use_matplotlib (bool): Matplotlibを使うか（False の場合はOpenCV）
    """
    
    print("=== dlib学習データ可視化開始 ===")
    
    # XMLファイル解析
    print("XMLファイルを解析中...")
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"XMLファイルが見つかりません: {xml_path}")
    
    images_data = parse_dlib_xml(xml_path)
    print(f"総画像数: {len(images_data)}")
    
    # サンプル数制限
    if max_samples and max_samples < len(images_data):
        images_data = images_data[:max_samples]
        print(f"サンプル数を{max_samples}に制限")
    
    # 出力ディレクトリ作成
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        print(f"出力ディレクトリ: {output_dir}")
        
        # 凡例画像作成
        if use_matplotlib:
            legend_fig = create_legend_image()
            legend_fig.savefig(output_dir / 'landmark_legend.png', 
                             dpi=150, bbox_inches='tight')
            plt.close(legend_fig)
            print("凡例画像を作成: landmark_legend.png")
    
    # 各画像を処理
    base_dir = xml_path.parent
    processed_count = 0
    
    for i, img_data in enumerate(images_data):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"処理中: {i + 1}/{len(images_data)}")
        
        # 画像読み込み
        img_path = base_dir / img_data['file']
        if not img_path.exists():
            print(f"警告: 画像ファイルが見つかりません: {img_path}")
            continue
        
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"警告: 画像の読み込みに失敗: {img_path}")
            continue
        
        # 顔が検出されているかチェック
        if not img_data['faces']:
            print(f"警告: 顔データがありません: {img_data['file']}")
            continue
        
        # 可視化
        if use_matplotlib:
            save_path = None
            if output_dir:
                save_path = output_dir / f'vis_{i:04d}_{Path(img_data["file"]).stem}.png'
            
            draw_landmarks_matplotlib(image, img_data, save_path, show_images)
        else:
            # OpenCV版
            vis_image = draw_landmarks_opencv(image, img_data)
            
            if output_dir:
                save_path = output_dir / f'vis_{i:04d}_{Path(img_data["file"]).stem}.png'
                cv2.imwrite(str(save_path), vis_image)
            
            if show_images:
                cv2.imshow(f'Visualization {i+1}', vis_image)
                key = cv2.waitKey(0) & 0xFF
                if key == 27:  # ESC
                    break
                cv2.destroyAllWindows()
        
        processed_count += 1
    
    if not use_matplotlib and show_images:
        cv2.destroyAllWindows()
    
    print(f"\n=== 可視化完了 ===")
    print(f"処理済み画像数: {processed_count}")
    if output_dir:
        print(f"結果保存先: {output_dir}")
        print(f"生成ファイル数: {len(list(output_dir.glob('vis_*.png')))}")


def main():
    parser = argparse.ArgumentParser(
        description='dlib学習用データの可視化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    # 全画像を可視化して保存
    python visualize_dlib_data.py --xml training_data.xml --output vis_results
    
    # 最初の10枚のみ表示
    python visualize_dlib_data.py --xml training_data.xml --show --samples 10
    
    # OpenCVで可視化（軽量版）
    python visualize_dlib_data.py --xml training_data.xml --output vis_results --opencv
        """
    )
    
    parser.add_argument('--xml', '-x', required=True,
                       help='dlib学習用XMLファイル')
    parser.add_argument('--output', '-o',
                       help='可視化結果の保存ディレクトリ')
    parser.add_argument('--samples', '-s', type=int,
                       help='可視化するサンプル数の上限')
    parser.add_argument('--show', action='store_true',
                       help='画像をウィンドウで表示')
    parser.add_argument('--opencv', action='store_true',
                       help='OpenCVを使用（Matplotlibの代わり）')
    
    args = parser.parse_args()
    
    # 引数チェック
    if not args.output and not args.show:
        print("エラー: --output または --show のいずれかを指定してください")
        sys.exit(1)
    
    if not os.path.exists(args.xml):
        print(f"エラー: XMLファイルが見つかりません: {args.xml}")
        sys.exit(1)
    
    try:
        visualize_dlib_dataset(
            xml_path=args.xml,
            output_dir=args.output,
            max_samples=args.samples,
            show_images=args.show,
            use_matplotlib=not args.opencv
        )
        
        print("\n可視化が完了しました！")
        if args.output:
            print(f"結果は {args.output} に保存されています")
        
    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
