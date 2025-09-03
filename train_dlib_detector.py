#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dlibの顔特徴点検出器学習スクリプト

近赤外線画像用の68点特徴点検出器を学習します。

使用方法:
    python train_dlib_detector.py --xml training_data.xml --output nir_shape_predictor.dat

オプション調整例:
    python train_dlib_detector.py --xml training_data.xml --output model.dat \
        --tree-depth 4 --cascade-depth 15 --nu 0.1 --feature-pool-size 500
"""

import dlib
import argparse
import os
import sys
import time
from pathlib import Path



def setup_training_options(tree_depth=4, cascade_depth=10, nu=0.1, 
                          feature_pool_size=400, oversampling_amount=10,
                          be_verbose=True):
    """
    学習オプションを設定
    
    Args:
        tree_depth (int): 各回帰木の深さ（2-5推奨）
        cascade_depth (int): カスケードの数（10-20推奨）
        nu (float): 学習率/正規化の強さ（0.01-0.25）
        feature_pool_size (int): 特徴プールサイズ（400-1000）
        oversampling_amount (int): データ増強の倍数
        be_verbose (bool): 詳細ログ表示
        
    Returns:
        dlib.shape_predictor_training_options: 設定済みオプション
    """
    
    options = dlib.shape_predictor_training_options()
    
    # 基本パラメータ
    options.tree_depth = tree_depth
    options.cascade_depth = cascade_depth 
    options.nu = nu
    options.feature_pool_size = feature_pool_size
    options.oversampling_amount = oversampling_amount
    options.be_verbose = be_verbose
    
    # 近赤外線画像用の調整
    options.num_threads = os.cpu_count()  # CPU並列化
    options.lambda_param = 0.1  # 正則化パラメータ
    
    return options


def print_training_info(options, xml_path):
    """学習情報を表示"""
    print("=== dlib顔特徴点検出器 学習開始 ===")
    print(f"学習データ: {xml_path}")
    print(f"CPU並列数: {options.num_threads}")
    print("\n--- 学習パラメータ ---")
    print(f"木の深さ (tree_depth): {options.tree_depth}")
    print(f"カスケード数 (cascade_depth): {options.cascade_depth}")
    print(f"学習率 (nu): {options.nu}")
    print(f"特徴プールサイズ (feature_pool_size): {options.feature_pool_size}")
    print(f"データ増強倍数 (oversampling_amount): {options.oversampling_amount}")
    print(f"正則化パラメータ (lambda): {options.lambda_param}")
    print()


def validate_xml_file(xml_path):
    """XMLファイルの妥当性をチェック"""
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XMLファイルが見つかりません: {xml_path}")
    
    # XMLファイル内の画像数をカウント（簡易チェック）
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            content = f.read()
            image_count = content.count('<image file=')
            if image_count == 0:
                raise ValueError("XMLファイルに画像データが含まれていません")
            print(f"学習データ: {image_count}枚の画像")
    except Exception as e:
        raise ValueError(f"XMLファイルの読み込みエラー: {e}")


def train_shape_predictor(xml_path, output_path, options):
    """
    dlibの特徴点検出器を学習
    
    Args:
        xml_path (str): 学習用XMLファイルパス
        output_path (str): 出力モデルファイルパス
        options: 学習オプション
        
    Returns:
        str: 学習済みモデルファイルパス
    """
    
    # 出力ディレクトリ作成
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 学習実行
    print("学習を開始します...")
    print("（この処理は時間がかかる場合があります）")
    
    start_time = time.time()
    
    try:
        # dlib学習関数呼び出し
        dlib.train_shape_predictor(xml_path, output_path, options)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"\n=== 学習完了 ===")
        print(f"学習時間: {training_time/60:.1f}分")
        print(f"モデルファイル: {output_path}")
        
        # ファイルサイズ表示
        model_size = os.path.getsize(output_path) / 1024 / 1024
        print(f"モデルサイズ: {model_size:.1f} MB")
        
        return output_path
        
    except Exception as e:
        raise RuntimeError(f"学習中にエラーが発生しました: {e}")


def test_trained_model(model_path):
    """学習済みモデルをテスト読み込み"""
    try:
        predictor = dlib.shape_predictor(model_path)
        print(f"✓ モデルの読み込みテスト成功")
        return True
    except Exception as e:
        print(f"✗ モデルの読み込みテスト失敗: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='dlibの顔特徴点検出器を学習',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
パラメータ調整の指針:
    tree_depth: 2-5 (大きいほど複雑だが過学習リスク)
    cascade_depth: 10-20 (大きいほど精度向上だがモデルサイズ増加)  
    nu: 0.01-0.25 (小さいほど慎重な学習)
    feature_pool_size: 400-1000 (大きいほど精度向上だが学習時間増加)

使用例:
    python train_dlib_detector.py --xml training_data.xml --output model.dat
    python train_dlib_detector.py --xml data.xml --output model.dat --tree-depth 3 --cascade-depth 12
        """
    )
    
    # 必須引数
    parser.add_argument('--xml', '-x', required=True,
                       help='学習用XMLファイルパス')
    parser.add_argument('--output', '-o', required=True,
                       help='出力モデルファイルパス (.dat)')
    
    # 学習パラメータ
    parser.add_argument('--tree-depth', type=int, default=4,
                       help='回帰木の深さ (デフォルト: 4)')
    parser.add_argument('--cascade-depth', type=int, default=10, 
                       help='カスケード数 (デフォルト: 10)')
    parser.add_argument('--nu', type=float, default=0.1,
                       help='学習率 (デフォルト: 0.1)')
    parser.add_argument('--feature-pool-size', type=int, default=400,
                       help='特徴プールサイズ (デフォルト: 400)')
    parser.add_argument('--oversampling', type=int, default=10,
                       help='データ増強倍数 (デフォルト: 10)')
    parser.add_argument('--quiet', action='store_true',
                       help='詳細ログを非表示')
    
    args = parser.parse_args()
    
    try:
        # 入力ファイル検証
        validate_xml_file(args.xml)
        
        # 学習オプション設定
        options = setup_training_options(
            tree_depth=args.tree_depth,
            cascade_depth=args.cascade_depth,
            nu=args.nu,
            feature_pool_size=args.feature_pool_size,
            oversampling_amount=args.oversampling,
            be_verbose=not args.quiet
        )
        
        # 学習情報表示
        if not args.quiet:
            print_training_info(options, args.xml)
        
        # 学習実行
        model_path = train_shape_predictor(args.xml, args.output, options)
        
        # モデルテスト
        test_trained_model(model_path)
        
        print(f"\n学習が正常に完了しました！")
        print(f"モデルファイル: {model_path}")
        print(f"\n次のコマンドでモデルをテストできます:")
        print(f"python test_nir_detector.py --model {model_path} --image test_image.png")
        
    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
