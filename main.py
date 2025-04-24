import sys
import os
import math
import logging
import random
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import io
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Optional, Dict
import psutil
from math import sqrt
import urllib.request
from torch.utils.tensorboard import SummaryWriter
import csv
import optim
from optuna.trial import TrialState
from sklearn.model_selection import KFold
import torch.optim as optim
from torch.optim import AdamW
import weakref
from functools import wraps
from collections import deque
import torch
import csv
import gmpy2
import math
# 再現性のためのシード固定
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedResultLogger:
    def __init__(self, result_dir="enhanced_results_loto7"):
        os.makedirs(result_dir, exist_ok=True)
        self.result_dir = result_dir
        self.file_handles = {}
        self.performance_history = deque(maxlen=100)

    def log_epoch(self, epoch, loss, lr, predictions_dict, confidence_dict, genetic_diversity=0.0, lyapunov_exponent=0.0):
            """拡張されたロギング機能"""
            for draw_num, predictions in predictions_dict.items():
                if draw_num not in self.file_handles:
                    file_path = os.path.join(self.result_dir, f"draw_{draw_num}.csv")
                    self.file_handles[draw_num] = open(file_path, 'a')
                    if os.stat(file_path).st_size == 0:
                        self.file_handles[draw_num].write("epoch,loss,lr,predictions,confidence,genetic_diversity,lyapunov\n")

                f = self.file_handles[draw_num]

                # predictions を空白で区切った文字列に変換
                predictions_str = " ".join(map(str, predictions))

                # confidence を空白で区切った文字列に変換 (修正)
                confidences = confidence_dict.get(draw_num, [])
                confidences_str = " ".join(f"{conf:.4f}" for conf in confidences)

                f.write(f"{epoch},{loss:.6f},{lr:.6e},{predictions_str},{confidences_str},{genetic_diversity:.4f},{lyapunov_exponent:.4f}\n")
                print(f"{draw_num},{epoch},{loss:.6f},{lr:.6e},{predictions_str},{confidences_str},{genetic_diversity:.4f},{lyapunov_exponent:.4f}\n")
                f.flush()

            self.performance_history.append(loss)

    def close(self):
        for f in self.file_handles.values():
            f.close()

class TimeSeriesFeatureEngineer:
    @staticmethod
    def add_derivative_features(raw_data, delta_norm):
        second_deriv = np.zeros_like(delta_norm)
        if delta_norm.shape[0] > 1:
            second_deriv[1:] = delta_norm[1:] - delta_norm[:-1]
        delta_features = np.concatenate([
            delta_norm,
            second_deriv,
            np.clip(delta_norm, 0, None),
            np.abs(delta_norm)
        ], axis=1)
        return delta_features

    @staticmethod
    def add_fourier_features(raw_data, window_size):
        n = len(raw_data) # Define n here
        fourier_features = np.zeros((n, 7), dtype=np.float32)
        for i in range(n):
            start = max(0, i - window_size + 1)
            window = raw_data[start:i+1]
            if len(window) <= 1:
                continue
            for pos in range(7):
                fft = np.fft.fft(window[:, pos], n=4)
                fourier_features[i, pos] = np.abs(fft)[0]
        if fourier_features.size == 0:
            return np.zeros((n, 7), dtype=np.float32)
        else:
            return fourier_features / (np.max(fourier_features, axis=0, keepdims=True) + 1e-6)

    @staticmethod
    def add_statistical_features(raw_data, window_size):
        n = len(raw_data)  # Define n here
        stats_features = np.zeros((n, 7, 3), dtype=np.float32)
        for i in range(n):
            for pos in range(7):
                start = max(0, i - window_size//2)
                window = raw_data[start:i+1, pos]
                if len(window) <= 1:
                    continue
                stats_features[i, pos] = [window.mean(), window.std(), np.median(window)]
        return stats_features
class EnhancedLotoDataset(Dataset):
    def __init__(self, csv_url, window_size=12, interval=5, exclude_numbers=None):
        self.csv_url = csv_url
        self.window_size = window_size
        self.interval = interval
        self.exclude_numbers = exclude_numbers if exclude_numbers is not None else []
        with urllib.request.urlopen(self.csv_url) as response:
            html = response.read().decode('shift-jis')
            df = pd.read_csv(io.StringIO(html),
                            usecols=['開催回','第1数字','第2数字','第3数字','第4数字','第5数字','第6数字','第7数字'])
        self.max_draw_num = df['開催回'].max() 
        
        # Load and process data
        self.raw_data = self._load_data()
        self.raw_data = self._exclude_data(self.raw_data)  # Remove excluded draws
        
        # Calculate valid indices
        self.valid_indices = list(range(window_size + interval - 1, len(self.raw_data)))
        
        # Precompute features
        self.feature_cache = []
        self._precompute_features()
        
        # Calculate feature dimension
        sample_features = self._create_sample_features(window_size + interval - 1)
        self.feature_dim = sample_features.shape[0]
        
    def _precompute_features(self):
        feature_engineer = TimeSeriesFeatureEngineer()
        for i in self.valid_indices:
            features = self._create_sample_features(i - self.interval)
            self.feature_cache.append(features.astype(np.float32))

    def __len__(self):
        return len(self.feature_cache)
    
    def __getitem__(self, idx):
        if idx >= len(self.feature_cache):
            raise IndexError(f"Index {idx} out of range for dataset with size {len(self)}")
            
        # Get precomputed features
        features = self.feature_cache[idx]
        
        # Get target (next draw)
        data_index = self.valid_indices[idx]
        target_index = data_index + self.interval
        
        if target_index >= len(self.raw_data):
            target = torch.zeros(7, dtype=torch.int64) # Changed dtype to torch.int64
        else:
            target = torch.tensor(self.raw_data[target_index], dtype=torch.int64) # Changed dtype to torch.int64
            
        return torch.tensor(features), target
    def _load_data(self):
        with urllib.request.urlopen(self.csv_url) as response:
            html = response.read().decode('shift-jis')
            df = pd.read_csv(io.StringIO(html),
                            usecols=['開催回','第1数字','第2数字','第3数字','第4数字','第5数字','第6数字','第7数字'])
            df = df.sort_values('開催回', ascending=True)
            data = df[['第1数字','第2数字','第3数字','第4数字','第5数字','第6数字','第7数字']].values.astype(np.int32)
            return data - 1  # Convert to 0-based index

    def _exclude_data(self, data):
        if self.exclude_numbers:
            mask = ~np.isin(np.arange(len(data)), self.exclude_numbers)
            data = data[mask]
        return data

    def _create_sample_features(self, index, feature_engineer=TimeSeriesFeatureEngineer()):
        window_data = self.raw_data[index - self.window_size + 1:index + 1]
        delta = np.diff(window_data, axis=0).astype(np.float32)
        delta_norm = delta / 36.0
        
        delta_features = feature_engineer.add_derivative_features(window_data, delta_norm)
        fourier_features = feature_engineer.add_fourier_features(window_data, self.window_size)
        stats_features = feature_engineer.add_statistical_features(window_data, self.window_size)
        
        features = np.hstack([
            window_data.flatten(),
            delta_features.flatten(),
            fourier_features.flatten(),
            stats_features.flatten()
        ])
        return features
class HybridQuantumLoss(nn.Module):
    """ハイブリッド量子古典損失関数"""
    def __init__(self, num_classes: int = 37, alpha: float = 0.3, beta: float = 0.2):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha  # 量子損失の重み
        self.beta = beta    # 多様性損失の重み

        # 基底損失関数
        self.base_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

        # 量子状態最適化器
        self.quantum_optimizer = QuantumCircuitOptimizer(num_qubits=num_classes)

    def quantum_divergence(self, pred_probs: torch.Tensor) -> torch.Tensor:
        """量子発散損失"""
        uniform = torch.ones_like(pred_probs) / self.num_classes

        # 量子最適化を適用
        optimized_pred = self.quantum_optimizer(pred_probs)
        optimized_uniform = self.quantum_optimizer(uniform)

        # KLダイバージェンス計算
        kl_loss = F.kl_div(
            (optimized_pred + 1e-6).log(),
            optimized_uniform,
            reduction='batchmean'
        )
        return kl_loss

    def diversity_loss(self, outputs: torch.Tensor) -> torch.Tensor:
        """予測の多様性を促進する損失"""
        # 各位置の予測分布間の類似度を計算
        sim_matrix = torch.zeros(7, 7, device=outputs.device)
        for i in range(7):
            for j in range(7):
                if i != j:
                    sim_matrix[i,j] = F.cosine_similarity(
                        outputs[:,i], outputs[:,j], dim=-1).mean()

        # 非対角成分の平均 (類似度が高いとペナルティ)
        return sim_matrix.mean()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 基本クロスエントロピー損失
        base_loss = sum(self.base_loss(outputs[:,i], targets[:,i]) for i in range(7)) / 7

        # 量子発散損失
        mean_probs = torch.softmax(outputs, dim=-1).mean(dim=1)
        q_loss = self.quantum_divergence(mean_probs)

        # 多様性損失
        div_loss = self.diversity_loss(outputs)

        # 複合損失
        total_loss = base_loss + self.alpha * q_loss + self.beta * div_loss

        return total_loss
class QuantumEnhancedTransformerBlock(nn.Module):
    """量子インスパイアード最適化を統合したTransformerブロック"""
    def __init__(self, d_model: int = 128, nhead: int = 4, dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 quantum_attention: bool = True,
                 quantum_layers: int = 5,
                 use_entanglement: bool = True,
                 quantum_noise: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.quantum_attention = quantum_attention
        self.use_entanglement = use_entanglement

        # 正規化層
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 量子注意機構
        if quantum_attention:
            self.quantum_attention_layers = nn.ModuleList([
                QuantumAttentionLayer(
                    d_model, num_heads,
                    quantum_layers=quantum_layers,
                    dropout=dropout,
                    feature_dim=feature_dim  # Pass feature_dim to QuantumAttentionLayer
                ) for _ in range(num_layers // 2)
            ])
            self.attention_mechanism = self.quantum_enhanced_attention
        else:
            self.attention_mechanism = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True
            )

        # フィードフォワードネットワーク
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        # 残差接続のスケーリング
        self.residual_scale = nn.Parameter(torch.ones(1))
        self.dropout = nn.Dropout(dropout)

    def quantum_enhanced_attention(self, query: torch.Tensor) -> torch.Tensor:
        """量子インスパイアード注意機構"""
        batch_size, seq_len, _ = query.size()

        # 通常の注意スコア計算
        scores = torch.matmul(query, query.transpose(-2, -1)) / sqrt(self.d_model)
        probs = F.softmax(scores, dim=-1)

        # 量子最適化を適用
        optimized_scores = []
        for b in range(batch_size):
            for i in range(seq_len):
                # 量子最適化を適用
                optimized_probs = self.quantum_optimizer.optimize(
                    probs[b, i].detach().cpu().numpy()
                )
                optimized_scores.append(torch.from_numpy(optimized_probs))

        optimized_scores = torch.stack(optimized_scores).view(batch_size, seq_len, seq_len).to(query.device)

        # 注意重みを適用
        return torch.matmul(F.softmax(optimized_scores, dim=-1), query)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 残差接続1
        residual = x
        x = self.norm1(x)

        # 注意機構
        if self.quantum_attention:
            attn_output = self.attention_mechanism(x)
        else:
            attn_output, _ = self.attention_mechanism(x, x, x)

        x = residual + self.dropout(attn_output) * self.residual_scale

        # 残差接続2
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout(x) * self.residual_scale
        return x
class AdvancedLotoPredictor(nn.Module):
    def __init__(self, feature_dim: int, window_size: int = 16, d_model: int = 256,
                 num_heads: int = 8, num_layers: int = 6, dropout: float = 0.1,
                 quantum_layers: int = 6, use_quantum: bool = True,
                 dataset: Optional[EnhancedLotoDataset] = None):
        super().__init__()
        self.window_size = window_size
        self.feature_dim = feature_dim  # Store feature_dim
        self.input_dim = feature_dim // window_size

        # 入力埋め込み
        self.embedding = nn.Sequential(
            nn.Linear(self.input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 位置エンコーディング
        self.pos_encoder = PositionalEncoding(d_model)

        # 量子拡張Transformer
        self.transformer = QuantumEnhancedTransformer(
            d_model, num_heads, num_layers,
            dim_feedforward=d_model*2,
            dropout=dropout,
            quantum_layers=quantum_layers,
            use_quantum=use_quantum
            #dataset=dataset  # データセットインスタンスを渡す
        )

        # 出力ヘッド (量子最適化付き)
        self.output_heads = nn.ModuleList([
            QuantumOutputHead(d_model, 37, quantum_layers=quantum_layers) # Create QuantumOutputHead here
            for _ in range(7)
        ])

        # 初期化
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.size()

        # Calculate the expected input dimension based on feature_dim and window_size
        expected_input_dim = (self.feature_dim + self.window_size - 1) // self.window_size

        # Ensure the tensor can be properly reshaped
        if L % expected_input_dim != 0:
            # Pad the input if necessary
            pad_size = expected_input_dim - (L % expected_input_dim)
            x = F.pad(x, (0, pad_size))

        # Reshape the input
        x = x.view(B, self.window_size, expected_input_dim)
        # 埋め込み
        x = self.embedding(x)
        x = x + self.pos_encoder(x)

        # Transformer処理
        x = self.transformer(x)

        # 出力
        outputs = [head(x[:, -1]) for head in self.output_heads]
        return torch.stack(outputs, dim=1)

    def predict_with_quantum(self, x: torch.Tensor, num_samples: int = 5) -> Tuple[List[List[int]], List[float]]:
        """量子インスパイアード最適化を使用した予測"""
        self.eval()
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(0)  # バッチ次元を追加

            # モデル出力 (batch_size, 7, 37)
            output = self.forward(x)

            # バッチサイズ1を想定
            probs = torch.softmax(output, dim=-1)[0]  # (7, 37)

            predictions = []
            confidence_scores = []

            for _ in range(num_samples):
                numbers = []
                confidences = []
                used_numbers = set()

                for pos in range(7):
                    # 現在の位置の確率分布を取得
                    pos_probs = probs[pos].clone().detach().cpu().numpy()  # (37,)

                    # 既に選ばれた数字を除外
                    pos_probs[list(used_numbers)] = 0

                    # 正規化（合計が1になるように）
                    pos_probs = self._safe_normalize(pos_probs)

                    # 量子最適化を適用
                    if hasattr(self, 'quantum_optimizer'):
                        # 量子最適化器への入力は2D (1, 37)
                        optimized_probs = self.quantum_optimizer(
                            torch.tensor(pos_probs).unsqueeze(0).to(self.device)
                        ).squeeze(0).cpu().numpy()
                        optimized_probs = self._safe_normalize(optimized_probs)
                    else:
                        optimized_probs = pos_probs

                    # サンプリング
                    try:
                        number = np.random.choice(37, p=optimized_probs) + 1
                        confidence = optimized_probs[number-1]  # 選択された数字の確率を信頼度とする

                        numbers.append(number)
                        confidences.append(confidence)
                        used_numbers.add(number - 1)
                    except ValueError as e:
                        print(f"Error in sampling: {e}")
                        print(f"Position: {pos}, Probabilities: {optimized_probs}")
                        # エラーが発生した場合は均一分布からサンプリング
                        valid_numbers = [i for i in range(37) if i not in used_numbers]
                        number = np.random.choice(valid_numbers) + 1
                        confidence = 1.0 / len(valid_numbers)

                        numbers.append(number)
                        confidences.append(confidence)
                        used_numbers.add(number - 1)

                predictions.append(sorted(numbers))
                confidence_scores.append(np.mean(confidences))

            return predictions, confidence_scores
    def _safe_normalize(self, probs: np.ndarray) -> np.ndarray:
        """確率分布を安全に正規化"""
        probs = np.nan_to_num(probs, nan=0.0)  # NaNを0に置換
        probs = np.clip(probs, 0.0, 1.0)  # 確率を0-1の範囲にクリップ

        # 有効な確率（>0）があるか確認
        if np.all(probs <= 0):
            probs = np.ones_like(probs)  # すべて0の場合は均一分布にする

        # 正規化
        probs_sum = probs.sum()
        if probs_sum > 0:
            probs = probs / probs_sum
        else:
            probs = np.ones_like(probs) / len(probs)

        return probs
class HyperparameterTuner:
    """ハイパーパラメータチューニング用クラス"""
    def __init__(self, dataset, num_trials=50):
        self.dataset = dataset
        self.num_trials = num_trials
        self.best_params = None
        self.study = None

    def objective(self, trial):
        # ハイパーパラメータの探索範囲
        config = {
            "d_model": trial.suggest_categorical("d_model", [64, 128, 256]),
            "nhead": trial.suggest_categorical("nhead", [2, 4, 8]),
            "num_layers": trial.suggest_int("num_layers", 2, 6),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "quantum_layers": trial.suggest_int("quantum_layers", 2, 8),
            "use_entanglement": trial.suggest_categorical("use_entanglement", [True, False]),
            "quantum_noise": trial.suggest_float("quantum_noise", 0.0, 0.1),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        }

        # クロスバリデーション
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        val_losses = []

        for train_idx, val_idx in kf.split(self.dataset):
            train_subset = torch.utils.data.Subset(self.dataset, train_idx)
            val_subset = torch.utils.data.Subset(self.dataset, val_idx)

            train_loader = DataLoader(
                    train_dataset,
                    batch_size=config["batch_size"],
                    shuffle=True,
                    num_workers=0,  # Or lower if already reduced
                    persistent_workers=False  # Disable persistent workers
                )

            val_loader = DataLoader(
                val_subset,
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=2
            )

            # モデル初期化
            model = AdvancedLotoPredictor(
                feature_dim=self.dataset.feature_dim,
                window_size=self.dataset.window_size,
                d_model=config["d_model"],
                nhead=config["nhead"],
                num_layers=config["num_layers"],
                dropout=config["dropout"],
                use_quantum=True,
                quantum_layers=config["quantum_layers"],
                use_entanglement=config["use_entanglement"],
                quantum_noise=config["quantum_noise"]
            ).to(device)

            # オプティマイザ
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config["lr"],
                weight_decay=config["weight_decay"]
            )

            # 損失関数
            loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

            # 訓練ループ (簡易版)
            best_val_loss = float('inf')
            for epoch in range(10):  # 各foldで10エポックのみ実行
                # 訓練
                model.train()
                for src, tgt in train_loader:
                    src, tgt = src.to(device), tgt.to(device)
                    optimizer.zero_grad()
                    outputs = model(src)
                    loss = sum(loss_fn(outputs[:, i], tgt[:, i]) for i in range(7)) / 7
                    loss.backward()
                    optimizer.step()

                # 検証
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for src, tgt in train_loader:
                        src, tgt = src.to(device), tgt.to(device)
                        outputs = model(src)
                        loss = sum(loss_fn(outputs[:, i], tgt[:, i]) for i in range(7)) / 7
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

            val_losses.append(best_val_loss)

        return np.mean(val_losses)

    def tune(self):
        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(self.objective, n_trials=self.num_trials)
        self.best_params = self.study.best_params
        return self.best_params
class QuantumOutputHead(nn.Module):
    """量子拡張出力ヘッド"""
    def __init__(self, input_dim: int, num_classes: int = 37, quantum_layers: int = 3):  # num_classesを追加
        super().__init__()
        self.input_dim = input_dim
        self.classical_head = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_classes)
        )

        # 量子最適化器
        self.quantum_optimizer = QuantumCircuitOptimizer(
            num_qubits=num_classes,
            num_layers=quantum_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 古典的な予測
        logits = self.classical_head(x)

        if self.training:
            return logits

        # 推論時は量子最適化を適用
        probs = torch.softmax(logits, dim=-1)
        optimized_probs = self.quantum_optimizer(probs)

        # 最適化された確率の対数を返す
        return torch.log(optimized_probs + 1e-6)
class QuantumTrainer:
    """改良版トレーナー"""
    def __init__(self, model, train_loader, val_loader, config,result_dir="results_loto7", checkpoint_path="checkpoints/best_model_loto7.pth", tensorboard_log_dir="runs"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.predict_numbers = config['predict_numbers']
        self.dataset = EnhancedLotoDataset(config["csv_url"], config["window_size"])
        # 損失関数
        self.criterion = HybridQuantumLoss()

        # オプティマイザ
        self.optimizer = AdamW( # Use the AdamW optimizer
            model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )

        # スケジューラ
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.get('lr', 1e-4),  
            steps_per_epoch=len(train_loader),  
            epochs=config['num_epochs'],  
            anneal_strategy='cos'  
        )

        # AMPスケーラー
        self.scaler = GradScaler(enabled=torch.cuda.is_available() and config.get('use_amp', True))
        # 早期停止
        self.best_loss = float('inf')
        self.patience = config.get('patience', 30)
        self.counter = 0
        self.predict_numbers = ['predict_numbers'] # 予測する回号
        # 結果ロガー
        self.result_logger = EnhancedResultLogger(result_dir)
        self.writer = SummaryWriter(tensorboard_log_dir)
        # チェックポイント
        self.checkpoint_path = checkpoint_path

        # モデルをデバイスに移動
        self.model.to(self.device)
        self.checkpoint_interval = 500

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x: Input tensor.

        Returns:
            Model output tensor.
        """
        return self.model(x)  # Delegate the forward pass to the model attribute


    def predict_with_quantum(self, x: torch.Tensor, num_samples: int = 5) -> Tuple[List[List[int]], List[float]]:
        """量子インスパイアード最適化を使用した予測"""
        self.eval()
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(0)  # バッチ次元を追加

            # モデル出力 (batch_size, 7, 37)
            output = self.forward(x)

            # バッチサイズ1を想定
            probs = torch.softmax(output, dim=-1)[0]  # (7, 37)

            predictions = []
            confidence_scores = []

            for _ in range(num_samples):
                numbers = []
                confidences = []
                used_numbers = set()

                for pos in range(7):
                    # 現在の位置の確率分布を取得
                    pos_probs = probs[pos].clone().detach().cpu().numpy()  # (37,)

                    # 既に選ばれた数字を除外
                    pos_probs[list(used_numbers)] = 0

                    # 正規化（合計が1になるように）
                    pos_probs = self._safe_normalize(pos_probs)

                    # 量子最適化を適用
                    if hasattr(self, 'quantum_optimizer'):
                        # 量子最適化器への入力は2D (1, 37)
                        optimized_probs = self.quantum_optimizer(
                            torch.tensor(pos_probs).unsqueeze(0).to(self.device)
                        ).squeeze(0).cpu().numpy()
                        optimized_probs = self._safe_normalize(optimized_probs)
                    else:
                        optimized_probs = pos_probs

                    # サンプリング
                    try:
                        number = np.random.choice(37, p=optimized_probs) + 1
                        confidence = optimized_probs[number-1]  # 選択された数字の確率を信頼度とする

                        numbers.append(number)
                        confidences.append(confidence)
                        used_numbers.add(number - 1)
                    except ValueError as e:
                        print(f"Error in sampling: {e}")
                        print(f"Position: {pos}, Probabilities: {optimized_probs}")
                        # エラーが発生した場合は均一分布からサンプリング
                        valid_numbers = [i for i in range(37) if i not in used_numbers]
                        number = np.random.choice(valid_numbers) + 1
                        confidence = 1.0 / len(valid_numbers)

                        numbers.append(number)
                        confidences.append(confidence)
                        used_numbers.add(number - 1)

                predictions.append(sorted(numbers))
                confidence_scores.append(np.mean(confidences))

        return predictions, confidence_scores

    def train_epoch(self, config, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
            src, tgt = batch
            src, tgt = src.to(self.device), tgt.to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.scaler.is_enabled()):
                outputs = self.model(src)
                loss = self.criterion(outputs, tgt)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )


            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        # 予測とロギング
        # Get the last input from the training data
        initial_input = src[-1]  # Assuming you want the last sample in the batch
        # Set the number of steps for recursive prediction
        num_steps = 1 #or whatever number you wish

        # 予測とロギング
        for i in  range(len(config["predict_numbers"])):
            predictions_dict, confidence_dict = self.recursive_predict(src[-1], [config["predict_numbers"][i]])  # target_draw_nums をリストで指定
            self.result_logger.log_epoch(epoch, avg_loss, self.scheduler.get_last_lr()[0], predictions_dict, confidence_dict)
        self.writer.add_scalar('Loss/train', avg_loss, epoch)

        if epoch % self.checkpoint_interval == 0:
            self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth")

        # Update the scheduler after each epoch
        self.scheduler.step()  

        return avg_loss

    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                src, tgt = batch
                src, tgt = src.to(self.device), tgt.to(self.device)

                outputs = self.model(src)
                loss = self.criterion(outputs, tgt)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)

        # 早期停止チェック
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.counter = 0
        else:
            self.counter += 1

        return avg_loss, self.counter >= self.patience

    # QuantumTrainerクラスのfitメソッドを修正
    def fit(self, config, epochs: int):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(config, epoch)
            val_loss, should_stop = self.validate()
            
            print(f"Epoch {epoch}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")


    def _safe_normalize(self, probs: np.ndarray) -> np.ndarray:
        """確率分布を安全に正規化"""
        probs = np.nan_to_num(probs, nan=0.0)  # NaNを0に置換
        probs = np.clip(probs, 0.0, 1.0)  # 確率を0-1の範囲にクリップ

        # 有効な確率（>0）があるか確認
        if np.all(probs <= 0):
            probs = np.ones_like(probs)  # すべて0の場合は均一分布にする

        # 正規化
        probs_sum = probs.sum()
        if probs_sum > 0:
            probs = probs / probs_sum
        else:
            probs = np.ones_like(probs) / len(probs)

        return probs

    def recursive_predict(self, initial_input, target_draw_nums, max_recursion_depth=5):
        predictions_dict = {}
        confidence_dict = {}

        for draw_num in target_draw_nums:
            current_input = initial_input
            data_index = draw_num - self.dataset.window_size - self.dataset.interval

            # Ensure data_index is within dataset bounds
            data_index = max(0, min(data_index, len(self.dataset) - 1))

            # 再帰的なインデックス調整
            def adjust_data_index(data_index, recursion_depth):
                if 0 <= data_index < len(self.dataset):
                    return data_index  # 有効なインデックスが見つかった場合
                elif recursion_depth < max_recursion_depth:
                    return adjust_data_index(data_index - 1, recursion_depth + 1)  # 再帰的に調整
                else:
                    return -1  # 最大再帰深度に達した場合

            data_index = adjust_data_index(data_index, 0)

            if data_index == -1:
                print(f"Warning: Could not find valid data index for draw_num {draw_num} after recursion. Skipping prediction.")
                continue  # 有効なインデックスが見つからない場合はスキップ

            current_input = self.dataset[data_index][0]  # 対応する入力データを取得

            # 予測
            with torch.no_grad():
                output = self.forward(current_input.unsqueeze(0).to(self.device))
            probs = torch.softmax(output, dim=-1)[0]

            # サンプリング (7つの数字を予測)
            predicted_numbers = []
            used_numbers = set()
            confidences = []

            for pos in range(7):
                pos_probs = probs[pos].detach().cpu().numpy()
                pos_probs[list(used_numbers)] = 0
                pos_probs = self._safe_normalize(pos_probs)

                # 量子最適化 (必要に応じて)
                if hasattr(self, 'quantum_optimizer'):
                    optimized_probs = self.quantum_optimizer.optimize(
                        torch.tensor(pos_probs).unsqueeze(0).to(self.device))
                    optimized_probs = optimized_probs.squeeze(0).cpu().numpy()
                    optimized_probs = self._safe_normalize(optimized_probs)
                else:
                    optimized_probs = pos_probs

                number = np.random.choice(37, p=optimized_probs) + 1
                confidence = optimized_probs[number - 1]  # 選択された数字の確率を信頼度とする
                predicted_numbers.append(number)
                confidences.append(confidence)
                used_numbers.add(number - 1)

            predictions_dict[draw_num] = sorted(predicted_numbers)
            confidence_dict[draw_num] = confidences

        return predictions_dict, confidence_dict  # 辞書を返す
    def predict(self, config):
        """指定された対象回を予測し、結果を保存する"""
        self.model.eval()  # モデルを評価モードに設定
        predict_numbers = config['predict_numbers']  # 予測する回号を取得

        for draw_num in predict_numbers:
            # 予測するデータの準備 (draw_num に基づいて)
            data_index = draw_num - self.dataset.window_size - self.dataset.interval  # インデックスを計算
            if 0 <= data_index < len(self.dataset):
                input_data = self.dataset[data_index][0].unsqueeze(0).to(self.device)  # データを取得し、バッチ次元を追加

                # 予測の実行
                predictions, confidence_scores = self.model.predict_with_quantum(input_data)

                # EnhancedResultLoggerを使用して予測結果を保存
                self.result_logger.log_epoch(
                    epoch=draw_num,  # draw_numをepochとして使用
                    loss=0,  # 損失はダミー値
                    lr=0,  # 学習率はダミー値
                    predictions_dict={draw_num: predictions},  # 予測結果を辞書で渡す
                    confidence_dict={draw_num: confidence_scores}  # 信頼度を辞書で渡す
                )

    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
        }, path)
        logger.info(f"Checkpoint saved at {path}")

    def save_bestcheckpoint(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
        }, self.checkpoint_path)
        logger.info(f"Best checkpoint saved at {self.checkpoint_path}")

    def close(self):
        self.result_logger.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
class QuantumEnhancedTransformer(nn.Module):
    """量子拡張Transformerの改良版"""
    def __init__(self, d_model: int = 256, num_heads: int = 8, num_layers: int = 6,
                 dim_feedforward: int = 512, dropout: float = 0.1,
                 quantum_layers: int = 6, use_quantum: bool = True,
                 dataset: Optional[EnhancedLotoDataset] = None):  # datasetパラメータ追加
        super().__init__()
        self.d_model = d_model
        self.use_quantum = use_quantum
        self.num_heads = num_heads # assign num_heads so it can be accessed in this scope
        #dataset = EnhancedLotoDataset
        # 量子注意機構
        self.quantum_attention_layers = nn.ModuleList([
            QuantumAttentionLayer(
                d_model, num_heads, # this num_heads parameter should be accessible now
                quantum_layers=quantum_layers,
                dropout=dropout
            ) for _ in range(num_layers // 2)
        ])


        # 古典的注意機構
        self.classical_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                d_model, num_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers - len(self.quantum_attention_layers))
        ])

        # フィードフォワードネットワーク
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        # 正規化層
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, (attn_layer, norm) in enumerate(zip(
            self.quantum_attention_layers + self.classical_attention_layers,
            self.norms
        )):
            residual = x
            x = norm(x)

            if i < len(self.quantum_attention_layers):
                x = attn_layer(x)
            else:
                x, _ = attn_layer(x, x, x)

            x = residual + x
            x = self.ffn(x) + x

        return x
class AdvancedLotoPredictor(nn.Module):
    def __init__(self, feature_dim: int, window_size: int = 16, d_model: int = 256,
                 num_heads: int = 8, num_layers: int = 6, dropout: float = 0.1,
                 quantum_layers: int = 6, use_quantum: bool = True,
                 dataset: Optional[EnhancedLotoDataset] = None):
        super().__init__()
        self.window_size = window_size
        self.feature_dim = feature_dim  # Store feature_dim
        self.input_dim = feature_dim // window_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Calculate the expected input dimension based on feature_dim and window_size
        self.expected_input_dim = (self.feature_dim + self.window_size - 1) // self.window_size
        self.quantum_optimizer = QuantumCircuitOptimizer(num_qubits=37, num_layers=quantum_layers)
        # 入力埋め込み
        self.embedding = nn.Sequential(
            nn.Linear(self.expected_input_dim, d_model), # Use expected_input_dim
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 位置エンコーディング
        self.pos_encoder = PositionalEncoding(d_model)

        # 量子拡張Transformer
        self.transformer = QuantumEnhancedTransformer(
            d_model, num_heads, num_layers,
            dim_feedforward=d_model*2,
            dropout=dropout,
            quantum_layers=quantum_layers,
            use_quantum=use_quantum
            #dataset=dataset  # データセットインスタンスを渡す
        )

        # 出力ヘッド (量子最適化付き)
        self.output_heads = nn.ModuleList([
            QuantumOutputHead(d_model, 37, quantum_layers=quantum_layers) # Create QuantumOutputHead here
            for _ in range(7)
        ])

        # 初期化
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.size()
        expected_input_dim = (self.feature_dim + self.window_size - 1) // self.window_size

        if L % expected_input_dim != 0:
            pad_size = expected_input_dim - (L % expected_input_dim)
            x = F.pad(x, (0, pad_size))

        x = x.view(B, self.window_size, expected_input_dim)
        x = self.embedding(x)
        x = x + self.pos_encoder(x)
        x = self.transformer(x)

        # 各位置ごとの出力を取得
        outputs = []
        for head in self.output_heads:
            # 各ヘッドへの入力は (batch_size, d_model)
            head_input = x[:, -1]  # 最後のタイムステップのみ使用
            if head_input.dim() == 1:
                head_input = head_input.unsqueeze(0)  # バッチ次元を追加
            outputs.append(head(head_input))

        return torch.stack(outputs, dim=1)  # (batch_size, 7, 37)

    def _safe_normalize(self, probs: np.ndarray) -> np.ndarray:
        """確率分布を安全に正規化"""
        probs = np.nan_to_num(probs, nan=0.0)  # NaNを0に置換
        probs = np.clip(probs, 0.0, 1.0)  # 確率を0-1の範囲にクリップ

        # 有効な確率（>0）があるか確認
        if np.all(probs <= 0):
            probs = np.ones_like(probs)  # すべて0の場合は均一分布にする

        # 正規化
        probs_sum = probs.sum()
        if probs_sum > 0:
            probs = probs / probs_sum
        else:
            probs = np.ones_like(probs) / len(probs)

        return probs
    def predict_with_quantum(self, x: torch.Tensor, num_samples: int = 5) -> Tuple[List[List[int]], List[float]]:
        """量子インスパイアード最適化を使用した予測"""
        self.eval()
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(0)  # バッチ次元を追加

            # モデル出力 (batch_size, 7, 37)
            output = self.forward(x)

            # バッチサイズ1を想定
            probs = torch.softmax(output, dim=-1)[0]  # (7, 37)

            predictions = []
            confidence_scores = []

            for _ in range(num_samples):
                numbers = []
                confidences = []
                used_numbers = set()

                for pos in range(7):
                    # 現在の位置の確率分布を取得
                    pos_probs = probs[pos].clone().detach().cpu().numpy()  # (37,)

                    # 既に選ばれた数字を除外
                    pos_probs[list(used_numbers)] = 0

                    # 正規化（合計が1になるように）
                    pos_probs = self._safe_normalize(pos_probs)

                    # 量子最適化を適用
                    if hasattr(self, 'quantum_optimizer'):
                        # 量子最適化器への入力は2D (1, 37)
                        optimized_probs = self.quantum_optimizer(
                            torch.tensor(pos_probs).unsqueeze(0).to(self.device)
                        ).squeeze(0).cpu().numpy()
                        optimized_probs = self._safe_normalize(optimized_probs)
                    else:
                        optimized_probs = pos_probs

                    # サンプリング
                    try:
                        number = np.random.choice(37, p=optimized_probs) + 1
                        confidence = optimized_probs[number-1]  # 選択された数字の確率を信頼度とする

                        numbers.append(number)
                        confidences.append(confidence)
                        used_numbers.add(number - 1)
                    except ValueError as e:
                        print(f"Error in sampling: {e}")
                        print(f"Position: {pos}, Probabilities: {optimized_probs}")
                        # エラーが発生した場合は均一分布からサンプリング
                        valid_numbers = [i for i in range(37) if i not in used_numbers]
                        number = np.random.choice(valid_numbers) + 1
                        confidence = 1.0 / len(valid_numbers)

                        numbers.append(number)
                        confidences.append(confidence)
                        used_numbers.add(number - 1)

                predictions.append(sorted(numbers))
                confidence_scores.append(np.mean(confidences))

            return predictions, confidence_scores
class FutureLotoPredictor:
    """将来の抽選を予測するクラス"""
    def __init__(self, model: nn.Module, dataset: EnhancedLotoDataset, num_future_draws: int = 10):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.num_future_draws = num_future_draws
        # self.current_features = dataset.feature_cache[-1].copy() #feature_cacheはdatasetにない
        self.current_draw_index = len(dataset.raw_data) -1 # 予測開始のインデックス
        self.device = next(model.parameters()).device
        self.model.eval()
        self.window_size = window_size
        # self.feature_engineer = ChaosEnhancedFeatureEngineer() #feature_engineerはない
    def _create_sample_features(self, index, feature_engineer=TimeSeriesFeatureEngineer()):
        window_data = self.raw_data[index - self.window_size + 1:index + 1]
        delta = np.diff(window_data, axis=0).astype(np.float32)
        delta_norm = delta / 36.0
        delta_features = feature_engineer.add_derivative_features(window_data, delta_norm)
        fourier_features = feature_engineer.add_fourier_features(window_data, self.window_size)
        stats_features = feature_engineer.add_statistical_features(window_data, self.window_size)
        features = np.hstack([
            window_data.flatten(),
            delta_features.flatten(),
            fourier_features.flatten(),
            stats_features.flatten()])
        return features
    def generate_future_features(self, predicted_numbers: List[int]) -> np.ndarray:
        """予測結果に基づいて特徴量を更新"""
        predicted_numbers_np = np.array(predicted_numbers)

        # 過去のデータと予測結果を結合して、次の予測のための入力特徴量を生成
        new_data = self.dataset.raw_data[self.current_draw_index - self.dataset.window_size + 1 : self.current_draw_index] + [predicted_numbers] #スライスする際にend indexも含める必要があるので+1
        new_features = self.dataset._extract_features(new_data)
        return new_features

    def predict_next(self):
        """次の抽選を予測"""
        with torch.no_grad():
            # データが足りない場合の再帰的予測
            if self.current_draw_index >= len(self.dataset.raw_data):  # current_draw_indexがデータの範囲外の場合
                # 必要な回数分の再帰的予測
                num_recursive_predictions = self.current_draw_index - len(self.dataset.raw_data) + 1
                for _ in range(num_recursive_predictions):
                    recursive_prediction, _ = self.predict_next()  # 再帰的に予測
                    self.dataset.raw_data = np.vstack([self.dataset.raw_data, [recursive_prediction]])  # 予測結果をデータに追加

            # 予測の実行
            input_features = self.dataset._create_sample_features(self.current_draw_index - self.dataset.window_size + 1)
            inputs = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            predictions, _ = self.model.predict_with_quantum(inputs, num_samples=5)
            best_prediction = predictions[0][:, 0]  # 最初のサンプルが最良の予測と仮定
            return best_prediction


    def predict_future_sequence(self) -> List[Tuple[List[int], List[float]]]: # 信頼度をList[float]で返すように修正
        """連続した将来の抽選を予測"""
        future_predictions = []
        for _ in range(self.num_future_draws):
            numbers, confidence = self.predict_next()
            future_predictions.append((numbers, confidence))
            self.current_draw_index += 1
        return future_predictions

def evaluate(model, loader, config):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(config['device'])
            targets = targets.to(config['device'])

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(loader)

class AdvancedLotoTrainer:
    def __init__(self, model, train_loader, config,result_dir="results_loto7", checkpoint_path="checkpoints/best_model_loto7.pth", tensorboard_log_dir="runs"):
        self.model = model
        self.loader = train_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.predict_numbers = config["predict_numbers"]
        self.result_logger = ResultLogger(result_dir)
        self.scaler = GradScaler(enabled=torch.cuda.is_available() and config.get('use_amp', True))
        self.checkpoint_interval = 500
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        steps_per_epoch = len(train_loader)
        self.optimizer = AdamW( # Use the AdamW optimizer
            model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )

        # スケジューラ初期化時に正しいパラメータを設定
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.get('lr', 1e-4),  
            steps_per_epoch=len(train_loader),  
            epochs=config['num_epochs'],  
            anneal_strategy='cos'  
        )

        self.checkpoint_path = checkpoint_path
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.model.to(self.device)
        self.best_loss = float('inf')
        self.future_predictor = FutureLotoPredictor(model, train_loader.dataset)
        self.writer = SummaryWriter(tensorboard_log_dir)

    def train_epoch(self,config, epoch):
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.loader, desc=f"Epoch {epoch}", leave=False)
        if torch.cuda.is_available():
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
            self.optimizer.step()

        # Step the scheduler after each batch update
        self.scheduler.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': self.scheduler.get_last_lr()[0]})
        del src, tgt, outputs, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.scheduler.step()
        avg_loss = total_loss / len(self.loader)
        # 予測とロギング
        initial_input = src[-1]
        num_steps = 1
        # 予測とロギング
        for i in  range(len(config["predict_numers"])):
            predictions_dict, confidence_dict = self.recursive_predict(src[-1], [config["predict_numers"][i]])  # target_draw_nums をリストで指定
            self.result_logger.log_epoch(epoch, avg_loss, self.scheduler.get_last_lr()[0], predictions_dict, confidence_dict)
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        if epoch % self.checkpoint_interval == 0:
            self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth")
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.save_bestcheckpoint()
        logger.info(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Best: {self.best_loss:.4f}")

def recursive_predict(self, initial_input, target_draw_nums, max_recursion_depth=5):
    predictions_dict = {}
    confidence_dict = {}

    for draw_num in target_draw_nums:
        current_input = initial_input
        data_index = draw_num - self.dataset.window_size - self.dataset.interval

        # Ensure data_index is within dataset bounds
        data_index = max(0, min(data_index, len(self.dataset) - 1))

            # 再帰的なインデックス調整
        def adjust_data_index(data_index, recursion_depth):
                if 0 <= data_index < len(self.dataset):
                    return data_index  # 有効なインデックスが見つかった場合
                elif recursion_depth < max_recursion_depth:
                    return adjust_data_index(data_index - 1, recursion_depth + 1)  # 再帰的に調整
                else:
                    return -1  # 最大再帰深度に達した場合

        data_index = adjust_data_index(data_index, 0)

        if data_index == -1:
                print(f"Warning: Could not find valid data index for draw_num {draw_num} after recursion. Skipping prediction.")
                continue  # 有効なインデックスが見つからない場合はスキップ

        current_input = self.dataset[data_index][0]  # 対応する入力データを取得

            # 予測
        with torch.no_grad():
            output = self.forward(current_input.unsqueeze(0).to(self.device))
        probs = torch.softmax(output, dim=-1)[0]

            # サンプリング (7つの数字を予測)
        predicted_numbers = []
        used_numbers = set()
        confidences = []

        for pos in range(7):
            pos_probs = probs[pos].detach().cpu().numpy()
            pos_probs[list(used_numbers)] = 0
            pos_probs = self._safe_normalize(pos_probs)

            # 量子最適化 (必要に応じて)
            if hasattr(self, 'quantum_optimizer'):
                optimized_probs = self.quantum_optimizer.optimize(
                    torch.tensor(pos_probs).unsqueeze(0).to(self.device))
                optimized_probs = optimized_probs.squeeze(0).cpu().numpy()
                optimized_probs = self._safe_normalize(optimized_probs)
            else:
                optimized_probs = pos_probs
                number = np.random.choice(37, p=optimized_probs) + 1
                confidence = optimized_probs[number - 1]  # 選択された数字の確率を信頼度とする
                predicted_numbers.append(number)
                confidences.append(confidence)
                used_numbers.add(number - 1)

            predictions_dict[draw_num] = sorted(predicted_numbers)
            confidence_dict[draw_num] = confidences

        return predictions_dict, confidence_dict  # 辞書を返す
    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
        }, path)
        logger.info(f"Checkpoint saved at {path}")

    def save_bestcheckpoint(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
        }, self.checkpoint_path)
        logger.info(f"Best checkpoint saved at {self.checkpoint_path}")

    def close(self):
        self.result_logger.close()
        self.writer.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def print_model_summary(model):
    logger.info("Model Summary:")
    logger.info(model)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total Parameters: {total_params}")
class Optimizer:
    """
    カスタムオプティマイザクラス
    """
    def __init__(self, params, lr=4e-9, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        # ハイパーパラメータの設定
        self.params = list(params)  # 最適化するパラメータのリスト
        self.lr = lr  # 学習率
        self.betas = betas  # モーメンタム係数
        self.eps = eps  # 数値安定化のための小さな値
        self.weight_decay = weight_decay  # 重み減衰

        # モーメンタムと分散の初期化
        self.m = [torch.zeros_like(p) for p in self.params]  # モーメンタム
        self.v = [torch.zeros_like(p) for p in self.params]  # 分散

    def zero_grad(self):
        """
        勾配をゼロにリセット
        """
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def step(self):
        """
        パラメータを更新
        """
        for i, p in enumerate(self.params):
            if p.grad is not None:
                # 重み減衰の適用
                if self.weight_decay != 0:
                    p.grad.data.add_(p.data, alpha=self.weight_decay)

                # モーメンタムの更新
                self.m[i].mul_(self.betas[0]).add_(p.grad, alpha=1 - self.betas[0])

                # 分散の更新
                self.v[i].mul_(self.betas[1]).addcmul_(p.grad, p.grad, value=1 - self.betas[1])

                # モーメンタムと分散のバイアス補正
                m_hat = self.m[i] / (1 - self.betas[0] ** (self.t + 1))
                v_hat = self.v[i] / (1 - self.betas[1] ** (self.t + 1))

                # パラメータの更新
                p.data.addcdiv_(m_hat, torch.sqrt(v_hat) + self.eps, value=-self.lr)

        self.t += 1  # タイムステップの更新
class QuantumAttentionLayer(nn.Module):
    """量子注意機構"""
    def __init__(self, d_model: int = 256, num_heads: int = 8,
                 quantum_layers: int = 4, dropout: float = 0.1,
                 feature_dim: int = 12):  # feature_dimを追加
        super().__init__()
        self.num_qubits = feature_dim
        self.num_layers = quantum_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.feature_dim = feature_dim  # Store feature_dim
        self.quantum_optimizer = QuantumCircuitOptimizer(
            num_qubits=self.feature_dim,  # num_qubitsをfeature_dimに設定
            num_layers=quantum_layers
        )

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        # query shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = query.size()

        # Reshape query to apply quantum layer individually to each element
        query = query.view(batch_size * seq_len, self.d_model)

        # Repeat for each head
        optimized_queries = []
        for _ in range(self.num_heads):
            # Apply QuantumCircuitOptimizer for optimization
            optimized_query = self.quantum_optimizer(query)
            optimized_queries.append(optimized_query)

        # Concatenate optimized queries from each head
        optimized_query = torch.cat(optimized_queries, dim=-1)

        # Reshape back to original shape
        optimized_query = optimized_query.view(batch_size, seq_len, self.d_model * self.num_heads)

        return optimized_query

    def _apply_rotation(self, state, rot_mat):
        """量子回転を適用 (3Dテンソル対応)"""
        # Convert the state to a complex tensor if it's not already
        if not torch.is_complex(state):
            state = state.type(torch.complex64) #changed to complex64

        # Apply the rotation to each element of the state tensor
        rotated_state = torch.zeros_like(state, dtype=torch.complex64)
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                # Reshape state[i, j] to a 2D tensor before applying matmul
                # The error is here. state[i, j] is a scalar,
                # and you can't reshape it to (2, 1).
                # Instead, just multiply the scalar by the rotation matrix.
                rotated_state[i, j] = (state[i, j] * rot_mat).sum() #removed item() to avoid recursion error


        return rotated_state
    def _apply_quantum_circuit(self, probs: torch.Tensor, head_idx: int) -> torch.Tensor:
        """量子回路を確率分布に適用"""
        batch_size, seq_len = probs.shape # probsのshapeを修正
        amplitudes = torch.sqrt(probs)  # 確率振幅に変換
        # 量子回路適用
        for layer in range(self.quantum_layers):
            # 回転ゲート適用
            angles = self.quantum_weights[layer, head_idx]
            rot_mat = self._rotation_matrix(angles)
            amplitudes = self._apply_rotation(amplitudes, rot_mat) #_apply_rotationを修正
            # エンタングルメント適用
            amplitudes = self._apply_entanglement(amplitudes, head_idx, num_qubits=37) # エンタングルメント適用, num_qubitsを渡す
        probabilities = torch.abs(amplitudes)**2
        return probabilities


    def _rotation_matrix(self, angles: torch.Tensor) -> torch.Tensor:
        """回転ゲート行列を生成"""
        phi, theta, omega = angles
        # Ensure the output is a complex tensor by using torch.complex
        # This will create a complex tensor with real and imaginary parts initialized from the cos and sin functions
        return torch.complex(
            torch.cos(theta / 2) * torch.exp(-1j * (phi + omega) / 2),
            -torch.sin(theta / 2) * torch.exp(1j * (phi - omega) / 2),
            torch.sin(theta / 2) * torch.exp(-1j * (phi - omega) / 2),
            torch.cos(theta / 2) * torch.exp(1j * (phi + omega) / 2)
        ).reshape(2, 2) # Reshape to 2x2 matrix
    def _apply_rotation(self, state, rot_mat):
        """量子回転を適用 (3Dテンソル対応)"""
        # Convert the state to a complex tensor if it's not already
        if not torch.is_complex(state):
            state = state.type(torch.complex64) #changed to complex64

        # Apply the rotation to each element of the state tensor
        rotated_state = torch.zeros_like(state, dtype=torch.complex64)
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                # Reshape state[i, j] to a 2D tensor before applying matmul
                # The error is here. state[i, j] is a scalar,
                # and you can't reshape it to (2, 1).
                # Instead, just multiply the scalar by the rotation matrix.
                rotated_state[i, j] = (state[i, j] * rot_mat).sum() #removed item() to avoid recursion error


        return rotated_state
    def _apply_entanglement(self, state: torch.Tensor, head_idx: int, num_qubits: int) -> torch.Tensor:
        """エンタングルメントゲートを適用 (修正版)"""
        # Convert the state to a complex tensor if it's not already
        if not torch.is_complex(state):
            state = state.type(torch.complex64)
        batch_size = state.shape[0]
        # num_qubits = 37 # 引数で受け取るように修正
        for i, (q1, q2) in enumerate(self.entanglement_pairs):
            angle = self.entanglement_weights[head_idx, i]
            for b in range(batch_size):
                for j in range(2 ** num_qubits):
                    control_bit = (j >> (num_qubits - 1 - q1)) & 1
                    target_bit = (j >> (num_qubits - 1 - q2)) & 1
                    if control_bit == 1 and target_bit == 1:
                        k = j ^ (1 << (num_qubits - 1 - q2))
                        temp = state[b, j].clone()
                        state[b, j] = state[b, k]
                        state[b, k] = temp
        return state
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # クエリ、キー、バリューの計算
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意スコア計算
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 注意重み計算
        attn_probs = F.softmax(attn_scores, dim=-1)

        # 量子最適化を各ヘッドに適用
        attn_output = torch.zeros_like(q) #attn_outputの初期化を修正
        attn_output = attn_output.type(torch.complex64) # changed to torch.complex64
        for h in range(self.num_heads):
            head_probs = attn_probs[:, h]
            optimized_probs = self._apply_quantum_circuit(head_probs, h)
            # 注意出力計算
            optimized_probs = optimized_probs.view(batch_size, seq_len, seq_len).to(v.device)  # vと同じデバイスに移動
            # バッチサイズと系列長次元を残して注意を適用
            attn_output[:, h] = torch.matmul(optimized_probs, v[:, h])

        # ドロップアウト適用
        attn_output = self.dropout(attn_output)

        # ヘッドの結合
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # 最終線形変換
        return self.out_proj(attn_output)
class QuantumCircuitOptimizer(nn.Module):
    def __init__(self, num_qubits, num_layers=4, entanglement_strength=0.5):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.rot_params = nn.Parameter(torch.rand(num_layers, num_qubits, 3) * math.pi)
        self.entanglement = nn.Parameter(torch.ones(num_qubits, num_qubits) * entanglement_strength)

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        # 確率分布を量子状態に変換 (batch_size, num_qubits)
        state = torch.sqrt(probs + 1e-6)
        state = state / torch.norm(state, dim=-1, keepdim=True)

        # 量子回路の適用
        for layer in range(self.num_layers):
            # 回転ゲート適用
            for qubit in range(self.num_qubits):
                rot_mat = self._rotation_matrix(self.rot_params[layer, qubit])
                state = self._apply_qubit_rotation(state, qubit, rot_mat)

            # エンタングルメント適用
            state = self._apply_entanglement(state)

        # 測定確率を計算し、正規化
        probabilities = (state.abs() ** 2)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)

        return probabilities
    def _rotation_matrix(self, angles: torch.Tensor) -> torch.Tensor:
        """回転ゲート行列を生成"""
        phi, theta, omega = angles
        return torch.tensor([
            [torch.cos(theta/2)*torch.exp(-1j*(phi+omega)/2),
             -torch.sin(theta/2)*torch.exp(1j*(phi-omega)/2)],
            [torch.sin(theta/2)*torch.exp(-1j*(phi-omega)/2),
             torch.cos(theta/2)*torch.exp(1j*(phi+omega)/2)]
        ], dtype=torch.complex64)

    def _apply_qubit_rotation(self, state: torch.Tensor, qubit: int, matrix: torch.Tensor) -> torch.Tensor:
        """特定のqubitに回転ゲートを適用"""
        # state: (batch_size, num_qubits)
        batch_size, num_qubits = state.shape

        # 単位行列を準備
        full_matrix = torch.eye(num_qubits, dtype=torch.complex64, device=state.device)

        # 対象qubitに回転ゲートを設定
        mask = 1 << (num_qubits - 1 - qubit)
        for i in range(num_qubits):
            target = i ^ mask
            if 0 <= target < num_qubits and target > i:
                full_matrix[i, i], full_matrix[i, target] = matrix[0, 0], matrix[0, 1]
                full_matrix[target, i], full_matrix[target, target] = matrix[1, 0], matrix[1, 1]

        # 回転適用 (batch_size, num_qubits) x (num_qubits, num_qubits) -> (batch_size, num_qubits)
        return torch.matmul(state.to(torch.complex64), full_matrix)
    def optimize(self, probs):
        """量子回路最適化を実行"""
        # probsをテンソルに変換
        probs_tensor = torch.tensor(probs, dtype=torch.float32, device=self.rot_params.device)

        # forwardメソッドを呼び出して最適化された確率を取得
        optimized_probs = self.forward(probs_tensor)

        # テンソルをNumPy配列に変換して返す
        return optimized_probs.detach().cpu().numpy()

    def _apply_entanglement(self, state: torch.Tensor) -> torch.Tensor:
        """エンタングルメントゲートを適用 (3Dテンソル対応版)"""
        # state: (batch_size, num_qubits)
        batch_size, num_qubits = state.shape

        # 複素数に変換
        state = state.to(torch.complex64)

        # エンタングルメント行列を準備 (num_qubits, num_qubits)
        entanglement_matrix = torch.exp(1j * self.entanglement[:num_qubits, :num_qubits])

        # バッチ処理用に3D化 (batch_size, num_qubits, num_qubits)
        entanglement_matrix = entanglement_matrix.unsqueeze(0).expand(batch_size, -1, -1)

        # 状態を3D化 (batch_size, 1, num_qubits)
        state_3d = state.unsqueeze(1)

        # エンタングルメント適用 (batch_size, 1, num_qubits) x (batch_size, num_qubits, num_qubits) -> (batch_size, 1, num_qubits)
        entangled_state = torch.bmm(state_3d, entanglement_matrix)

        # 元の形状に戻す (batch_size, num_qubits)
        return entangled_state.squeeze(1)

    def _apply_rotation(self, state, rot_mat):
        """量子回転を適用 (3Dテンソル対応)"""
        # Convert the state to a complex tensor if it's not already
        if not torch.is_complex(state):
            state = state.type(torch.complex64) #changed to complex64

        # Apply the rotation to each element of the state tensor
        rotated_state = torch.zeros_like(state, dtype=torch.complex64)
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                # Reshape state[i, j] to a 2D tensor before applying matmul
                # The error is here. state[i, j] is a scalar,
                # and you can't reshape it to (2, 1).
                # Instead, just multiply the scalar by the rotation matrix.
                rotated_state[i, j] = (state[i, j] * rot_mat).sum() #removed item() to avoid recursion error


        return rotated_state
    def _apply_quantum_circuit(self, probs: torch.Tensor, head_idx: int) -> torch.Tensor:
        """量子回路を確率分布に適用"""
        batch_size, seq_len = probs.shape # probsのshapeを修正
        amplitudes = torch.sqrt(probs)  # 確率振幅に変換
        # 量子回路適用
        for layer in range(self.quantum_layers):
            # 回転ゲート適用
            angles = self.quantum_weights[layer, head_idx]
            rot_mat = self._rotation_matrix(angles)
            amplitudes = self._apply_rotation(amplitudes, rot_mat) #_apply_rotationを修正
            # エンタングルメント適用
            amplitudes = self._apply_entanglement(amplitudes, head_idx, num_qubits=37) # エンタングルメント適用, num_qubitsを渡す
        probabilities = torch.abs(amplitudes)**2
        return probabilities

class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1, verbose=False):
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0
        self.verbose = verbose

        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_lr

    def get_lr(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def print_lr(self, is_verbose, lr):
        """Display the current learning rate.
        """
        if is_verbose:
            print('Adjusting learning rate'
                  ' to {}'.format(lr))

    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values = self._get_closed_form_lr()
                else:
                    values = self.get_lr()

        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
class CosineAnnealingWarmRestarts(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)

    When last_epoch=-1, sets initial lr as lr.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart.
        T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.
        verbose (bool, optional): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1708.07120
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """Step could be called after every batch update

        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> iters = len(dataloader)
            >>> for epoch in range(20):
            >>>     for i, sample in enumerate(dataloader):
            >>>         scheduler.step(epoch + i / iters)
            >>>         train(...)

        Args:
            epoch (int, optional): The epoch number. Default: None.
        """
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                self.T_cur = epoch - self.T_0
                self.T_i = self.T_0 * self.T_mult ** ((epoch - self.T_0) // (self.T_0 * (self.T_mult - 1)))
                if (epoch - self.T_0) % (self.T_0 * (self.T_mult - 1)) == 0:
                    self.T_i = self.T_i / self.T_mult
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.last_epoch = math.floor(epoch)
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        self.print_lr(self.verbose, self.get_lr())

        self._last_lr = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
        self._get_lr_called_within_step = True
        return self._last_lr
class PositionalEncoding(nn.Module):
    """位置エンコーディング"""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        position = torch.arange(seq_len, device=x.device, dtype=torch.float32).unsqueeze(1)
        pe = torch.zeros(1, seq_len, self.d_model, device=x.device, dtype=torch.float32)
        pe[0, :, 0::2] = torch.sin(position * self.div_term)
        pe[0, :, 1::2] = torch.cos(position * self.div_term)
        return pe


def main():
    # 設定
    config = {
        "csv_url": "https://loto7.thekyo.jp/data/loto7.csv",
        "predict_numbers":[26,30,618,619,620,621,622,623,624,625],
        "window_size": 50,
        "batch_size": 1024,
        "num_epochs": 1000,
        "d_model": 128,
        "nhead": 1,
        "num_layers": 1,
        "dropout": 0.1,
        "use_quantum": True,
        "quantum_layers": 15,
        "lr": 4e-9,
        "weight_decay": 1e-8,
        "result_dir": "results_loto7",
        "checkpoint_path": "checkpoints/best_model_loto7.pth",
        "mode": "train",  # "train", "retrain", "predict" のいずれかを指定
        "num_future_draws": 10, # 予測モードでの予測回数
        "use_hyperparameter_tuning": False, # ハイパーパラメータチューニングを行うかどうか
        "num_trials": 10 #optunaの試行回数
    }
    os.makedirs(config["result_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(config["checkpoint_path"]), exist_ok=True)

    dataset = EnhancedLotoDataset(config["csv_url"], config["window_size"])
    predict_numbers = config["predict_numbers"]  # Store predict_numbers separately

    # Access feature_dim directly from the dataset object:
    if dataset.feature_dim == 0:
        raise ValueError("Failed to compute feature dimensions - check data loading")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=0,  # Or lower if already reduced
            persistent_workers=False  # Disable persistent workers
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        num_workers=2
    )
    print(f"Loaded dataset with {len(dataset)} samples, feature dim: {dataset.feature_dim}, max draw num: {dataset.max_draw_num}")

    # モデル初期化
    if config["mode"] == "retrain" and os.path.exists(config["checkpoint_path"]):
        model = AdvancedLotoPredictor(
            feature_dim=dataset.feature_dim,
            window_size=config["window_size"],
            d_model=config["d_model"],
            num_heads=config["nhead"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            quantum_layers=config["quantum_layers"],
            use_quantum=config["use_quantum"],
            dataset=dataset
        )
        model.load_state_dict(torch.load(config["checkpoint_path"])['model_state_dict'])
        logger.info("Loaded model from checkpoint for retraining.")
    else:
        model = AdvancedLotoPredictor(
            feature_dim=dataset.feature_dim,
            window_size=config["window_size"],
            d_model=config["d_model"],
            num_heads=config["nhead"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            quantum_layers=config["quantum_layers"],
            use_quantum=config["use_quantum"],
            dataset=dataset
        )
        logger.info("Initialized a new model.")

    if config["use_hyperparameter_tuning"]:
        tuner = HyperparameterTuner(dataset)
        best_params = tuner.tune()
        logger.info(f"Best Hyperparameters: {best_params}")
        #configをbest_paramsで更新
        config.update(best_params)
        # チューニングされたハイパーパラメータでモデルを再定義
        model = AdvancedLotoPredictor(
            feature_dim=dataset.feature_dim,
            window_size=config["window_size"],
            d_model=config["d_model"],
            num_heads=config["nhead"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            quantum_layers=config["quantum_layers"],
            use_quantum=config["use_quantum"],
            dataset=dataset
        )

    # モードに応じた処理
    if config["mode"] in ["train", "retrain"]:
        trainer = QuantumTrainer(model, train_loader, val_loader, config)
        trainer.fit(config,config["num_epochs"])
        trainer.close()
    elif config["mode"] == "predict":
        future_predictor = FutureLotoPredictor(model, dataset, config["num_future_draws"])
        future_predictions = future_predictor.predict_future_sequence()
        print("Future Predictions:")
        for i, (numbers, confidence) in enumerate(future_predictions):
            print(f"Draw {dataset.max_draw_num + i + 1}: Numbers={numbers}, Confidence={confidence:.4f}")
    else:
        logger.error(f"Invalid mode: {config['mode']}")

if __name__ == "__main__":
    main()
