"""
Yana (雅娜) Sampler - Riemannian Centroid Diffusion Sampler (RCDS)
修復版本：解決模糊、數值穩定性和擾動強度問題

修復的主要問題：
1. 擾動強度設定不合理 - 增加預設值並改進自適應計算
2. 數值穩定性問題 - 添加穩定性檢查和容錯機制
3. 自適應擾動計算錯誤 - 使用更合理的噪聲水平歸一化
4. 質心計算精度問題 - 改進迭代算法和收斂條件
"""

import torch
import math
import numpy as np
from typing import Optional, Callable, Union, Dict, Any
from tqdm.auto import tqdm
import logging


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    expanded = x[(...,) + (None,) * dims_to_append]
    return expanded.detach().clone() if expanded.device.type == 'mps' else expanded


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


class RiemannianManifold:
    """Riemannian manifold operations for RCDS - 修復版本"""
    
    def __init__(self, manifold_type="euclidean"):
        self.manifold_type = manifold_type
        self.eps = 1e-8  # 數值穩定性常數
    
    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map: Exp_x(v) maps tangent vector v at point x to the manifold"""
        if self.manifold_type == "euclidean":
            return x + v
        elif self.manifold_type == "sphere":
            # 改進的球面指數映射，更穩定
            original_shape = x.shape
            x_flat = x.view(x.shape[0], -1)
            v_flat = v.view(v.shape[0], -1)
            
            v_norm = torch.norm(v_flat, dim=1, keepdim=True)
            # 避免除零和數值不穩定
            v_norm_safe = torch.clamp(v_norm, min=self.eps)
            v_normalized = v_flat / v_norm_safe
            
            # 使用更穩定的公式
            cos_norm = torch.cos(v_norm)
            sin_norm = torch.sin(v_norm)
            
            # 處理小角度情況
            small_angle_mask = v_norm < 1e-4
            
            result = torch.where(
                small_angle_mask,
                x_flat + v_flat,  # 線性近似
                x_flat * cos_norm + v_normalized * sin_norm
            )
            
            return result.view(original_shape)
        else:
            return x + v
    
    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Logarithmic map: Log_x(y) maps point y to tangent space T_x M at x"""
        if self.manifold_type == "euclidean":
            return y - x
        elif self.manifold_type == "sphere":
            # 改進的球面對數映射
            original_shape = x.shape
            x_flat = x.view(x.shape[0], -1)
            y_flat = y.view(y.shape[0], -1)
            
            # 確保輸入在球面上
            x_flat = x_flat / (torch.norm(x_flat, dim=1, keepdim=True) + self.eps)
            y_flat = y_flat / (torch.norm(y_flat, dim=1, keepdim=True) + self.eps)
            
            dot_product = torch.sum(x_flat * y_flat, dim=1, keepdim=True)
            dot_product = torch.clamp(dot_product, -1.0 + self.eps, 1.0 - self.eps)
            
            theta = torch.acos(dot_product)
            sin_theta = torch.sin(theta)
            
            # 更穩定的小角度處理
            small_angle_mask = sin_theta.abs() < 1e-6
            
            result = torch.where(
                small_angle_mask,
                y_flat - x_flat * dot_product,
                theta * (y_flat - x_flat * dot_product) / sin_theta
            )
            
            return result.view(original_shape)
        else:
            return y - x
    
    def geodesic_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute geodesic distance d_g(x, y) between two points"""
        if self.manifold_type == "euclidean":
            x_flat = x.view(x.shape[0], -1)
            y_flat = y.view(y.shape[0], -1)
            return torch.norm(x_flat - y_flat, dim=1)
        elif self.manifold_type == "sphere":
            x_flat = x.view(x.shape[0], -1)
            y_flat = y.view(y.shape[0], -1)
            
            # 正規化到單位球面
            x_flat = x_flat / (torch.norm(x_flat, dim=1, keepdim=True) + self.eps)
            y_flat = y_flat / (torch.norm(y_flat, dim=1, keepdim=True) + self.eps)
            
            dot_product = torch.sum(x_flat * y_flat, dim=1)
            dot_product = torch.clamp(dot_product, -1.0 + self.eps, 1.0 - self.eps)
            return torch.acos(dot_product)
        else:
            x_flat = x.view(x.shape[0], -1)
            y_flat = y.view(y.shape[0], -1)
            return torch.norm(x_flat - y_flat, dim=1)
    
    def project_to_manifold(self, x: torch.Tensor) -> torch.Tensor:
        """Project point to the manifold"""
        if self.manifold_type == "euclidean":
            return x
        elif self.manifold_type == "sphere":
            original_shape = x.shape
            x_flat = x.view(x.shape[0], -1)
            norm = torch.norm(x_flat, dim=1, keepdim=True)
            x_normalized = x_flat / (norm + self.eps)
            return x_normalized.view(original_shape)
        else:
            return x
    
    def random_tangent_vector(self, x: torch.Tensor, std: float = 1.0) -> torch.Tensor:
        """Generate random vector in tangent space T_x M - 修復版本"""
        if self.manifold_type == "euclidean":
            # 使用合適的標準差，避免擾動過小
            return torch.randn_like(x) * std
        elif self.manifold_type == "sphere":
            # 改進的球面切空間隨機向量
            v = torch.randn_like(x) * std
            
            # 投影到切空間
            original_shape = x.shape
            x_flat = x.view(x.shape[0], -1)
            v_flat = v.view(v.shape[0], -1)
            
            # 確保 x 在球面上
            x_flat = x_flat / (torch.norm(x_flat, dim=1, keepdim=True) + self.eps)
            
            # 投影：v_tangent = v - ⟨v, x⟩x
            dot_product = torch.sum(v_flat * x_flat, dim=1, keepdim=True)
            v_tangent = v_flat - dot_product * x_flat
            
            return v_tangent.view(original_shape)
        else:
            return torch.randn_like(x) * std


class RiemannianCentroidCalculator:
    """Calculator for Riemannian centroid (Fréchet mean) - 修復版本"""
    
    def __init__(self, manifold: RiemannianManifold, max_iterations: int = 20, tolerance: float = 1e-8):
        self.manifold = manifold
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def compute_centroid(self, points: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute Riemannian centroid G = argmin_z (1/K) Σ w_k d_g(y_k, z)^2
        修復版本：改進數值穩定性和收斂性
        """
        K = points.shape[0]
        
        if weights is None:
            weights = torch.ones(K, device=points.device, dtype=points.dtype) / K
        else:
            weights = weights / (torch.sum(weights) + self.manifold.eps)
        
        if self.manifold.manifold_type == "euclidean":
            # 歐幾里得空間的閉式解
            weight_dims = [1] * (points.ndim - 1)
            centroid = torch.sum(weights.view(-1, *weight_dims) * points, dim=0)
            return centroid
        
        # 非歐幾里得流形：改進的黎曼梯度下降
        z = points[0].clone()
        
        for iteration in range(self.max_iterations):
            gradient = torch.zeros_like(z)
            
            for k in range(K):
                log_vec = self.manifold.log_map(z, points[k])
                gradient += weights[k] * log_vec
            
            grad_norm = torch.norm(gradient)
            if grad_norm < self.tolerance:
                break
            
            # 改進的自適應步長
            step_size = min(1.0, 2.0 / (iteration + 2))  # 更保守的步長
            
            # 應用更新
            z = self.manifold.exp_map(z, -step_size * gradient)
            z = self.manifold.project_to_manifold(z)
        
        return z


class UnifiedPredictor:
    """Unified predict_next function for different training paradigms - 修復版本"""
    
    def __init__(self, model):
        self.model = model
    
    def predict_next(self, x_t: torch.Tensor, t: float, delta_t: float, extra_args: dict = None) -> torch.Tensor:
        """
        Unified predict_next function - 修復版本
        改進了數值穩定性和精度
        """
        if extra_args is None:
            extra_args = {}
        
        # 確保數值穩定性
        if abs(delta_t) < 1e-8:
            return x_t
        
        s_in = x_t.new_ones([x_t.shape[0]])
        sigma = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=x_t.dtype)
        
        # 獲取模型預測
        model_output = self.model(x_t, sigma * s_in, **extra_args)
        
        # 計算導數（k-diffusion 風格）
        d = to_d(x_t, sigma, model_output)
        
        # 歐拉步驟到下一時間
        dt = -delta_t  # 負號因為我們在時間上往後退
        x_next = x_t + d * dt
        
        return x_next


@torch.no_grad()
def sample_yana(
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: Optional[Dict[str, Any]] = None,
    callback: Optional[Callable] = None,
    disable: Optional[bool] = None,
    num_centroids: int = 3,
    manifold_type: str = "euclidean",
    perturbation_strength: float = 0.1,  # 增加預設值
    adaptive_perturbation: bool = True,
    geometric_interpolation: str = "riemannian",
    **kwargs
) -> torch.Tensor:
    """
    RCDS (Riemannian Centroid Diffusion Sampler) - 修復版本
    
    修復的主要問題：
    1. 更合理的擾動強度設定
    2. 改進的數值穩定性
    3. 更好的自適應擾動計算
    4. 改進的質心計算精度
    """
    extra_args = extra_args or {}
    
    # 初始化流形和質心計算器
    manifold = RiemannianManifold(manifold_type)
    centroid_calculator = RiemannianCentroidCalculator(manifold, max_iterations=20, tolerance=1e-8)
    predictor = UnifiedPredictor(model)
    
    # 確保合理的質心數量
    num_centroids = max(2, min(num_centroids, 8))
    
    # 主要 RCDS 算法
    x_current = x
    
    for i in tqdm(range(len(sigmas) - 1), disable=disable, desc="RCDS Sampling"):
        sigma_t = sigmas[i]
        sigma_next = sigmas[i + 1]
        
        if sigma_t == 0:
            continue
        
        # 轉換為連續時間表示
        t = float(sigma_t)
        delta_t = float(sigma_t - sigma_next)
        
        # 改進的自適應擾動強度
        if adaptive_perturbation:
            # 基於當前噪聲水平調整，更合理的公式
            noise_level = t / float(sigmas[0])  # 歸一化噪聲水平
            sigma_perturb = perturbation_strength * noise_level * 0.95  # 穩定性因子
            # 避免過小的擾動
            sigma_perturb = max(sigma_perturb, perturbation_strength * 0.1)
        else:
            sigma_perturb = perturbation_strength
        
        # 步驟 1：生成 K 個擾動並計算擾動點
        perturbed_points = []
        for k in range(num_centroids):
            if k == 0:
                # 第一個分支：無擾動
                x_t_k = x_current
            else:
                # 生成擾動 η_k ~ N(0, σ_perturb^2 I) 在切空間中
                eta_k = manifold.random_tangent_vector(x_current, sigma_perturb)
                # 計算擾動點 x_t^{(k)} = Exp_{x_t}(η_k)
                x_t_k = manifold.exp_map(x_current, eta_k)
                x_t_k = manifold.project_to_manifold(x_t_k)
            
            perturbed_points.append(x_t_k)
        
        # 步驟 2：獲取每個擾動點的預測
        predictions = []
        for k in range(num_centroids):
            x_t_k = perturbed_points[k]
            # y_k = predict_next(x_t^{(k)}, t, Δt)
            y_k = predictor.predict_next(x_t_k, t, delta_t, extra_args)
            predictions.append(y_k)
        
        # 步驟 3：計算黎曼質心
        predictions_tensor = torch.stack(predictions, dim=0)
        
        # 添加權重以改進質心計算
        weights = torch.ones(num_centroids, device=x.device, dtype=x.dtype)
        # 給第一個分支（無擾動）更高權重
        weights[0] = 2.0
        
        x_next = centroid_calculator.compute_centroid(predictions_tensor, weights)
        
        # 數值穩定性檢查
        if torch.any(torch.isnan(x_next)) or torch.any(torch.isinf(x_next)):
            logging.warning(f"NaN or Inf detected at step {i}, falling back to first prediction")
            x_next = predictions[0]
        
        # 應用回調函數
        if callback is not None:
            callback({
                'x': x_current,
                'i': i,
                'sigma': sigma_t,
                'sigma_hat': sigma_t,
                'denoised': x_next
            })
        
        x_current = x_next
    
    return x_current


# 向後兼容的類包裝器
class YanaSampler:
    """Fixed version wrapper for backward compatibility"""
    
    def __init__(self, manifold_type="euclidean", num_centroids=3, **kwargs):
        self.manifold_type = manifold_type
        self.num_centroids = num_centroids
        self.kwargs = kwargs