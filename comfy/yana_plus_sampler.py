"""
Yana Plus (雅娜+) - Optimized Riemannian Centroid Diffusion Sampler (RCDS+)
修復版本：解決噪點問題、錯誤估計和數值穩定性問題

修復的主要問題：
1. 錯誤估計計算不準確導致的噪點問題 - 使用更穩定的相對誤差估計
2. 自適應擾動強度計算錯誤 - 限制擾動強度範圍和增長率
3. 數值穩定性問題 - 添加容錯機制和回退策略
4. 快速質心計算的精度問題 - 改進加速算法和收斂條件
"""

import torch
import math
import numpy as np
from typing import Optional, Callable, Union, Dict, Any
from tqdm.auto import tqdm
import logging
from .yana_sampler import (
    RiemannianManifold, 
    RiemannianCentroidCalculator, 
    UnifiedPredictor,
    append_dims,
    to_d
)


class ErrorEstimator:
    """Estimates local error for adaptive branching - 修復版本"""
    
    def __init__(self, model):
        self.model = model
        self.cached_predictions = {}
        self.prediction_count = 0
    
    def estimate_local_error(self, x_t: torch.Tensor, t: float, delta_t: float, extra_args: dict = None) -> float:
        """
        修復版本的錯誤估計
        使用更穩定的方法來估計局部截斷誤差
        """
        if extra_args is None:
            extra_args = {}
        
        # 避免過小的時間步長
        if abs(delta_t) < 1e-6:
            return 0.001  # 返回小的默認錯誤
        
        # 使用更小的擾動來估計錯誤
        eps_t = min(abs(delta_t) * 0.01, 0.001)
        
        try:
            # 獲取當前時間的預測
            s_in = x_t.new_ones([x_t.shape[0]])
            sigma_t = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=x_t.dtype)
            
            # 緩存管理
            cache_key = (self.prediction_count, t)
            if cache_key not in self.cached_predictions:
                pred_t = self.model(x_t, sigma_t * s_in, **extra_args)
                self.cached_predictions[cache_key] = pred_t
                # 限制緩存大小
                if len(self.cached_predictions) > 10:
                    oldest_key = min(self.cached_predictions.keys())
                    del self.cached_predictions[oldest_key]
            else:
                pred_t = self.cached_predictions[cache_key]
            
            # 獲取稍後時間的預測
            sigma_t_eps = torch.full((x_t.shape[0],), t + eps_t, device=x_t.device, dtype=x_t.dtype)
            pred_t_eps = self.model(x_t, sigma_t_eps * s_in, **extra_args)
            
            # 計算預測差異
            pred_diff = pred_t - pred_t_eps
            
            # 使用相對誤差而不是絕對誤差
            pred_norm = torch.norm(pred_t).item()
            diff_norm = torch.norm(pred_diff).item()
            
            if pred_norm > 1e-8:
                relative_error = diff_norm / pred_norm
            else:
                relative_error = diff_norm
            
            # 按時間步長縮放
            error_per_step = relative_error * abs(delta_t) / eps_t
            
            # 限制錯誤範圍避免極端值
            error_per_step = max(1e-6, min(error_per_step, 1.0))
            
            self.prediction_count += 1
            return error_per_step
            
        except Exception as e:
            logging.warning(f"Error estimation failed: {e}, returning default error")
            return 0.001  # 返回安全的默認值
    
    def clear_cache(self):
        """清除預測緩存"""
        self.cached_predictions.clear()


class FastCentroidCalculator:
    """Fast centroid calculator - 修復版本"""
    
    def __init__(self, manifold: RiemannianManifold, max_iterations: int = 10, tolerance: float = 1e-6):
        self.manifold = manifold
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def compute_centroid_fast(self, points: torch.Tensor, weights: Optional[torch.Tensor] = None, 
                            error_threshold: float = 1e-6) -> torch.Tensor:
        """
        修復版本的快速質心計算
        改進了數值穩定性和精度
        """
        K = points.shape[0]
        
        if weights is None:
            weights = torch.ones(K, device=points.device, dtype=points.dtype) / K
        else:
            weights = weights / (torch.sum(weights) + 1e-10)
        
        if self.manifold.manifold_type == "euclidean":
            # 歐幾里得空間的快速路徑
            weight_dims = [1] * (points.ndim - 1)
            centroid = torch.sum(weights.view(-1, *weight_dims) * points, dim=0)
            return centroid
        
        elif self.manifold.manifold_type == "sphere":
            # 球面的改進快速近似
            weight_dims = [1] * (points.ndim - 1)
            centroid_approx = torch.sum(weights.view(-1, *weight_dims) * points, dim=0)
            
            # 投影到球面
            centroid_projected = self.manifold.project_to_manifold(centroid_approx)
            
            # 進行幾次迭代改進
            for _ in range(min(3, self.max_iterations)):
                gradient = torch.zeros_like(centroid_projected)
                for k in range(K):
                    log_vec = self.manifold.log_map(centroid_projected, points[k])
                    gradient += weights[k] * log_vec
                
                if torch.norm(gradient) < self.tolerance:
                    break
                
                # 小步長更新
                step_size = 0.3
                centroid_projected = self.manifold.exp_map(centroid_projected, -step_size * gradient)
                centroid_projected = self.manifold.project_to_manifold(centroid_projected)
            
            return centroid_projected
        
        else:
            # 一般情況：加速的 Weiszfeld 算法
            return self._accelerated_weiszfeld_fixed(points, weights, error_threshold)
    
    def _accelerated_weiszfeld_fixed(self, points: torch.Tensor, weights: torch.Tensor, 
                                   error_threshold: float) -> torch.Tensor:
        """
        修復版本的加速 Weiszfeld 算法
        """
        K = points.shape[0]
        
        # 初始化為加權平均
        weight_dims = [1] * (points.ndim - 1)
        z = torch.sum(weights.view(-1, *weight_dims) * points, dim=0)
        z = self.manifold.project_to_manifold(z)
        
        prev_z = z.clone()
        
        for iteration in range(self.max_iterations):
            gradient = torch.zeros_like(z)
            total_weight = 0.0
            
            for k in range(K):
                try:
                    dist = self.manifold.geodesic_distance(z.unsqueeze(0), points[k].unsqueeze(0))
                    dist_safe = torch.clamp(dist, min=1e-8)
                    
                    log_vec = self.manifold.log_map(z, points[k])
                    weight_k = weights[k] / (dist_safe + 1e-8)
                    
                    gradient += weight_k * log_vec
                    total_weight += weight_k
                    
                except Exception as e:
                    # 如果計算失敗，跳過這個點
                    continue
            
            # 歸一化梯度
            if total_weight > 1e-8:
                gradient = gradient / total_weight
            
            # 自適應步長
            step_size = min(0.5, 1.0 / (iteration + 1))
            
            # 動量項（改進的版本）
            if iteration > 0:
                momentum = 0.7
                momentum_term = momentum * (z - prev_z)
                gradient = momentum_term + (1 - momentum) * gradient
            
            prev_z = z.clone()
            
            # 更新使用指數映射
            try:
                z = self.manifold.exp_map(z, -step_size * gradient)
                z = self.manifold.project_to_manifold(z)
            except Exception as e:
                # 如果更新失敗，使用歐幾里得更新
                z = z - step_size * gradient
                z = self.manifold.project_to_manifold(z)
            
            # 檢查收斂
            if torch.norm(gradient) < self.tolerance:
                break
        
        return z


class AdaptiveStepController:
    """Controls adaptive step size - 修復版本"""
    
    def __init__(self, initial_dt: float, min_dt: float = 1e-5, max_dt: float = 1.0):
        self.initial_dt = initial_dt
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.step_history = []
        self.error_history = []
    
    def adjust_step_size(self, current_dt: float, error: float, threshold: float) -> float:
        """
        修復版本的自適應步長調整
        更保守的調整策略
        """
        # 記錄歷史
        self.step_history.append(current_dt)
        self.error_history.append(error)
        
        # 更保守的調整因子
        if error > threshold * 2.0:
            # 錯誤太大，減小步長
            new_dt = max(current_dt * 0.8, self.min_dt)
        elif error < threshold * 0.5:
            # 錯誤很小，可以增加步長
            new_dt = min(current_dt * 1.1, self.max_dt)
        else:
            # 錯誤可接受，保持當前步長
            new_dt = current_dt
        
        # 限制步長變化幅度
        max_change = 0.2
        change_ratio = new_dt / current_dt
        if change_ratio > (1 + max_change):
            new_dt = current_dt * (1 + max_change)
        elif change_ratio < (1 - max_change):
            new_dt = current_dt * (1 - max_change)
        
        return new_dt


@torch.no_grad()
def sample_yana_plus(
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: Optional[Dict[str, Any]] = None,
    callback: Optional[Callable] = None,
    disable: Optional[bool] = None,
    initial_centroids: int = 3,
    max_centroids: int = 6,
    manifold_type: str = "euclidean",
    perturbation_strength: float = 0.05,  # 調整默認值
    error_threshold: float = 0.01,        # 調整默認值
    adaptive_branching: bool = True,
    adaptive_step_size: bool = False,     # 默認關閉，避免不穩定
    fast_centroid: bool = True,
    fallback_to_euler: bool = True,
    **kwargs
) -> torch.Tensor:
    """
    RCDS+ (Optimized Riemannian Centroid Diffusion Sampler) - 修復版本
    
    修復的主要問題：
    1. 錯誤估計計算更準確
    2. 自適應擾動強度更合理
    3. 數值穩定性改進
    4. 快速質心計算精度提升
    """
    extra_args = extra_args or {}
    
    # 初始化組件
    manifold = RiemannianManifold(manifold_type)
    
    if fast_centroid:
        centroid_calculator = FastCentroidCalculator(manifold, max_iterations=10)
    else:
        centroid_calculator = RiemannianCentroidCalculator(manifold, max_iterations=15)
    
    predictor = UnifiedPredictor(model)
    error_estimator = ErrorEstimator(model) if adaptive_branching else None
    
    # 初始化自適應步長控制器
    if adaptive_step_size and len(sigmas) > 1:
        initial_dt = float(sigmas[0] - sigmas[1])
        step_controller = AdaptiveStepController(initial_dt)
    else:
        step_controller = None
    
    # 確保合理的參數範圍
    initial_centroids = max(1, min(initial_centroids, max_centroids))
    max_centroids = min(max_centroids, 8)  # 限制最大值
    
    # 主要 RCDS+ 算法
    x_current = x
    
    # 統計信息
    stats = {
        'adaptive_k_usage': [],
        'error_estimates': [],
        'fallback_count': 0,
        'numerical_issues': 0
    }
    
    for i in tqdm(range(len(sigmas) - 1), disable=disable, desc="RCDS+ Sampling"):
        sigma_t = sigmas[i]
        sigma_next = sigmas[i + 1]
        
        if sigma_t == 0:
            continue
        
        # 轉換為連續時間表示
        t = float(sigma_t)
        delta_t = float(sigma_t - sigma_next)
        
        # 自適應步長控制（如果啟用）
        if step_controller is not None:
            last_error = stats['error_estimates'][-1] if stats['error_estimates'] else 0.0
            delta_t = step_controller.adjust_step_size(delta_t, last_error, error_threshold)
        
        # 步驟 1：自適應分支 - 估計錯誤並選擇 K
        if adaptive_branching and error_estimator:
            try:
                local_error = error_estimator.estimate_local_error(x_current, t, delta_t, extra_args)
                stats['error_estimates'].append(local_error)
                
                # 更保守的 K 選擇策略
                if local_error > error_threshold * 2.0:
                    K = min(initial_centroids + 1, max_centroids)
                elif local_error < error_threshold * 0.2:
                    K = max(1, initial_centroids - 1)
                else:
                    K = initial_centroids
                
                # 改進的自適應擾動強度
                noise_factor = t / float(sigmas[0])  # 歸一化噪聲因子
                error_factor = min(math.sqrt(local_error), 1.0)  # 限制錯誤因子
                effective_perturbation = perturbation_strength * noise_factor * error_factor * 0.9
                
                # 確保擾動強度在合理範圍內
                effective_perturbation = max(effective_perturbation, perturbation_strength * 0.1)
                effective_perturbation = min(effective_perturbation, perturbation_strength * 2.0)
                
            except Exception as e:
                # 錯誤估計失敗時的回退
                K = initial_centroids
                effective_perturbation = perturbation_strength
                local_error = error_threshold
                stats['numerical_issues'] += 1
        else:
            K = initial_centroids
            effective_perturbation = perturbation_strength
            local_error = 0.0
        
        stats['adaptive_k_usage'].append(K)
        
        # 步驟 2：處理回退到標準採樣器
        if K == 1 and fallback_to_euler:
            # 回退到標準歐拉步驟
            s_in = x_current.new_ones([x_current.shape[0]])
            sigma_tensor = torch.full((x_current.shape[0],), t, device=x_current.device, dtype=x_current.dtype)
            
            denoised = model(x_current, sigma_tensor * s_in, **extra_args)
            d = to_d(x_current, sigma_tensor, denoised)
            x_next = x_current + d * (-delta_t)
            
            stats['fallback_count'] += 1
            
        else:
            # 步驟 3：生成 K 個擾動
            perturbed_points = []
            for k in range(K):
                if k == 0:
                    # 第一個分支：無擾動
                    x_t_k = x_current
                else:
                    # 生成擾動
                    eta_k = manifold.random_tangent_vector(x_current, effective_perturbation)
                    x_t_k = manifold.exp_map(x_current, eta_k)
                    x_t_k = manifold.project_to_manifold(x_t_k)
                
                perturbed_points.append(x_t_k)
            
            # 步驟 4：獲取預測
            predictions = []
            for k in range(K):
                x_t_k = perturbed_points[k]
                try:
                    y_k = predictor.predict_next(x_t_k, t, delta_t, extra_args)
                    predictions.append(y_k)
                except Exception as e:
                    logging.warning(f"Prediction failed for branch {k}: {e}")
                    predictions.append(x_t_k)  # 回退到原始點
            
            # 步驟 5：快速質心計算
            predictions_tensor = torch.stack(predictions, dim=0)
            
            # 改進的權重策略
            weights = torch.ones(K, device=x.device, dtype=x.dtype)
            weights[0] = 1.5  # 給無擾動分支稍高權重
            
            try:
                if fast_centroid:
                    x_next = centroid_calculator.compute_centroid_fast(predictions_tensor, weights, error_threshold)
                else:
                    x_next = centroid_calculator.compute_centroid(predictions_tensor, weights)
                
                # 數值穩定性檢查
                if torch.any(torch.isnan(x_next)) or torch.any(torch.isinf(x_next)):
                    x_next = predictions[0]  # 回退到第一個預測
                    stats['numerical_issues'] += 1
                    
            except Exception as e:
                logging.warning(f"Centroid calculation failed: {e}")
                x_next = predictions[0]  # 回退到第一個預測
                stats['numerical_issues'] += 1
        
        # 應用回調函數
        if callback is not None:
            callback({
                'x': x_current,
                'i': i,
                'sigma': sigma_t,
                'sigma_hat': sigma_t,
                'denoised': x_next,
                'adaptive_k': K,
                'error_estimate': local_error
            })
        
        x_current = x_next
        
        # 定期清理錯誤估計器緩存
        if error_estimator and i % 5 == 0:
            error_estimator.clear_cache()
    
    # 記錄最終統計
    if stats['adaptive_k_usage']:
        avg_k = np.mean(stats['adaptive_k_usage'])
        avg_error = np.mean(stats['error_estimates']) if stats['error_estimates'] else 0.0
        logging.info(f"RCDS+ completed. Avg K: {avg_k:.2f}, Avg error: {avg_error:.6f}, "
                    f"Fallbacks: {stats['fallback_count']}, Issues: {stats['numerical_issues']}")
    
    return x_current


# 向後兼容的類包裝器
class YanaPlusSampler:
    """Fixed version wrapper for backward compatibility"""
    
    def __init__(self, manifold_type="euclidean", initial_centroids=3, **kwargs):
        self.manifold_type = manifold_type
        self.initial_centroids = initial_centroids
        self.kwargs = kwargs