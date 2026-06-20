
import torch
import torch.nn.functional as F
from typing import Optional, Literal, Tuple, List, Dict
import math
from collections import deque


class SmartEarlyStopping:
    
    def __init__(
        self,
        patience: int = 30,
        warmup: int = 50,
        rel_tol: float = 1e-4,
        grad_tol: float = 1e-6,
        ema_alpha: float = 0.1,
        oscillation_window: int = 20,
        min_steps: int = 100,
    ):
        self.patience = patience
        self.warmup = warmup
        self.rel_tol = rel_tol
        self.grad_tol = grad_tol
        self.ema_alpha = ema_alpha
        self.oscillation_window = oscillation_window
        self.min_steps = min_steps
        
        self.best_loss = float('inf')
        self.best_step = 0
        self.patience_counter = 0
        self.loss_history: deque = deque(maxlen=max(oscillation_window, 60))
        self.ema_loss = None
        self.grad_history: deque = deque(maxlen=10)
        self.stop_reason = None
        
        self.mse_history: deque = deque(maxlen=oscillation_window)
        self.nll_history: deque = deque(maxlen=oscillation_window)
        
    def step(
        self,
        loss: float,
        step: int,
        grad_norm: Optional[float] = None,
        mse_component: Optional[float] = None,
        nll_component: Optional[float] = None,
    ) -> bool:
        if step < self.warmup:
            self._update_state(loss, grad_norm, mse_component, nll_component)
            return False
        
        if step < self.min_steps:
            self._update_state(loss, grad_norm, mse_component, nll_component)
            return False
        
        if self.ema_loss is None:
            self.ema_loss = loss
        else:
            self.ema_loss = self.ema_alpha * loss + (1 - self.ema_alpha) * self.ema_loss
        
        if math.isinf(self.best_loss):
            self.best_loss = loss
            self.best_step = step
            self.patience_counter = 0
        else:
            if loss < self.best_loss * (1 - self.rel_tol):
                self.best_loss = loss
                self.best_step = step
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        
        early_factor = 1.5
        decay_steps = 100
        adaptive_patience = int(self.patience * (1 + early_factor * math.exp(-step / decay_steps)))
        
        if self.patience_counter >= adaptive_patience:
            self.stop_reason = f"no improvement for {adaptive_patience} steps"
            return True
        
        if grad_norm is not None:
            self.grad_history.append(grad_norm)
            if len(self.grad_history) >= 5:
                mean_grad = sum(self.grad_history) / len(self.grad_history)
                if mean_grad < self.grad_tol:
                    self.stop_reason = f"gradient vanished ({mean_grad:.2e})"
                    return True
        
        self.loss_history.append(loss)
        if len(self.loss_history) >= self.oscillation_window:
            losses = list(self.loss_history)
            diffs = [losses[i+1] - losses[i] for i in range(len(losses)-1)]
            sign_changes = sum(1 for i in range(len(diffs)-1) if diffs[i]*diffs[i+1] < 0)
            
            oscillation_rate = sign_changes / (len(diffs) - 1)
            if oscillation_rate > 0.7:
                amplitude = max(losses) - min(losses)
                if amplitude < abs(self.ema_loss) * 0.001:
                    self.stop_reason = f"converged oscillation (amp={amplitude:.2e})"
                    return True
        
        if len(self.loss_history) >= 50 and step > 150:
            losses = list(self.loss_history)
            loss_50_ago = losses[-50]
            loss_now = losses[-1]
            
            total_improvement = loss_50_ago - loss_now
            rel_improvement_per_step = total_improvement / (abs(loss_50_ago) + 1e-8) / 50
            
            if rel_improvement_per_step < 1e-5:
                self.stop_reason = f"diminishing returns ({rel_improvement_per_step*100:.4f}%/step)"
                return True
        
        self._update_state(loss, grad_norm, mse_component, nll_component)
        
        return False
    
    def _update_state(
        self,
        loss: float,
        grad_norm: Optional[float] = None,
        mse_component: Optional[float] = None,
        nll_component: Optional[float] = None,
    ):
        if mse_component is not None:
            self.mse_history.append(mse_component)
        if nll_component is not None:
            self.nll_history.append(nll_component)
        if grad_norm is not None:
            self.grad_history.append(grad_norm)
    
    def get_status(self) -> Dict:
        return {
            'best_loss': self.best_loss,
            'best_step': self.best_step,
            'patience_counter': self.patience_counter,
            'ema_loss': self.ema_loss,
            'stop_reason': self.stop_reason,
        }
    
    def reset(self):
        self.best_loss = float('inf')
        self.best_step = 0
        self.patience_counter = 0
        self.loss_history.clear()
        self.ema_loss = None
        self.grad_history.clear()
        self.stop_reason = None
        self.mse_history.clear()
        self.nll_history.clear()


def compute_grad_norm(params: Dict[str, torch.Tensor]) -> float:
    total_norm = 0.0
    for p in params.values():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total_norm)


def logI0_approx(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, min=0.0, max=700.0)
    
    x2 = x * x
    small_approx = torch.log1p(x2/4 + x2*x2/64 + x2*x2*x2/2304)
    
    large_approx = x - 0.5 * torch.log(2 * math.pi * x.clamp(min=0.1)) + torch.log1p(1/(8*x.clamp(min=0.1)))
    
    t = torch.sigmoid(2 * (x - 3.75))
    
    result = (1 - t) * small_approx + t * large_approx
    
    return result


def rician_nll_fast(pred: torch.Tensor, obs: torch.Tensor, 
                   sigma: torch.Tensor) -> torch.Tensor:
    sigma2 = sigma ** 2 + 1e-8
    
    z = obs * pred / sigma2
    
    nll = (torch.log(sigma2) + 
           (obs**2 + pred**2) / (2 * sigma2) - 
           torch.log(obs.clamp(min=1e-8)) - 
           logI0_approx(z))
    
    return nll


def charbonnier_loss(residual: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    r_scaled = residual / delta
    loss = delta**2 * (torch.sqrt(1 + r_scaled**2) - 1)
    return loss


def compute_shell_weights(bvals: torch.Tensor, 
                         shell_ids: torch.Tensor,
                         mode: Literal['inverse_var', 'inverse_std', 'uniform'] = 'inverse_var',
                         clamp_range: Tuple[float, float] = (0.1, 10.0)
                         ) -> torch.Tensor:
    if mode == 'uniform':
        return torch.ones_like(bvals)
    
    D_approx = 0.7e-3
    
    
    if mode == 'inverse_var':
        weights = torch.exp(-2 * bvals * D_approx)
    elif mode == 'inverse_std':
        weights = torch.exp(-bvals * D_approx)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    weights = weights / weights.mean()
    
    weights = torch.clamp(weights, min=clamp_range[0], max=clamp_range[1])
    
    return weights


def compute_shell_sigma_from_data(target: torch.Tensor,
                                   shell_ids: torch.Tensor,
                                   bvals: torch.Tensor,
                                   method: Literal['residual', 'rayleigh', 'background'] = 'residual'
                                   ) -> torch.Tensor:
    device = target.device
    n_meas = target.shape[-1]
    unique_shells = torch.unique(shell_ids)
    
    sigma_per_meas = torch.ones(n_meas, device=device) * 0.05
    
    for shell_id in unique_shells:
        shell_mask = shell_ids == shell_id
        shell_data = target[..., shell_mask]
        
        if method == 'residual':
            shell_mean = shell_data.mean(dim=-1, keepdim=True)
            residuals = shell_data - shell_mean
            sigma_shell = 1.4826 * residuals.abs().median()
        elif method == 'rayleigh':
            min_val = shell_data.min()
            sigma_shell = min_val / 1.2533
        else:
            sigma_shell = shell_data.flatten().quantile(0.05)
        
        sigma_per_meas[shell_mask] = sigma_shell.clamp(min=1e-6)
    
    return sigma_per_meas


def compute_shell_weights_from_sigma(sigma_per_meas: torch.Tensor,
                                      shell_ids: torch.Tensor,
                                      clamp_range: Tuple[float, float] = (0.1, 10.0),
                                      normalize: bool = True
                                      ) -> torch.Tensor:
    weights = 1.0 / (sigma_per_meas ** 2 + 1e-8)
    
    if normalize:
        weights = weights / weights.mean()
    
    weights = torch.clamp(weights, min=clamp_range[0], max=clamp_range[1])
    
    return weights


def rician_debias(y: torch.Tensor, sigma: torch.Tensor, 
                  eps: float = 1e-6) -> torch.Tensor:
    y_squared = y ** 2
    bias_term = 2 * sigma ** 2
    
    debiased_squared = torch.clamp(y_squared - bias_term, min=eps)
    
    return torch.sqrt(debiased_squared)


def loss_dmri(pred: torch.Tensor, 
              target: torch.Tensor,
              shell_ids: torch.Tensor,
              bvals: torch.Tensor,
              sigma: Optional[torch.Tensor] = None,
              shell_gains: Optional[torch.Tensor] = None,
              w_shell: Optional[torch.Tensor] = None,
              delta: float = 1.0,
              eps_floor: float = 1e-3,
              mode: Literal['huber', 'huber_debiased', 'huber_squared', 'squared_magnitude', 
                           'sqmag_charb', 'sqmag_charb_cosine', 'rician', 'rician_lite', 'mse'] = 'huber',
              shell_weight_mode: Literal['inverse_var', 'inverse_std', 'uniform'] = 'inverse_var',
              lambda_cosine: float = 0.05,
              clamp_pred: bool = True,
              clamp_factor: float = 1.5,
              reduction: Literal['mean', 'sum', 'none'] = 'mean',
              mask: Optional[torch.Tensor] = None
              ) -> Tuple[torch.Tensor, dict]:
    assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"
    assert shell_ids.shape[0] == pred.shape[-1], f"shell_ids length mismatch"
    
    device = pred.device
    n_measurements = pred.shape[-1]
    
    if shell_gains is not None:
        gains = shell_gains[shell_ids]
        for _ in range(pred.ndim - 1):
            gains = gains.unsqueeze(0)
        pred_norm = pred / (gains + 1e-8)
        target_norm = target / (gains + 1e-8)
    else:
        pred_norm = pred
        target_norm = target
    
    if clamp_pred:
        b0_mask = bvals < 100
        if b0_mask.any():
            S0 = target[..., b0_mask].mean(dim=-1, keepdim=True)
        else:
            S0 = target.max(dim=-1, keepdim=True).values
        
        max_val = clamp_factor * S0
        pred_norm = torch.clamp(pred_norm, min=0.0)
        pred_norm = torch.minimum(pred_norm, max_val)
    
    if sigma is None:
        b0_mask = bvals < 100
        if b0_mask.any():
            b0_residuals = (pred_norm[..., b0_mask] - target_norm[..., b0_mask])
            sigma = b0_residuals.std() + eps_floor
        else:
            sigma = eps_floor
    
    if not isinstance(sigma, torch.Tensor):
        sigma = torch.tensor(sigma, device=device)
    sigma = sigma.clamp(min=eps_floor)
    
    if w_shell is not None:
        weights = w_shell[shell_ids]
    else:
        weights = compute_shell_weights(bvals, shell_ids, mode=shell_weight_mode)
    
    weights = weights.to(device)
    for _ in range(pred.ndim - 1):
        weights = weights.unsqueeze(0)
    
    if mode == 'mse':
        loss_per_elem = (pred_norm - target_norm) ** 2
        
    elif mode == 'huber':
        residual = (pred_norm - target_norm) / (sigma + eps_floor)
        
        residual = torch.clamp(residual, min=-100, max=100)
        
        loss_per_elem = charbonnier_loss(residual, delta=delta)
    
    elif mode == 'huber_debiased':
        target_debiased = rician_debias(target_norm, sigma, eps=eps_floor)
        
        residual = (pred_norm - target_debiased) / (sigma + eps_floor)
        residual = torch.clamp(residual, min=-100, max=100)
        
        loss_per_elem = charbonnier_loss(residual, delta=delta)
    
    elif mode == 'huber_squared':
        pred_sq = pred_norm ** 2
        target_sq = target_norm ** 2
        
        sigma_sq = 2 * sigma ** 2
        residual = (pred_sq - target_sq) / (sigma_sq + eps_floor)
        residual = torch.clamp(residual, min=-100, max=100)
        
        loss_per_elem = charbonnier_loss(residual, delta=delta)
    
    elif mode == 'squared_magnitude':
        loss_per_elem = squared_magnitude_charbonnier(pred_norm, target_norm, sigma, delta, eps_floor)
    
    elif mode == 'sqmag_charb':
        loss_per_elem = squared_magnitude_charbonnier(pred_norm, target_norm, sigma, delta, eps_floor)
    
    elif mode == 'sqmag_charb_cosine':
        total_loss, info = sqmag_charb_cosine_loss(
            pred_norm, target_norm, shell_ids, bvals, sigma,
            delta=delta, lambda_cosine=lambda_cosine,
            use_shell_weights=True, eps=eps_floor, reduction=reduction
        )
        return total_loss, info
        
    elif mode == 'rician':
        loss_per_elem = rician_nll_fast(pred_norm, target_norm, sigma)
        
        loss_per_elem = torch.clamp(loss_per_elem, max=100)
    
    elif mode == 'rician_lite':
        loss_per_elem = rician_nll_fast(pred_norm, target_norm, sigma)
        loss_per_elem = torch.clamp(loss_per_elem, max=100)
        
    else:
        raise ValueError(f"Unknown mode: {mode}. Valid: mse, huber, huber_debiased, huber_squared, squared_magnitude, sqmag_charb, sqmag_charb_cosine, rician, rician_lite")
    
    weighted_loss = loss_per_elem * weights
    
    if mask is not None:
        while mask.ndim < weighted_loss.ndim:
            mask = mask.unsqueeze(-1)
        weighted_loss = weighted_loss * mask
        n_valid = mask.sum() * n_measurements
    else:
        n_valid = weighted_loss.numel()
    
    if reduction == 'mean':
        loss = weighted_loss.sum() / (n_valid + 1e-8)
    elif reduction == 'sum':
        loss = weighted_loss.sum()
    else:
        loss = weighted_loss
    
    info = {
        'sigma': sigma.mean().item() if isinstance(sigma, torch.Tensor) else sigma,
        'weights_range': (weights.min().item(), weights.max().item()),
        'residual_mean': (pred_norm - target_norm).abs().mean().item(),
        'loss_per_shell': {},
    }
    
    unique_shells = torch.unique(shell_ids)
    for s in unique_shells:
        shell_mask = shell_ids == s
        shell_loss = loss_per_elem[..., shell_mask].mean().item()
        info['loss_per_shell'][s.item()] = shell_loss
    
    return loss, info


class DMRILoss(torch.nn.Module):
    
    def __init__(self,
                 bvals: torch.Tensor,
                 shell_ids: torch.Tensor,
                 mode: str = 'huber_debiased',
                 delta: float = 1.0,
                 eps_floor: float = 1e-3,
                 shell_weight_mode: str = 'inverse_var',
                 shell_gains: Optional[torch.Tensor] = None,
                 sigma: Optional[float] = None):
        super().__init__()
        
        self.register_buffer('bvals', bvals)
        self.register_buffer('shell_ids', shell_ids)
        if shell_gains is not None:
            self.register_buffer('shell_gains', shell_gains)
        else:
            self.shell_gains = None
            
        self.mode = mode
        self.delta = delta
        self.eps_floor = eps_floor
        self.shell_weight_mode = shell_weight_mode
        self.sigma = sigma
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                sigma: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        
        return loss_dmri(
            pred=pred,
            target=target,
            shell_ids=self.shell_ids,
            bvals=self.bvals,
            sigma=sigma if sigma is not None else self.sigma,
            shell_gains=self.shell_gains,
            delta=self.delta,
            eps_floor=self.eps_floor,
            mode=self.mode,
            shell_weight_mode=self.shell_weight_mode,
            mask=mask
        )



def huber_dmri(pred: torch.Tensor, target: torch.Tensor,
               bvals: torch.Tensor, shell_ids: torch.Tensor,
               sigma: Optional[torch.Tensor] = None,
               delta: float = 1.0) -> torch.Tensor:
    loss, _ = loss_dmri(pred, target, shell_ids, bvals, sigma=sigma,
                        mode='huber', delta=delta)
    return loss


def rician_fast(pred: torch.Tensor, target: torch.Tensor,
                bvals: torch.Tensor, shell_ids: torch.Tensor,
                sigma: torch.Tensor) -> torch.Tensor:
    loss, _ = loss_dmri(pred, target, shell_ids, bvals, sigma=sigma,
                        mode='rician')
    return loss



def squared_magnitude_charbonnier(pred: torch.Tensor, 
                                   target: torch.Tensor,
                                   sigma: torch.Tensor,
                                   delta: float = 1.0,
                                   eps: float = 1e-6) -> torch.Tensor:
    s2 = pred ** 2
    y2 = target ** 2
    sigma2 = sigma ** 2
    
    residual_sq = y2 - (s2 + 2 * sigma2)
    
    var_y2 = 4 * sigma2 * (s2 + sigma2)
    
    r_whitened = residual_sq / torch.sqrt(var_y2 + eps)
    
    return charbonnier_loss(r_whitened, delta=delta)


def per_shell_cosine_loss(pred: torch.Tensor,
                          target: torch.Tensor,
                          shell_ids: torch.Tensor,
                          bvals: Optional[torch.Tensor] = None,
                          mean_centered: bool = True,
                          eps: float = 1e-6) -> torch.Tensor:
    unique_shells = torch.unique(shell_ids)
    
    if bvals is not None:
        b0_mask = bvals < 100
        b0_shells = torch.unique(shell_ids[b0_mask])
    else:
        b0_shells = torch.tensor([0], device=shell_ids.device)
    
    b0_shells_set = set(b0_shells.tolist())
    
    angular_shells = [s for s in unique_shells.tolist() if s not in b0_shells_set]
    
    if len(angular_shells) == 0:
        return torch.tensor(0.0, device=pred.device)
    
    total_loss = torch.tensor(0.0, device=pred.device)
    
    for shell_id in angular_shells:
        shell_mask = shell_ids == shell_id
        
        pred_shell = pred[..., shell_mask]
        target_shell = target[..., shell_mask]
        
        if mean_centered:
            pred_shell = pred_shell - pred_shell.mean(dim=-1, keepdim=True)
            target_shell = target_shell - target_shell.mean(dim=-1, keepdim=True)
        
        dot_product = (pred_shell * target_shell).sum(dim=-1)
        pred_norm = torch.sqrt((pred_shell ** 2).sum(dim=-1) + eps)
        target_norm = torch.sqrt((target_shell ** 2).sum(dim=-1) + eps)
        
        cosine_sim = dot_product / (pred_norm * target_norm + eps)
        
        shell_loss = (1 - cosine_sim).mean()
        total_loss = total_loss + shell_loss
    
    return total_loss / len(angular_shells)


def per_shell_zscore_mse(pred: torch.Tensor,
                         target: torch.Tensor,
                         shell_ids: torch.Tensor,
                         eps: float = 1e-6) -> torch.Tensor:
    unique_shells = torch.unique(shell_ids)
    angular_shells = unique_shells[unique_shells > 0]
    
    if len(angular_shells) == 0:
        return torch.tensor(0.0, device=pred.device)
    
    total_loss = torch.tensor(0.0, device=pred.device)
    
    for shell_id in angular_shells:
        shell_mask = shell_ids == shell_id
        
        pred_shell = pred[..., shell_mask]
        target_shell = target[..., shell_mask]
        
        pred_mean = pred_shell.mean(dim=-1, keepdim=True)
        pred_std = pred_shell.std(dim=-1, keepdim=True) + eps
        pred_z = (pred_shell - pred_mean) / pred_std
        
        target_mean = target_shell.mean(dim=-1, keepdim=True)
        target_std = target_shell.std(dim=-1, keepdim=True) + eps
        target_z = (target_shell - target_mean) / target_std
        
        shell_loss = ((pred_z - target_z) ** 2).mean()
        total_loss = total_loss + shell_loss
    
    return total_loss / len(angular_shells)


def monotonic_decay_loss(pred: torch.Tensor,
                         bvals: torch.Tensor,
                         shell_ids: torch.Tensor,
                         margin: float = 0.0) -> torch.Tensor:
    unique_shells = torch.unique(shell_ids)
    n_shells = len(unique_shells)
    
    if n_shells <= 1:
        return torch.tensor(0.0, device=pred.device)
    
    shell_means = []
    shell_bvals = []
    
    for shell_id in unique_shells:
        shell_mask = shell_ids == shell_id
        mean_signal = pred[..., shell_mask].mean(dim=-1)
        shell_means.append(mean_signal)
        shell_bvals.append(bvals[shell_mask].mean())
    
    shell_bvals = torch.stack(shell_bvals)
    sorted_idx = torch.argsort(shell_bvals)
    shell_means = [shell_means[i] for i in sorted_idx]
    
    total_loss = torch.tensor(0.0, device=pred.device)
    n_pairs = 0
    
    for k in range(len(shell_means) - 1):
        S_k = shell_means[k]
        S_k1 = shell_means[k + 1]
        
        violation = S_k1 - S_k * (1.0 + margin)
        
        hinge_loss = torch.relu(violation).mean()
        total_loss = total_loss + hinge_loss
        n_pairs += 1
    
    return total_loss / n_pairs if n_pairs > 0 else total_loss


def curvature_regularizer(pred: torch.Tensor,
                          bvals: torch.Tensor,
                          shell_ids: torch.Tensor) -> torch.Tensor:
    unique_shells = torch.unique(shell_ids)
    n_shells = len(unique_shells)
    
    if n_shells <= 2:
        return torch.tensor(0.0, device=pred.device)
    
    eps = 1e-8
    
    log_signals = []
    shell_bvals = []
    
    for shell_id in unique_shells:
        shell_mask = shell_ids == shell_id
        mean_signal = pred[..., shell_mask].mean(dim=-1).clamp(min=eps)
        log_s = torch.log(mean_signal)
        log_signals.append(log_s)
        shell_bvals.append(bvals[shell_mask].mean())
    
    shell_bvals = torch.stack(shell_bvals)
    sorted_idx = torch.argsort(shell_bvals)
    log_signals = torch.stack([log_signals[i] for i in sorted_idx], dim=-1)
    sorted_bvals = shell_bvals[sorted_idx]
    
    total_curvature = torch.tensor(0.0, device=pred.device)
    n_interior = 0
    
    for k in range(1, n_shells - 1):
        db1 = sorted_bvals[k] - sorted_bvals[k-1]
        db2 = sorted_bvals[k+1] - sorted_bvals[k]
        db_mean = (db1 + db2) / 2
        
        if db_mean < 1e-6:
            continue
            
        curvature = (log_signals[..., k+1] - 2*log_signals[..., k] + log_signals[..., k-1]) / (db_mean ** 2)
        total_curvature = total_curvature + (curvature ** 2).mean()
        n_interior += 1
    
    return total_curvature / n_interior if n_interior > 0 else total_curvature


def student_t_nll(pred: torch.Tensor,
                  target: torch.Tensor,
                  sigma: torch.Tensor,
                  nu: float = 4.0,
                  eps: float = 1e-6) -> torch.Tensor:
    z = (target - pred) / (sigma + eps)
    
    nll = 0.5 * (nu + 1) * torch.log(1 + z**2 / nu + eps)
    
    return nll



def sqmag_charb_cosine_loss(pred: torch.Tensor,
                             target: torch.Tensor,
                             shell_ids: torch.Tensor,
                             bvals: torch.Tensor,
                             sigma: torch.Tensor,
                             delta: float = 1.0,
                             lambda_cosine: float = 0.05,
                             use_shell_weights: bool = True,
                             clamp_weights: Tuple[float, float] = (0.1, 10.0),
                             eps: float = 1e-6,
                             reduction: str = 'mean') -> Tuple[torch.Tensor, dict]:
    device = pred.device
    
    if not isinstance(sigma, torch.Tensor):
        sigma = torch.tensor(sigma, device=device, dtype=pred.dtype)
    
    s2 = pred ** 2
    y2 = target ** 2
    sigma2 = sigma ** 2
    
    residual_sq = y2 - (s2 + 2 * sigma2)
    
    var_y2 = 4 * sigma2 * (s2 + sigma2)
    
    z = residual_sq / torch.sqrt(var_y2 + eps)
    
    z_scaled = z / delta
    amp_loss_elem = delta ** 2 * (torch.sqrt(1 + z_scaled ** 2) - 1)
    
    if use_shell_weights:
        weights = compute_shell_weights(bvals, shell_ids, mode='inverse_var',
                                         clamp_range=clamp_weights)
        weights = weights.to(device)
        for _ in range(pred.ndim - 1):
            weights = weights.unsqueeze(0)
        amp_loss_weighted = amp_loss_elem * weights
    else:
        amp_loss_weighted = amp_loss_elem
    
    if reduction == 'mean':
        amp_loss = amp_loss_weighted.mean()
    elif reduction == 'sum':
        amp_loss = amp_loss_weighted.sum()
    else:
        amp_loss = amp_loss_weighted
    
    if lambda_cosine > 0:
        cos_loss = per_shell_cosine_loss(pred, target, shell_ids, bvals, 
                                          mean_centered=True, eps=eps)
    else:
        cos_loss = torch.tensor(0.0, device=device)
    
    total_loss = amp_loss + lambda_cosine * cos_loss
    
    info = {
        'amp_loss': amp_loss.item() if reduction != 'none' else amp_loss.mean().item(),
        'cos_loss': cos_loss.item(),
        'total_loss': total_loss.item() if reduction != 'none' else total_loss.mean().item(),
        'sigma': sigma.mean().item() if sigma.numel() > 1 else sigma.item(),
        'z_mean': z.abs().mean().item(),
        'z_max': z.abs().max().item(),
    }
    
    return total_loss, info


def composite_dmri_loss(pred: torch.Tensor,
                        target: torch.Tensor,
                        shell_ids: torch.Tensor,
                        bvals: torch.Tensor,
                        sigma: torch.Tensor,
                        delta: float = 1.0,
                        lambda_cosine: float = 0.1,
                        lambda_monotonic: float = 0.01,
                        lambda_curvature: float = 0.001,
                        data_loss_mode: str = 'squared_magnitude',
                        eps: float = 1e-6,
                        reduction: str = 'mean') -> Tuple[torch.Tensor, dict]:
    device = pred.device
    
    if data_loss_mode == 'squared_magnitude':
        data_loss_elem = squared_magnitude_charbonnier(pred, target, sigma, delta, eps)
    elif data_loss_mode == 'debiased_huber':
        target_db = rician_debias(target, sigma, eps)
        residual = (pred - target_db) / (sigma + eps)
        data_loss_elem = charbonnier_loss(residual, delta)
    elif data_loss_mode == 'student_t':
        data_loss_elem = student_t_nll(pred, target, sigma)
    else:
        raise ValueError(f"Unknown data_loss_mode: {data_loss_mode}")
    
    weights = compute_shell_weights(bvals, shell_ids, mode='inverse_var')
    for _ in range(pred.ndim - 1):
        weights = weights.unsqueeze(0)
    weights = weights.to(device)
    
    weighted_data_loss = data_loss_elem * weights
    
    if reduction == 'mean':
        data_loss = weighted_data_loss.mean()
    elif reduction == 'sum':
        data_loss = weighted_data_loss.sum()
    else:
        data_loss = weighted_data_loss
    
    if lambda_cosine > 0:
        cosine_loss = per_shell_cosine_loss(pred, target, shell_ids, eps)
    else:
        cosine_loss = torch.tensor(0.0, device=device)
    
    if lambda_monotonic > 0:
        mono_loss = monotonic_decay_loss(pred, bvals, shell_ids)
    else:
        mono_loss = torch.tensor(0.0, device=device)
    
    if lambda_curvature > 0:
        curv_loss = curvature_regularizer(pred, bvals, shell_ids)
    else:
        curv_loss = torch.tensor(0.0, device=device)
    
    total_loss = data_loss + lambda_cosine * cosine_loss + \
                 lambda_monotonic * mono_loss + lambda_curvature * curv_loss
    
    info = {
        'data_loss': data_loss.item() if isinstance(data_loss, torch.Tensor) else data_loss,
        'cosine_loss': cosine_loss.item(),
        'monotonic_loss': mono_loss.item(),
        'curvature_loss': curv_loss.item(),
        'total_loss': total_loss.item(),
    }
    
    return total_loss, info


class CompositeDMRILoss(torch.nn.Module):
    
    def __init__(self,
                 bvals: torch.Tensor,
                 shell_ids: torch.Tensor,
                 sigma: float = 0.05,
                 delta: float = 1.0,
                 lambda_cosine: float = 0.1,
                 lambda_monotonic: float = 0.01,
                 lambda_curvature: float = 0.001,
                 data_loss_mode: str = 'squared_magnitude'):
        super().__init__()
        
        self.register_buffer('bvals', bvals)
        self.register_buffer('shell_ids', shell_ids)
        
        self.sigma = sigma
        self.delta = delta
        self.lambda_cosine = lambda_cosine
        self.lambda_monotonic = lambda_monotonic
        self.lambda_curvature = lambda_curvature
        self.data_loss_mode = data_loss_mode
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                sigma: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        
        if sigma is None:
            sigma = torch.tensor(self.sigma, device=pred.device)
        
        return composite_dmri_loss(
            pred=pred,
            target=target,
            shell_ids=self.shell_ids,
            bvals=self.bvals,
            sigma=sigma,
            delta=self.delta,
            lambda_cosine=self.lambda_cosine,
            lambda_monotonic=self.lambda_monotonic,
            lambda_curvature=self.lambda_curvature,
            data_loss_mode=self.data_loss_mode
        )
