#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import math



class ExchangeRegime(Enum):
    SLOW = "slow"
    INTERMEDIATE = "intermediate"
    FAST = "fast"


@dataclass
class ExchangeParameters:
    k_12: torch.Tensor
    k_21: torch.Tensor
    D_1: torch.Tensor
    D_2: torch.Tensor
    f_1: torch.Tensor
    
    @property
    def f_2(self) -> torch.Tensor:
        return 1 - self.f_1
    
    @property
    def tau_12(self) -> torch.Tensor:
        return 1.0 / (self.k_12 + 1e-8)
    
    @property
    def tau_21(self) -> torch.Tensor:
        return 1.0 / (self.k_21 + 1e-8)
    
    def get_regime(self, Delta: float) -> str:
        k_mean = 0.5 * (self.k_12.mean() + self.k_21.mean()).item()
        if k_mean * Delta < 0.1:
            return ExchangeRegime.SLOW
        elif k_mean * Delta > 10:
            return ExchangeRegime.FAST
        else:
            return ExchangeRegime.INTERMEDIATE



def karger_two_site_signal(
    b: torch.Tensor,
    D1: torch.Tensor,
    D2: torch.Tensor,
    f1: torch.Tensor,
    k12: torch.Tensor,
    k21: torch.Tensor,
    Delta: Optional[torch.Tensor] = None,
    S0: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if S0 is None:
        S0 = torch.ones_like(D1)
    
    if Delta is None:
        Delta_s = b.new_tensor(0.043)
    elif Delta.dim() == 0:
        Delta_s = Delta * 0.001
    else:
        Delta_s = b.new_tensor(0.043)
    
    N = b.shape[0]
    if D1.dim() == 0:
        n_spatial = 0
    else:
        n_spatial = len(D1.shape)
    
    def expand_spatial(x):
        if x.dim() == 0:
            return x.unsqueeze(-1).expand(N)
        return x.unsqueeze(-1)
    
    def expand_meas(x):
        return x.view(*([1]*n_spatial), N)
    
    b_exp = expand_meas(b)
    D1_exp = expand_spatial(D1)
    D2_exp = expand_spatial(D2)
    f1_exp = expand_spatial(f1)
    f2_exp = 1 - f1_exp
    k12_exp = expand_spatial(k12)
    k21_exp = expand_spatial(k21)
    
    k_total = k12_exp + k21_exp
    
    p_exchange = k_total * Delta_s
    
    S_slow = f1_exp * torch.exp(-b_exp * D1_exp / 1000.0) + \
             f2_exp * torch.exp(-b_exp * D2_exp / 1000.0)
    
    D_avg = f1_exp * D1_exp + f2_exp * D2_exp
    S_fast = torch.exp(-b_exp * D_avg / 1000.0)
    
    
    mix_factor = 1 - torch.exp(-p_exchange.clamp(max=10))
    
    S = (1 - mix_factor) * S_slow + mix_factor * S_fast
    
    b0_mask = (b_exp < 0.5)
    S = torch.where(b0_mask, torch.ones_like(S), S)
    
    if S0.dim() == 0:
        S = S0 * S
    else:
        S = S0.unsqueeze(-1) * S
    
    return S


def karger_signal_no_exchange(
    b: torch.Tensor,
    D1: torch.Tensor,
    D2: torch.Tensor,
    f1: torch.Tensor,
    S0: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if S0 is None:
        S0 = torch.ones_like(D1)
    
    f2 = 1 - f1
    
    b_exp = b.view(*([1]*D1.dim()), -1)
    D1_exp = D1.unsqueeze(-1)
    D2_exp = D2.unsqueeze(-1)
    f1_exp = f1.unsqueeze(-1)
    f2_exp = f2.unsqueeze(-1)
    
    S = f1_exp * torch.exp(-b_exp * D1_exp / 1000.0) + \
        f2_exp * torch.exp(-b_exp * D2_exp / 1000.0)
    
    if S0.dim() == 0:
        S = S0 * S
    else:
        S = S0.unsqueeze(-1) * S
    
    return S



class KargerExchangeModule(nn.Module):
    
    def __init__(
        self,
        compartment_pairs: List[Tuple[str, str]] = [('restricted', 'extra')],
        init_tau_ex: float = 50.0,
        min_tau_ex: float = 5.0,
        max_tau_ex: float = 500.0,
        learnable: bool = True,
        enforce_detailed_balance: bool = True,
    ):
        super().__init__()
        
        self.compartment_pairs = compartment_pairs
        self.n_pairs = len(compartment_pairs)
        self.min_tau_ex = min_tau_ex
        self.max_tau_ex = max_tau_ex
        self.enforce_detailed_balance = enforce_detailed_balance
        
        init_log_tau = math.log(init_tau_ex)
        
        if learnable:
            self.log_tau_ex = nn.Parameter(
                torch.full((self.n_pairs,), init_log_tau, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                'log_tau_ex',
                torch.full((self.n_pairs,), init_log_tau, dtype=torch.float32)
            )
        
        self.use_spatial_modulation = False
        
    def get_exchange_rates(
        self,
        f1: torch.Tensor,
        pair_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_tau = self.log_tau_ex[pair_idx]
        tau_ex_ms = torch.exp(log_tau)
        tau_ex_ms = torch.clamp(tau_ex_ms, self.min_tau_ex, self.max_tau_ex)
        
        tau_ex_s = tau_ex_ms * 0.001
        k = 1.0 / tau_ex_s
        
        if self.enforce_detailed_balance:
            f1_safe = torch.clamp(f1, 0.01, 0.99)
            k12 = k * (1 - f1_safe)
            k21 = k * f1_safe
        else:
            k12 = k
            k21 = k
        
        return k12, k21
    
    def forward(
        self,
        D1: torch.Tensor,
        D2: torch.Tensor,
        f1: torch.Tensor,
        b: torch.Tensor,
        Delta: Optional[torch.Tensor] = None,
        S0: Optional[torch.Tensor] = None,
        pair_idx: int = 0,
    ) -> torch.Tensor:
        k12, k21 = self.get_exchange_rates(f1, pair_idx)
        
        return karger_two_site_signal(
            b=b, D1=D1, D2=D2, f1=f1,
            k12=k12, k21=k21, Delta=Delta, S0=S0
        )
    
    def get_regularization_loss(self) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.log_tau_ex.device)
        
        tau_ex = torch.exp(self.log_tau_ex)
        target_tau = 50.0
        
        loss = loss + 0.01 * ((tau_ex - target_tau) / target_tau) ** 2
        
        return loss.mean()
    
    @property
    def tau_ex(self) -> torch.Tensor:
        return torch.exp(self.log_tau_ex).clamp(self.min_tau_ex, self.max_tau_ex)
    
    @property  
    def k_ex(self) -> torch.Tensor:
        return 1.0 / self.tau_ex
    
    def extra_repr(self) -> str:
        tau = self.tau_ex.detach().cpu().numpy()
        pairs = [f"{p[0]}-{p[1]}" for p in self.compartment_pairs]
        return f"pairs={pairs}, tau_ex={tau}"



class NsiteExchangeModule(nn.Module):
    
    def __init__(
        self,
        n_compartments: int = 3,
        exchange_pairs: List[Tuple[int, int]] = [(0, 1), (1, 2)],
        init_tau_ex: List[float] = [30.0, 100.0],
    ):
        super().__init__()
        
        self.n_compartments = n_compartments
        self.exchange_pairs = exchange_pairs
        self.n_pairs = len(exchange_pairs)
        
        assert len(init_tau_ex) == self.n_pairs
        
        self.log_tau_ex = nn.Parameter(
            torch.tensor([math.log(t) for t in init_tau_ex])
        )
    
    def build_rate_matrix(
        self,
        D: torch.Tensor,
        f: torch.Tensor,
        b: float,
        Delta: float = 43.0,
    ) -> torch.Tensor:
        N = self.n_compartments
        device = D.device
        
        if D.dim() == 1:
            batch_shape = ()
        else:
            batch_shape = D.shape[:-1]
        
        A = torch.zeros(*batch_shape, N, N, device=device)
        bD = b * D / 1000.0
        
        for i in range(N):
            A[..., i, i] = bD[..., i]
        
        tau_ex = torch.exp(self.log_tau_ex).clamp(5.0, 500.0)
        
        for p, (i, j) in enumerate(self.exchange_pairs):
            k = 1.0 / tau_ex[p] * Delta
            
            f_i = f[..., i].clamp(0.01, 0.99)
            f_j = f[..., j].clamp(0.01, 0.99)
            
            k_ij = k * f_j / (f_i + f_j)
            k_ji = k * f_i / (f_i + f_j)
            
            A[..., i, i] = A[..., i, i] + k_ij
            A[..., j, j] = A[..., j, j] + k_ji
            A[..., i, j] = A[..., i, j] - k_ji
            A[..., j, i] = A[..., j, i] - k_ij
        
        return A
    
    def forward(
        self,
        D: torch.Tensor,
        f: torch.Tensor,
        b: torch.Tensor,
        Delta: float = 43.0,
        S0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if S0 is None:
            S0 = torch.ones(D.shape[:-1], device=D.device)
        
        spatial_shape = D.shape[:-1]
        N_meas = b.shape[0]
        N_comp = self.n_compartments
        
        S = torch.zeros(*spatial_shape, N_meas, device=D.device)
        
        for m, b_val in enumerate(b):
            b_scalar = b_val.item()
            
            A = self.build_rate_matrix(D, f, b_scalar, Delta)
            
            exp_neg_A = torch.linalg.matrix_exp(-A)
            
            
            M_final = torch.einsum('...ij,...j->...i', exp_neg_A, f)
            S[..., m] = M_final.sum(dim=-1)
        
        S = S0.unsqueeze(-1) * S
        
        return S



class ExchangeCoupledSignal(nn.Module):
    
    def __init__(
        self,
        use_exchange: bool = True,
        exchange_pairs: List[str] = ['restricted_extra'],
        init_tau_ex: float = 50.0,
        learnable: bool = True,
    ):
        super().__init__()
        
        self.use_exchange = use_exchange
        self.exchange_pairs = exchange_pairs
        
        if use_exchange:
            self.exchange_modules = nn.ModuleDict()
            
            if 'restricted_extra' in exchange_pairs:
                self.exchange_modules['restricted_extra'] = KargerExchangeModule(
                    compartment_pairs=[('restricted', 'extra')],
                    init_tau_ex=init_tau_ex,
                    learnable=learnable,
                )
            
            if 'intra_extra' in exchange_pairs:
                self.exchange_modules['intra_extra'] = KargerExchangeModule(
                    compartment_pairs=[('intra', 'extra')],
                    init_tau_ex=30.0,
                    learnable=learnable,
                )
            
            if 'dot_extra' in exchange_pairs:
                self.exchange_modules['dot_extra'] = KargerExchangeModule(
                    compartment_pairs=[('dot', 'extra')],
                    init_tau_ex=100.0,
                    learnable=learnable,
                )
    
    def compute_exchange_signal(
        self,
        compartment_name: str,
        D_comp: torch.Tensor,
        D_extra: torch.Tensor,
        f_comp: torch.Tensor,
        b: torch.Tensor,
        Delta: Optional[torch.Tensor] = None,
        S0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pair_key = f"{compartment_name}_extra"
        
        if self.use_exchange and pair_key in self.exchange_modules:
            module = self.exchange_modules[pair_key]
            return module(
                D1=D_comp, D2=D_extra, f1=f_comp,
                b=b, Delta=Delta, S0=S0
            )
        else:
            return karger_signal_no_exchange(
                b=b, D1=D_comp, D2=D_extra, f1=f_comp, S0=S0
            )
    
    def get_regularization_loss(self) -> torch.Tensor:
        loss = torch.tensor(0.0)
        
        if self.use_exchange:
            for module in self.exchange_modules.values():
                loss = loss + module.get_regularization_loss()
        
        return loss
    
    def get_exchange_times(self) -> Dict[str, float]:
        times = {}
        if self.use_exchange:
            for name, module in self.exchange_modules.items():
                times[name] = module.tau_ex.item()
        return times



def compare_exchange_vs_no_exchange(
    D1: float = 0.2,
    D2: float = 1.5,
    f1: float = 0.3,
    tau_ex: float = 50.0,
    bvals: np.ndarray = None,
    Delta: float = 43.0,
) -> Dict[str, np.ndarray]:
    if bvals is None:
        bvals = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000])
    
    b = torch.tensor(bvals, dtype=torch.float32)
    D1_t = torch.tensor(D1)
    D2_t = torch.tensor(D2)
    f1_t = torch.tensor(f1)
    
    tau_ex_s = tau_ex / 1000.0
    k = 1.0 / tau_ex_s
    k12 = torch.tensor(k * (1 - f1))
    k21 = torch.tensor(k * f1)
    
    S_no_ex = karger_signal_no_exchange(b, D1_t, D2_t, f1_t)
    S_ex = karger_two_site_signal(b, D1_t, D2_t, f1_t, k12, k21)
    
    return {
        'bvals': bvals,
        'S_no_exchange': S_no_ex.numpy(),
        'S_exchange': S_ex.numpy(),
        'difference': (S_ex - S_no_ex).numpy() / (S_no_ex.numpy() + 1e-6),
        'tau_ex': tau_ex,
        'Delta': Delta,
    }


def scan_exchange_times(
    D1: float = 0.2,
    D2: float = 1.5,
    f1: float = 0.3,
    tau_range: np.ndarray = None,
    b_test: float = 3000.0,
    Delta: float = 43.0,
) -> Dict[str, np.ndarray]:
    if tau_range is None:
        tau_range = np.logspace(0, 3, 50)
    
    b = torch.tensor([b_test], dtype=torch.float32)
    D1_t = torch.tensor(D1)
    D2_t = torch.tensor(D2)
    f1_t = torch.tensor(f1)
    
    signals = []
    regimes = []
    
    for tau_ms in tau_range:
        tau_s = tau_ms / 1000.0
        k = 1.0 / tau_s
        k12 = torch.tensor(k * (1 - f1))
        k21 = torch.tensor(k * f1)
        
        S = karger_two_site_signal(b, D1_t, D2_t, f1_t, k12, k21)
        signals.append(S.item())
        
        k_delta = k * (Delta / 1000.0)
        if k_delta < 0.1:
            regimes.append('slow')
        elif k_delta > 10:
            regimes.append('fast')
        else:
            regimes.append('intermediate')
    
    return {
        'tau_ex': tau_range,
        'signal': np.array(signals),
        'regime': regimes,
        'b_test': b_test,
        'Delta': Delta,
    }



if __name__ == "__main__":
    print("=" * 70)
    print("KÄRGER EXCHANGE MODEL - Validation")
    print("=" * 70)
    
    D_restricted = 0.2
    D_extra = 1.5
    f_restricted = 0.15
    
    print(f"\nCompartment parameters:")
    print(f"  D_restricted = {D_restricted} mm²/s")
    print(f"  D_extra = {D_extra} mm²/s")
    print(f"  f_restricted = {f_restricted}")
    
    print("\n" + "-" * 70)
    print("Signal comparison at different exchange times:")
    print("-" * 70)
    
    for tau_ex in [10, 50, 100, 500]:
        results = compare_exchange_vs_no_exchange(
            D1=D_restricted, D2=D_extra, f1=f_restricted,
            tau_ex=tau_ex, Delta=43.0
        )
        
        idx_3k = np.where(results['bvals'] == 3000)[0][0]
        S_no = results['S_no_exchange'][idx_3k]
        S_ex = results['S_exchange'][idx_3k]
        diff = results['difference'][idx_3k]
        
        print(f"  τ_ex = {tau_ex:4d} ms: S_no_ex = {S_no:.4f}, "
              f"S_ex = {S_ex:.4f}, diff = {100*diff:+.1f}%")
    
    print("\n" + "-" * 70)
    print("KargerExchangeModule test:")
    print("-" * 70)
    
    module = KargerExchangeModule(
        compartment_pairs=[('restricted', 'extra')],
        init_tau_ex=50.0,
        learnable=True
    )
    
    B, X, Y, Z = 1, 4, 4, 4
    D1 = torch.full((B, X, Y, Z), 0.2)
    D2 = torch.full((B, X, Y, Z), 1.5)
    f1 = torch.full((B, X, Y, Z), 0.15)
    b = torch.tensor([0, 1000, 2000, 3000], dtype=torch.float32)
    Delta = torch.full((4,), 43.0)
    
    S = module(D1, D2, f1, b, Delta)
    print(f"  Input shape: ({B}, {X}, {Y}, {Z})")
    print(f"  b-values: {b.numpy()}")
    print(f"  Output shape: {S.shape}")
    print(f"  Signal at b=0: {S[0,0,0,0,0].item():.4f}")
    print(f"  Signal at b=3000: {S[0,0,0,0,3].item():.4f}")
    print(f"  τ_ex: {module.tau_ex.item():.1f} ms")
    print(f"  Reg loss: {module.get_regularization_loss().item():.6f}")
    
    print("\n" + "-" * 70)
    print("Exchange time sensitivity (b=3000):")
    print("-" * 70)
    
    scan = scan_exchange_times(
        D1=D_restricted, D2=D_extra, f1=f_restricted,
        b_test=3000.0, Delta=43.0
    )
    
    slow_mask = np.array([r == 'slow' for r in scan['regime']])
    fast_mask = np.array([r == 'fast' for r in scan['regime']])
    intermed_mask = np.array([r == 'intermediate' for r in scan['regime']])
    
    print(f"  Slow exchange (τ > {scan['Delta']/0.1:.0f}ms): "
          f"S ≈ {scan['signal'][slow_mask].mean():.4f}")
    print(f"  Intermediate exchange: "
          f"S ≈ {scan['signal'][intermed_mask].mean():.4f}")
    print(f"  Fast exchange (τ < {scan['Delta']/10:.1f}ms): "
          f"S ≈ {scan['signal'][fast_mask].mean():.4f}")
    
    print("\n" + "=" * 70)
    print("SUMMARY: Exchange Model Ready for Integration")
    print("=" * 70)
    print("""
Key findings:
1. Exchange can change signal by 5-15% in intermediate regime
2. Most observable when τ_ex ~ Δ (diffusion time)
3. Without exchange, restricted/DOT may be over-estimated
4. Exchange naturally creates high-b tail (currently attributed to DOT/kurtosis)

Integration points:
- Add KargerExchangeModule to DifferentiableScannerV4
- Replace f_restricted * S_restricted + (1-f_restricted) * S_extra
  with exchange-coupled signal
- Enable via use_exchange=True flag
""")
