import torch
import torch.nn as nn

from quantization.functions import adaptive_po2
from quantization.utils.binary_packer import binary_packer, binary_unpacker
import torch.nn.functional as F

class LittleBitITQSpecLinear(nn.Module):
    def __quant_convert__(
        self,
        do_train: bool,
        quant_func: torch.autograd.Function,
        *,
        is_po2: bool = False,
        split_dim: int = 1024,
        eff_bit: float | None = None,
        residual: bool = False,
        ratio_factor: float = 1.0,
        min_split_dim: int = 8,
        defer_init: bool = False,
        resume_eff_bit: float | None = None,
        resume_eff_bit_2: float | None = None,
        **kwargs,
    ):
        self.do_train = do_train
        self.quant_func = quant_func
        self.is_po2 = is_po2
        self.residual = residual
        self.defer_init = defer_init
        self.resume_eff_bit = resume_eff_bit
        self.resume_eff_bit_2 = kwargs.get("resume_eff_bit_2", resume_eff_bit_2)
        # Flag to track if weights are binarized to int8 for inference
        self._binarized = False
        a, b = self.in_features, self.out_features
        eff_bit_target = eff_bit
        self.ratio_factor = ratio_factor
        is_stage3 = (self.resume_eff_bit_2 is not None and self.resume_eff_bit_2 > 0.0)
        is_matryo = (self.resume_eff_bit is not None and self.resume_eff_bit > 0.0)
        
        # --- Matryoshka 모드 시 차원(용량) 개별 분할 ---
        # if self.defer_init and self.resume_eff_bit is not None:
        #     self.residual = True  # Matryo 모드에서는 Residual 학습이 강제됨
        if is_stage3:
            self.residual = True
            
            p_split = self._estimate_split_dim(a, b, self.resume_eff_bit, False)
            if p_split: p_split *= ratio_factor
            self.split_dim = self._finalize_split_dim(p_split, split_dim, min_split_dim)

            res_eff_bit = (self.resume_eff_bit_2 - self.resume_eff_bit)
            if res_eff_bit < 0: res_eff_bit = 0.0
            r_split = self._estimate_split_dim(a, b, res_eff_bit, False)
            if r_split: r_split *= ratio_factor
            self.split_dim_R = self._finalize_split_dim(r_split, split_dim, min_split_dim)

            res_eff_bit_2 = (eff_bit_target - self.resume_eff_bit_2) if eff_bit_target is not None else 0.0
            if res_eff_bit_2 < 0: res_eff_bit_2 = 0.0
            r2_split = self._estimate_split_dim(a, b, res_eff_bit_2, False)
            if r2_split: r2_split *= ratio_factor
            self.split_dim_R2 = self._finalize_split_dim(r2_split, split_dim, min_split_dim)

            eff_bit_actual = (self._compute_eff_bits(a, b, self.split_dim, False) + 
                              self._compute_eff_bits(a, b, self.split_dim_R, False) + 
                              self._compute_eff_bits(a, b, self.split_dim_R2, False))
            self.register_buffer("_split_dim_R2_final", torch.tensor(self.split_dim_R2))
            
        elif is_matryo:
            self.residual = True
            
            p_split = self._estimate_split_dim(a, b, self.resume_eff_bit, False)
            if p_split: p_split *= ratio_factor
            self.split_dim = self._finalize_split_dim(p_split, split_dim, min_split_dim)

            res_eff_bit = (eff_bit_target - self.resume_eff_bit) if eff_bit_target is not None else 0.0
            if res_eff_bit < 0: res_eff_bit = 0.0
            
            if ratio_factor > 1.0:
                r_split = p_split
            else:
                r_split = self._estimate_split_dim(a, b, res_eff_bit, False)
                if r_split: r_split *= ratio_factor
            self.split_dim_R = self._finalize_split_dim(r_split, split_dim, min_split_dim)

            eff_bit_actual = self._compute_eff_bits(a, b, self.split_dim, False) + self._compute_eff_bits(a, b, self.split_dim_R, False)
        else:
            split_calc_float = self._estimate_split_dim(a, b, eff_bit_target, residual)
            if split_calc_float: split_calc_float *= ratio_factor
            self.split_dim = self._finalize_split_dim(split_calc_float, split_dim, min_split_dim)
            self.split_dim_R = self.split_dim
            eff_bit_actual = self._compute_eff_bits(a, b, self.split_dim, residual)
        
        
        # split_calc_float = self._estimate_split_dim(a, b, eff_bit_target, residual)

        # if split_calc_float:
        #     split_calc_float *= ratio_factor

        # final_split_dim = self._finalize_split_dim(split_calc_float, split_dim, min_split_dim)
        # self.split_dim = final_split_dim

        # eff_bit_actual = self._compute_eff_bits(a, b, final_split_dim, residual)
        self.register_buffer("_eff_bit_target", torch.tensor(-1.0 if eff_bit_target is None else float(eff_bit_target)))
        self.register_buffer("_split_dim_final", torch.tensor(self.split_dim))
        self.register_buffer("_split_dim_R_final", torch.tensor(self.split_dim_R))
        self.register_buffer("_eff_bit_actual", torch.tensor(eff_bit_actual))

        if self.defer_init:
            self._initialize_deferred_parameters()
        elif self.do_train and hasattr(self, 'weight') and self.weight is not None:
            self._initialize_parameters()
        else:
            self._initialize_empty_parameters()

        # if self.do_train and hasattr(self, 'weight') and self.weight is not None:
        #     self._initialize_parameters()
        # else:
        #     self._initialize_empty_parameters()
            
        print("_eff_bit_target",self._eff_bit_target)
        print("_split_dim_final",self._split_dim_final)
        print("_split_dim_R_final",self._split_dim_R_final)
        print("_eff_bit_actual",self._eff_bit_actual)

    @staticmethod
    def _estimate_split_dim(a, b, eff_bit_target, residual) -> float | None:
        """Estimate the initial (float) value of split_dim based on bit target."""
        if eff_bit_target is None or a * b == 0:
            return None

        base = a + b + 16
        if residual:
            numerator = a * b * eff_bit_target - 32 * (a + b)
            denominator = 2 * base
        else:
            numerator = a * b * eff_bit_target - 16 * (a + b)
            denominator = base
        return numerator / denominator if denominator else None

    @staticmethod
    def _finalize_split_dim(
        split_float: float | None,
        split_default: int,
        min_split_dim: int,
    ) -> int:
        """Round down to nearest multiple of 8 and apply minimum fallback."""
        # Use default if no split estimate is available
        cand = split_float if split_float is not None else split_default
        cand = int(cand) if cand is not None else 0

        # Round down to a multiple of 8
        cand = (cand // 8) * 8
        if cand == 0:
            cand = min_split_dim

        return max(cand, min_split_dim)

    @staticmethod
    def _compute_eff_bits(a: int, b: int, s: int, residual: bool) -> float:
        """Calculate the actual effective bits used based on configuration."""
        if a * b == 0:
            return float("inf")

        if residual:
            num = s * 2 * (a + b + 16) + 32 * (a + b)
        else:
            num = s * (a + b + 16) + 16 * (a + b)
        return num / (a * b)

    def forward(self, x):
        *seqlen, hidden_dim = x.shape
        seqlen.append(self.out_features)
        hidden_output_dim = tuple(seqlen)
        x = x.view(-1, hidden_dim)

        # Compute main forward pass
        y = self._compute_forward(x, self.V, self.U, self.v2, self.v1, self.u2, self.u1)

        if self.residual:
            if self.ratio_factor > 1.0:
                res = self._compute_forward(x, self.V, self.U_R, self.v2, self.v1, self.u2_R, self.u1_R)
            else:            
                # Compute residual forward pass
                res = self._compute_forward(x, self.V_R, self.U_R, self.v2_R, self.v1_R, self.u2_R, self.u1_R)
            y = y + res

        if getattr(self, "resume_eff_bit_2", 0.0) > 0.0:
            res2 = self._compute_forward(x, self.V_R2, self.U_R2, self.v2_R2, self.v1_R2, self.u2_R2, self.u1_R2)
            y = y + res2

        if self.bias is not None:
            y += self.bias
        y = y.reshape(hidden_output_dim)
        return y

    def _compute_forward(self, x, V, U, v2, v1, u2, u1):
        """Helper method to compute the forward pass for both main and residual components."""
        Vq = self.quantize(V.to(x.dtype))
        Uq = self.quantize(U.to(x.dtype))
        v1u2 = v1 * u2

        if self.is_po2:
            v2 = adaptive_po2(v2)
            v1u2 = adaptive_po2(v1u2)
            u1 = adaptive_po2(u1)

        # ((((x * v2) @ Vq^T) * (v1 * u2)) @ Uq^T) * u1
        return ((((x * v2) @ Vq.t()) * v1u2) @ Uq.t()) * u1

    def quantize(self, x):
        # If weights are already binarized, return them directly
        if self._binarized:
            return x
        # Otherwise, apply quantization function
        return self.quant_func(x)

    def extra_repr(self):
        params = {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.bias is not None,
            "split_dim": self._split_dim_final,
            "eff_bit_target": f"{self.eff_bit_target:.4f}" if self.eff_bit_target is not None else "N/A",
            "eff_bit_actual": f"{self.eff_bit_actual:.4f}",
            "residual": self.residual,
            "is_po2": self.is_po2,
            "total_bit_usage": f"{self.total_bit_usage:.0f}"
        }

        return ", ".join(f"{key}={value}" for key, value in params.items())

    def _initialize_empty_parameters(self):
        """Initialize with empty parameters for memory efficiency during inference"""
        dtype = torch.bfloat16  # temporary dtype, actual values loaded from state_dict
        device = "meta"  # use meta device to prevent actual memory allocation

        def create_param(*shape):
            return nn.Parameter(torch.empty(*shape, device=device, dtype=dtype), requires_grad=self.do_train)

        def create_buffer(name, *shape):
            self.register_buffer(name, torch.empty(*shape, device=device, dtype=dtype))

        if self.defer_init:
            create_buffer("U", self.out_features, self.split_dim)
            create_buffer("V", self.split_dim, self.in_features)
            create_buffer("u1", 1, self.out_features)
            create_buffer("u2", 1, self.split_dim)
            create_buffer("v1", 1, self.split_dim)
            create_buffer("v2", 1, self.in_features)
        else:
            self.U = create_param(self.out_features, self.split_dim)
            self.V = create_param(self.split_dim, self.in_features)
            self.u1 = create_param(1, self.out_features)
            self.u2 = create_param(1, self.split_dim)
            self.v1 = create_param(1, self.split_dim)
            self.v2 = create_param(1, self.in_features)

        if self.residual:
            # Initialize residual parameters
            self.U_R = create_param(self.out_features, self.split_dim_R)
            self.V_R = create_param(self.split_dim_R, self.in_features)
            self.u1_R = create_param(1, self.out_features)
            self.u2_R = create_param(1, self.split_dim_R)
            self.v1_R = create_param(1, self.split_dim_R)
            self.v2_R = create_param(1, self.in_features)

        if getattr(self, "resume_eff_bit_2", 0.0) > 0.0:
            if self.defer_init:
                create_buffer("U_R2", self.out_features, self.split_dim_R2)
                create_buffer("V_R2", self.split_dim_R2, self.in_features)
                create_buffer("u1_R2", 1, self.out_features)
                create_buffer("u2_R2", 1, self.split_dim_R2)
                create_buffer("v1_R2", 1, self.split_dim_R2)
                create_buffer("v2_R2", 1, self.in_features)
            else:
                self.U_R2 = create_param(self.out_features, self.split_dim_R2)
                self.V_R2 = create_param(self.split_dim_R2, self.in_features)
                self.u1_R2 = create_param(1, self.out_features)
                self.u2_R2 = create_param(1, self.split_dim_R2)
                self.v1_R2 = create_param(1, self.split_dim_R2)
                self.v2_R2 = create_param(1, self.in_features)
            
        # Delete original weight
        if hasattr(self, 'weight'):
            del self.weight
        self.register_parameter('weight', None)

    def _initialize_deferred_parameters(self):
        """Primary Path는 Freeze하기 위해 Buffer로 등록하고 Residual만 학습 파라미터로 생성합니다."""
        dtype = self.weight.dtype if hasattr(self, 'weight') and self.weight is not None else torch.bfloat16
        device = self.weight.device if hasattr(self, 'weight') and self.weight is not None else "meta"

        def create_param(*shape):
            return nn.Parameter(torch.empty(*shape, device=device, dtype=dtype), requires_grad=True)
            
        def create_buffer(name, *shape):
            self.register_buffer(name, torch.zeros(*shape, device=device, dtype=dtype))

        create_buffer("U", self.out_features, self.split_dim)
        create_buffer("V", self.split_dim, self.in_features)
        create_buffer("u1", 1, self.out_features)
        create_buffer("u2", 1, self.split_dim)
        create_buffer("v1", 1, self.split_dim)
        create_buffer("v2", 1, self.in_features)

        if self.ratio_factor > 1.0:
            if getattr(self, "resume_eff_bit_2", 0.0) > 0.0:
                create_buffer("U_R", self.out_features, self.split_dim_R)
                create_buffer("u1_R", 1, self.out_features)
                create_buffer("u2_R", 1, self.split_dim_R)

                self.U_R2 = create_param(self.out_features, self.split_dim_R2)
                self.u1_R2 = create_param(1, self.out_features)
                self.u2_R2 = create_param(1, self.split_dim_R2)
            else:

                self.U_R = create_param(self.out_features, self.split_dim_R)
                self.u1_R = create_param(1, self.out_features)
                self.u2_R = create_param(1, self.split_dim_R)
        else:
            if getattr(self, "resume_eff_bit_2", 0.0) > 0.0:
                create_buffer("U_R", self.out_features, self.split_dim_R)
                create_buffer("V_R", self.split_dim_R, self.in_features)
                create_buffer("u1_R", 1, self.out_features)
                create_buffer("u2_R", 1, self.split_dim_R)
                create_buffer("v1_R", 1, self.split_dim_R)
                create_buffer("v2_R", 1, self.in_features)

                self.U_R2 = create_param(self.out_features, self.split_dim_R2)
                self.V_R2 = create_param(self.split_dim_R2, self.in_features)
                self.u1_R2 = create_param(1, self.out_features)
                self.u2_R2 = create_param(1, self.split_dim_R2)
                self.v1_R2 = create_param(1, self.split_dim_R2)
                self.v2_R2 = create_param(1, self.in_features)
            else:

                self.U_R = create_param(self.out_features, self.split_dim_R)
                self.V_R = create_param(self.split_dim_R, self.in_features)
                self.u1_R = create_param(1, self.out_features)
                self.u2_R = create_param(1, self.split_dim_R)
                self.v1_R = create_param(1, self.split_dim_R)
                self.v2_R = create_param(1, self.in_features)

    def init_deferred_residual(self):
        """1차 모델 로딩 완료 후, 원래 가중치에서 1차 근사치를 뺀 값을 Residual SVD합니다."""
        if not getattr(self, "defer_init", False) or not hasattr(self, 'weight') or self.weight is None:
            return
            
        with torch.no_grad():
            orig_device = self.weight.device
            W_orig = self.weight.data.float()
            
            U_f = self.U.to(orig_device).float()
            V_f = self.V.to(orig_device).float()
            u1_f = self.u1.to(orig_device).float()
            u2_f = self.u2.to(orig_device).float()
            v1_f = self.v1.to(orig_device).float()
            v2_f = self.v2.to(orig_device).float()

            if getattr(self, "is_po2", False):
                from quantization.functions import adaptive_po2
                v2_f = adaptive_po2(v2_f)
                v1u2_f = adaptive_po2(v1_f * u2_f)
                u1_f = adaptive_po2(u1_f)
                W_approx = (self.quantize(U_f) * u1_f.t()) @ torch.diag(v1u2_f.squeeze(0)) @ (self.quantize(V_f) * v2_f)
            else:
                W_approx = (self.quantize(U_f) * (u1_f.t() @ u2_f)) @ (self.quantize(V_f) * (v1_f.t() @ v2_f))
            
            # W_approx를 제외한 나머지를 잔차로 추출
            residual_W = W_orig - W_approx
            
            if self.ratio_factor > 1.0:
                U_R, u1_R, u2_R = self._decompose_matrix_with_V(residual_W, self.split_dim_R, (self.quantize(V_f) * (v1_f.t() @ v2_f)))
                self.U_R.data.copy_(U_R)
                self.u1_R.data.copy_(u1_R)
                self.u2_R.data.copy_(u2_R)

            else:
                U_R, V_R, u1_R, u2_R, v1_R, v2_R = self._decompose_matrix(residual_W, self.split_dim_R)
            
                self.U_R.data.copy_(U_R)
                self.V_R.data.copy_(V_R)
                self.u1_R.data.copy_(u1_R)
                self.u2_R.data.copy_(u2_R)
                self.v1_R.data.copy_(v1_R)
                self.v2_R.data.copy_(v2_R)

        del self.weight
        self.register_parameter('weight', None)
        self._binarized = False

    def init_deferred_residual_stage3(self):
        if not getattr(self, "defer_init", False) or not hasattr(self, 'weight') or self.weight is None:
            return
            
        with torch.no_grad():
            orig_device = self.weight.device
            W_orig = self.weight.data.float()
            
            # Primary W_approx 계산
            U_f, V_f = self.U.to(orig_device).float(), self.V.to(orig_device).float()
            u1_f, u2_f = self.u1.to(orig_device).float(), self.u2.to(orig_device).float()
            v1_f, v2_f = self.v1.to(orig_device).float(), self.v2.to(orig_device).float()

            if getattr(self, "is_po2", False):
                from quantization.functions import adaptive_po2
                v2_f = adaptive_po2(v2_f)
                v1u2_f = adaptive_po2(v1_f * u2_f)
                u1_f = adaptive_po2(u1_f)
                W_approx_1 = (self.quantize(U_f) * u1_f.t()) @ torch.diag(v1u2_f.squeeze(0)) @ (self.quantize(V_f) * v2_f)
            else:
                W_approx_1 = (self.quantize(U_f) * (u1_f.t() @ u2_f)) @ (self.quantize(V_f) * (v1_f.t() @ v2_f))

            # Stage 2 (R1) W_approx 계산
            UR_f, VR_f = self.U_R.to(orig_device).float(), self.V_R.to(orig_device).float()
            u1R_f, u2R_f = self.u1_R.to(orig_device).float(), self.u2_R.to(orig_device).float()
            v1R_f, v2R_f = self.v1_R.to(orig_device).float(), self.v2_R.to(orig_device).float()

            if getattr(self, "is_po2", False):
                from quantization.functions import adaptive_po2
                v2R_f = adaptive_po2(v2R_f)
                v1u2R_f = adaptive_po2(v1R_f * u2R_f)
                u1R_f = adaptive_po2(u1R_f)
                W_approx_2 = (self.quantize(UR_f) * u1R_f.t()) @ torch.diag(v1u2R_f.squeeze(0)) @ (self.quantize(VR_f) * v2R_f)
            else:
                W_approx_2 = (self.quantize(UR_f) * (u1R_f.t() @ u2R_f)) @ (self.quantize(VR_f) * (v1R_f.t() @ v2R_f))
            
            # 원본 W에서 Stage 1과 2를 모두 빼서 Residual 2 (Stage 3) 타겟을 만듭니다.
            residual_W = W_orig - W_approx_1 - W_approx_2
            
            U_R2, V_R2, u1_R2, u2_R2, v1_R2, v2_R2 = self._decompose_matrix(residual_W, self.split_dim_R2)
            
            self.U_R2.data.copy_(U_R2); self.V_R2.data.copy_(V_R2)
            self.u1_R2.data.copy_(u1_R2); self.u2_R2.data.copy_(u2_R2)
            self.v1_R2.data.copy_(v1_R2); self.v2_R2.data.copy_(v2_R2)

        del self.weight
        self.register_parameter('weight', None)
        self._binarized = False


    def _initialize_parameters(self):
        W = self.weight.data.float() if self.do_train and self.weight is not None else None

        U, V, u1, u2, v1, v2 = self._decompose_matrix(W, self.split_dim)

        def create_param(tensor):
            return nn.Parameter(tensor, requires_grad=self.do_train)

        self.U = create_param(U)
        self.V = create_param(V)
        self.v1 = create_param(v1)
        self.v2 = create_param(v2)
        self.u1 = create_param(u1)
        self.u2 = create_param(u2)

        if self.residual:
            residual_W = None
            if self.do_train:
                if getattr(self, "is_po2", False):
                    from quantization.functions import adaptive_po2
                    v2_f = adaptive_po2(v2)
                    v1u2_f = adaptive_po2(v1 * u2)
                    u1_f = adaptive_po2(u1)
                    W_approx = (self.quantize(U) * u1_f.t()) @ torch.diag(v1u2_f.squeeze(0)) @ (self.quantize(V) * v2_f)
                else:
                    W_approx = (self.quantize(U) * (u1.t() @ u2)) @ (self.quantize(V) * (v1.t() @ v2))
                residual_W = self.weight.data.float() - W_approx

            U_R, V_R, u1_R, u2_R, v1_R, v2_R = self._decompose_matrix(residual_W, self.split_dim_R)

            self.U_R = create_param(U_R)
            self.V_R = create_param(V_R)
            self.v1_R = create_param(v1_R)
            self.v2_R = create_param(v2_R)
            self.u1_R = create_param(u1_R)
            self.u2_R = create_param(u2_R)

        self.register_parameter('weight', None)
        self._binarized = False


    def pack_weights(self):
        """
        Pack binary weights. Shapes are converted to tensors to ensure
        the state_dict contains only tensors.
        """
        packed_data = {}

        # Helper function to binarize and pack a parameter
        def pack_param(param, name):
            param_bin = param.data.sign().to(torch.int8)
            packed_data[f'{name}_packed'] = binary_packer(param_bin)
            packed_data[f'{name}_shape'] = torch.tensor(param.shape, dtype=torch.long)

        # Pack main parameters
        pack_param(self.U, 'U')
        pack_param(self.V, 'V')

        if self.residual:
            # Pack residual parameters
            pack_param(self.U_R, 'U_R')
            if self.ratio_factor == 1.0:    
                pack_param(self.V_R, 'V_R')

        if getattr(self, "resume_eff_bit_2", 0.0) > 0.0:
            pack_param(self.U_R2, 'U_R2')
            pack_param(self.V_R2, 'V_R2')

        return packed_data

    def state_dict(self, *args, **kwargs):
        """Always return the state_dict in a binarized & packed format."""
        # Get non-binary parameters like scaling factors and bias first.
        prefix = kwargs.get('prefix', '')
        state = super().state_dict(*args, **kwargs)

        # Remove the decomposed float parameters (U, V, U_R, V_R).
        # We only want to save the packed versions.
        keys_to_remove = [k for k in state.keys() if k.startswith(prefix + 'U') or k.startswith(prefix + 'V')]
        for k in keys_to_remove:
            if k in state:
                del state[k]

        # Pack the binary weights and add them to the state_dict.
        packed_weights = self.pack_weights()
        for k, v in packed_weights.items():
            state[prefix + k] = v

        return state

    def _compute_itq_rotation(self, X, n_iter=20):
        """
        Finds optimal Rotation R that aligns data to the binary hypercube.
        Objective: min || B - X @ R ||_F^2  s.t. B = sign(X @ R)
        """
        with torch.no_grad():
            N, dim = X.shape
            device = X.device
            
            # 정밀도를 위해 잠시 float32 사용
            X_f = X.float()
            
            # 1. Initialize R with Random Orthogonal (기존 방식과 동일한 출발점)
            R = torch.empty((dim, dim), device=device, dtype=torch.float32)
            torch.nn.init.orthogonal_(R)
            
            # 2. Iterative Optimization (Alternating Minimization)
            for _ in range(n_iter):
                # Step A: R 고정, Binary Target B 업데이트
                Z = X_f @ R
                B = torch.sign(Z)
                
                # Step B: B 고정, R 업데이트 (Orthogonal Procrustes Problem)
                # Maximize Tr(B^T @ X @ R) -> SVD of B^T @ X
                M = B.t() @ X_f
                U_p, _, Vt_p = torch.linalg.svd(M, full_matrices=False)
                
                # Optimal R = V @ U^T
                R = Vt_p.t() @ U_p.t()
                
            return R.to(X.dtype)

    def _decompose_matrix_with_V(self, X=None, target_split_dim=None, V=None):
        if target_split_dim is None:
            target_split_dim = self.split_dim

        if self.do_train:
            X_f = X.float()
            V_f = V.float()
            V_pinv  = torch.linalg.pinv(V_f)
            U = X_f @ V_pinv
            
            u1, u2 = self._rank_one_decompose(torch.abs(U))

            dtype = torch.bfloat16
            return (
                U.to(dtype), 
                u1.to(dtype), u2.to(dtype)
            )
        else:
            U = torch.empty(self.out_features, target_split_dim)
            u1 = torch.empty(1, self.out_features); u2 = torch.empty(1, target_split_dim)
            return U, u1, u2

    def _decompose_matrix(self, X=None, target_split_dim=None):
        if target_split_dim is None:
            target_split_dim = self.split_dim

        if self.do_train:
            X_f = X.float()
            U_t, S_t, Vh_t = torch.linalg.svd(X_f, full_matrices=False)

            S_sqrt_vec = torch.sqrt(S_t[:target_split_dim]) 
            S_sqrt_mat = torch.diag(S_sqrt_vec)

            U_temp = U_t[:, :target_split_dim] @ S_sqrt_mat  
            V_temp = S_sqrt_mat @ Vh_t[:target_split_dim, :]  

            with torch.no_grad():
                X_combined = torch.cat([U_temp, V_temp.t()], dim=0)
                R = self._compute_itq_rotation(X_combined, n_iter=50)
                U = (U_temp @ R).contiguous()
                V = (R.t() @ V_temp).contiguous()

            v1, v2 = self._rank_one_decompose(torch.abs(V))
            u1, u2 = self._rank_one_decompose(torch.abs(U))

            dtype = torch.bfloat16
            return (
                U.to(dtype), V.to(dtype), 
                u1.to(dtype), u2.to(dtype), 
                v1.to(dtype), v2.to(dtype)
            )
        else:
            U = torch.empty(self.out_features, target_split_dim)
            V = torch.empty(target_split_dim, self.in_features)
            u1 = torch.empty(1, self.out_features); u2 = torch.empty(1, target_split_dim)
            v1 = torch.empty(1, target_split_dim); v2 = torch.empty(1, self.in_features)
            return U, V, u1, u2, v1, v2

    def _rank_one_decompose(self, X):
        """
        Perform rank-one decomposition on matrix X via SVD and return two vectors.
        """
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        sqrt_S0 = torch.sqrt(S[0])
        u_component = (U[:, :1] * sqrt_S0).t().contiguous()
        v_component = (sqrt_S0 * Vh[:1, :]).contiguous()
        return u_component, v_component

    @property
    def eff_bit_target(self):
        v = self._eff_bit_target.item()
        return None if v < 0 else v

    @property
    def eff_bit_actual(self):
        return self._eff_bit_actual.item()

    @property
    def split_dim_used(self):
        return int(self._split_dim.item())

    @property
    def total_bit_usage(self):
        return self.eff_bit_actual * self.in_features * self.out_features


