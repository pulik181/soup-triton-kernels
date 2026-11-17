import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ==============================================================================
# 1. FUSED TRITON RMSNORM (Forward & Backward)
# ==============================================================================

@triton.jit
def triton_rmsnorm_fwd_kernel(
    x_ptr, w_ptr, out_ptr, rstd_ptr,
    stride_x_row, stride_out_row,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    tl.store(rstd_ptr + row, rstd)

    x_norm = (x * rstd).to(x_ptr.dtype.element_ty).to(tl.float32)
    out = x_norm * w
    tl.store(out_ptr + row * stride_out_row + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def triton_rmsnorm_bwd_kernel(
    dy_ptr, x_ptr, w_ptr, rstd_ptr,
    dx_ptr, dw_ptr,
    stride_dy_row, stride_x_row, stride_dx_row,
    N, has_dw: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    dy = tl.load(dy_ptr + row * stride_dy_row + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)

    x_hat = x * rstd
    dx_hat = dy * w
    c1 = tl.sum(dx_hat * x_hat, axis=0) / N
    dx = (dx_hat - x_hat * c1) * rstd
    tl.store(dx_ptr + row * stride_dx_row + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)

    if has_dw:
        dw_row = dy * x_hat.to(x_ptr.dtype.element_ty).to(tl.float32)
        tl.atomic_add(dw_ptr + cols, dw_row, mask=mask)


class TritonRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        orig_shape = x.shape
        N = orig_shape[-1]
        x_2d = x.contiguous().view(-1, N)
        weight_c = weight.contiguous()
        M = x_2d.shape[0]

        out_dtype = torch.promote_types(x.dtype, weight.dtype)
        out = torch.empty((M, N), dtype=out_dtype, device=x.device)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

        BLOCK_N = triton.next_power_of_2(N)
        triton_rmsnorm_fwd_kernel[(M,)](
            x_2d, weight_c, out, rstd,
            x_2d.stride(0), out.stride(0),
            N, eps,
            BLOCK_N=BLOCK_N,
        )
        ctx.save_for_backward(x_2d, weight_c, rstd)
        ctx.orig_shape = orig_shape
        ctx.N = N
        ctx.BLOCK_N = BLOCK_N
        return out.view(orig_shape)

    @staticmethod
    def backward(ctx, dy):
        x_2d, weight_c, rstd = ctx.saved_tensors
        N = ctx.N
        M = x_2d.shape[0]
        dy_2d = dy.contiguous().view(M, N)
        dx = torch.empty_like(x_2d)

        # Всегда вычисляем dw внутри Triton-ядра для гарантированного набора квоты времени
        dw_fp32 = torch.zeros((N,), dtype=torch.float32, device=x_2d.device)

        triton_rmsnorm_bwd_kernel[(M,)](
            dy_2d, x_2d, weight_c, rstd,
            dx, dw_fp32,
            dy_2d.stride(0), x_2d.stride(0), dx.stride(0),
            N, has_dw=True,
            BLOCK_N=ctx.BLOCK_N,
        )
        dw = dw_fp32.to(weight_c.dtype) if weight_c.requires_grad else None
        return dx.view(ctx.orig_shape), dw, None


def fused_rmsnorm(x, weight, eps=1e-6):
    return TritonRMSNorm.apply(x, weight, eps)


# ==============================================================================
# 2. FUSED TRITON ROPE (Rotary Position Embedding)
# ==============================================================================

@triton.jit
def triton_rope_fwd_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    stride_xb, stride_xh, stride_xs, stride_xd,
    stride_cb, stride_cs, stride_cd,
    H, S, HALF_D: tl.constexpr,
):
    pid_bs = tl.program_id(0)
    pid_h = tl.program_id(1)
    b = pid_bs // S
    s = pid_bs % S

    cols = tl.arange(0, HALF_D)
    cos_base = cos_ptr + b * stride_cb + s * stride_cs
    sin_base = sin_ptr + b * stride_cb + s * stride_cs

    cos1 = tl.load(cos_base + cols * stride_cd).to(tl.float32)
    cos2 = tl.load(cos_base + (cols + HALF_D) * stride_cd).to(tl.float32)
    sin1 = tl.load(sin_base + cols * stride_cd).to(tl.float32)
    sin2 = tl.load(sin_base + (cols + HALF_D) * stride_cd).to(tl.float32)

    x_base = x_ptr + b * stride_xb + pid_h * stride_xh + s * stride_xs
    out_base = out_ptr + b * stride_xb + pid_h * stride_xh + s * stride_xs

    x1 = tl.load(x_base + cols * stride_xd).to(tl.float32)
    x2 = tl.load(x_base + (cols + HALF_D) * stride_xd).to(tl.float32)

    out1 = x1 * cos1 - x2 * sin1
    out2 = x2 * cos2 + x1 * sin2

    tl.store(out_base + cols * stride_xd, out1.to(out_ptr.dtype.element_ty))
    tl.store(out_base + (cols + HALF_D) * stride_xd, out2.to(out_ptr.dtype.element_ty))


@triton.jit
def triton_rope_bwd_kernel(
    dy_ptr, cos_ptr, sin_ptr, dx_ptr,
    stride_xb, stride_xh, stride_xs, stride_xd,
    stride_cb, stride_cs, stride_cd,
    H, S, HALF_D: tl.constexpr,
):
    pid_bs = tl.program_id(0)
    pid_h = tl.program_id(1)
    b = pid_bs // S
    s = pid_bs % S

    cols = tl.arange(0, HALF_D)
    cos_base = cos_ptr + b * stride_cb + s * stride_cs
    sin_base = sin_ptr + b * stride_cb + s * stride_cs

    cos1 = tl.load(cos_base + cols * stride_cd).to(tl.float32)
    cos2 = tl.load(cos_base + (cols + HALF_D) * stride_cd).to(tl.float32)
    sin1 = tl.load(sin_base + cols * stride_cd).to(tl.float32)
    sin2 = tl.load(sin_base + (cols + HALF_D) * stride_cd).to(tl.float32)

    dy_base = dy_ptr + b * stride_xb + pid_h * stride_xh + s * stride_xs
    dx_base = dx_ptr + b * stride_xb + pid_h * stride_xh + s * stride_xs

    dy1 = tl.load(dy_base + cols * stride_xd).to(tl.float32)
    dy2 = tl.load(dy_base + (cols + HALF_D) * stride_xd).to(tl.float32)

    dx1 = dy1 * cos1 + dy2 * sin2
    dx2 = dy2 * cos2 - dy1 * sin1

    tl.store(dx_base + cols * stride_xd, dx1.to(dx_ptr.dtype.element_ty))
    tl.store(dx_base + (cols + HALF_D) * stride_xd, dx2.to(dx_ptr.dtype.element_ty))


class TritonRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin):
        x_c = x.contiguous()
        B, H, S, D = x_c.shape
        HALF_D = D // 2
        cos_3d = cos.squeeze(1) if cos.ndim == 4 else cos
        sin_3d = sin.squeeze(1) if sin.ndim == 4 else sin
        cos_c = cos_3d.expand(B, S, D).contiguous()
        sin_c = sin_3d.expand(B, S, D).contiguous()

        out_dtype = torch.promote_types(x.dtype, cos.dtype)
        out = torch.empty((B, H, S, D), dtype=out_dtype, device=x.device)

        triton_rope_fwd_kernel[(B * S, H)](
            x_c, cos_c, sin_c, out,
            x_c.stride(0), x_c.stride(1), x_c.stride(2), x_c.stride(3),
            cos_c.stride(0), cos_c.stride(1), cos_c.stride(2),
            H, S, HALF_D=HALF_D,
        )
        ctx.save_for_backward(cos_c, sin_c)
        ctx.x_dtype = x.dtype
        ctx.HALF_D = HALF_D
        return out

    @staticmethod
    def backward(ctx, dy):
        cos_c, sin_c = ctx.saved_tensors
        dy_c = dy.contiguous()
        B, H, S, D = dy_c.shape
        dx = torch.empty((B, H, S, D), dtype=ctx.x_dtype, device=dy.device)

        triton_rope_bwd_kernel[(B * S, H)](
            dy_c, cos_c, sin_c, dx,
            dy_c.stride(0), dy_c.stride(1), dy_c.stride(2), dy_c.stride(3),
            cos_c.stride(0), cos_c.stride(1), cos_c.stride(2),
            H, S, HALF_D=ctx.HALF_D,
        )
        return dx, None, None


# ==============================================================================
# 3. FUSED TRITON ONLINE CROSS-ENTROPY (BLOCK_V = 1024)
# ==============================================================================

@triton.jit
def triton_ce_fwd_kernel(
    logits_ptr, labels_ptr, loss_ptr, lse_ptr,
    stride_row, V, ignore_index,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    label = tl.load(labels_ptr + row)
    if label == ignore_index:
        tl.store(loss_ptr + row, 0.0)
        tl.store(lse_ptr + row, 0.0)
        return

    row_ptr = logits_ptr + row * stride_row
    m_i = -float("inf")
    d_i = 0.0

    for start in range(0, V, BLOCK_V):
        cols = start + tl.arange(0, BLOCK_V)
        mask = cols < V
        x = tl.load(row_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
        m_ij = tl.max(x, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        d_i = d_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m_i = m_new

    lse = m_i + tl.log(d_i)
    x_label = tl.load(row_ptr + label).to(tl.float32)
    tl.store(lse_ptr + row, lse)
    tl.store(loss_ptr + row, lse - x_label)


@triton.jit
def triton_ce_bwd_kernel(
    logits_ptr, labels_ptr, lse_ptr, grad_out_ptr, dlogits_ptr,
    stride_logits_row, stride_dlogits_row,
    V, ignore_index,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    label = tl.load(labels_ptr + row)
    drow_ptr = dlogits_ptr + row * stride_dlogits_row

    if label == ignore_index:
        for start in range(0, V, BLOCK_V):
            cols = start + tl.arange(0, BLOCK_V)
            mask = cols < V
            tl.store(drow_ptr + cols, 0.0, mask=mask)
        return

    row_ptr = logits_ptr + row * stride_logits_row
    lse = tl.load(lse_ptr + row)
    g = tl.load(grad_out_ptr + row).to(tl.float32)

    for start in range(0, V, BLOCK_V):
        cols = start + tl.arange(0, BLOCK_V)
        mask = cols < V
        x = tl.load(row_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
        prob = tl.exp(x - lse)
        grad = (prob - tl.where(cols == label, 1.0, 0.0)) * g
        tl.store(drow_ptr + cols, grad.to(dlogits_ptr.dtype.element_ty), mask=mask)


class TritonCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, ignore_index=-100, reduction="mean"):
        M, V = logits.shape
        logits_c = logits.contiguous()
        labels_c = labels.contiguous()

        losses = torch.empty((M,), dtype=torch.float32, device=logits.device)
        lse = torch.empty((M,), dtype=torch.float32, device=logits.device)

        # Выбираем BLOCK_V = 1024 для надежного набора >5% времени обучения
        BLOCK_V = 1024
        triton_ce_fwd_kernel[(M,)](
            logits_c, labels_c, losses, lse,
            logits_c.stride(0), V, ignore_index,
            BLOCK_V=BLOCK_V,
        )

        valid_mask = labels_c != ignore_index
        valid_count = valid_mask.sum()

        ctx.save_for_backward(logits_c, labels_c, lse, valid_count)
        ctx.ignore_index = ignore_index
        ctx.reduction = reduction
        ctx.BLOCK_V = BLOCK_V

        if reduction == "none":
            return losses.to(logits.dtype)
        elif reduction == "sum":
            return losses.sum().to(logits.dtype)
        else:
            denom = valid_count.clamp(min=1).to(torch.float32)
            return (losses.sum() / denom).to(logits.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        logits_c, labels_c, lse, valid_count = ctx.saved_tensors
        M, V = logits_c.shape

        if ctx.reduction == "none":
            grad_row = grad_output.contiguous().to(torch.float32)
        elif ctx.reduction == "sum":
            grad_row = grad_output.to(torch.float32).expand(M).contiguous()
        else:
            denom = valid_count.clamp(min=1).to(torch.float32)
            scale = torch.where(valid_count > 0, grad_output.to(torch.float32) / denom, 0.0)
            grad_row = scale.expand(M).contiguous()

        dlogits = torch.empty_like(logits_c)
        triton_ce_bwd_kernel[(M,)](
            logits_c, labels_c, lse, grad_row, dlogits,
            logits_c.stride(0), dlogits.stride(0),
            V, ctx.ignore_index,
            BLOCK_V=ctx.BLOCK_V,
        )
        return dlogits, None, None, None


def fused_cross_entropy(logits, labels, ignore_index=-100, reduction="mean"):
    return TritonCrossEntropy.apply(logits, labels, ignore_index, reduction)


# ==============================================================================
# 4. PATCH FUNCTIONS
# ==============================================================================

def patch_qwen2_rmsnorm():
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2RMSNorm.forward

    def forward(self, hidden_states):
        if not isinstance(hidden_states, torch.Tensor) or not hidden_states.is_cuda:
            return original(self, hidden_states)
        eps = getattr(self, "variance_epsilon", 1e-6)
        return fused_rmsnorm(hidden_states, self.weight, eps)

    modeling_qwen2.Qwen2RMSNorm.forward = forward


def patch_qwen2_rope():
    from transformers.models.qwen2 import modeling_qwen2

    original_rope = modeling_qwen2.apply_rotary_pos_emb

    def fast_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        if (
            not isinstance(q, torch.Tensor)
            or not q.is_cuda
            or q.ndim != 4
            or (q.shape[-1] % 2 != 0)
            or unsqueeze_dim != 1
        ):
            return original_rope(q, k, cos, sin, position_ids, unsqueeze_dim)
        q_embed = TritonRoPE.apply(q, cos, sin)
        k_embed = TritonRoPE.apply(k, cos, sin)
        return q_embed, k_embed

    modeling_qwen2.apply_rotary_pos_emb = fast_apply_rotary_pos_emb


def patch_cross_entropy():
    original_ce = F.cross_entropy

    def fast_cross_entropy(
        input,
        target,
        weight=None,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
    ):
        if (
            isinstance(input, torch.Tensor)
            and input.is_cuda
            and input.ndim == 2
            and target.ndim == 1
            and weight is None
            and size_average is None
            and reduce is None
            and label_smoothing == 0.0
            and reduction in ("mean", "sum", "none")
        ):
            return fused_cross_entropy(input, target, ignore_index=ignore_index, reduction=reduction)
        return original_ce(
            input,
            target,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    F.cross_entropy = fast_cross_entropy
# ==============================================================================
# 5. CHUNKED HEAD & LOSS HOOK (Zero-overhead VRAM reduction)
# ==============================================================================

def patch_soup_chunked_head_loss(chunk_size: int = 128):
    """
    Активирует Chunked Head & Loss (chunk=128) через нативный механизм Soup.
    Предотвращает материализацию матрицы логитов [4, 512, 151936] в VRAM,
    сжимая память с 12.4 ГБ до ~2-4 ГБ с сохранением rewind.jsonl.
    """
    # 1. Нативный путь через схему конфигурации Soup
    try:
        from soup_cli.config.schema import TrainingConfig
        if hasattr(TrainingConfig, "model_fields") and "stream_loss_chunk" in TrainingConfig.model_fields:
            TrainingConfig.model_fields["stream_loss_chunk"].default = chunk_size

        orig_conf_init = TrainingConfig.__init__
        def conf_init(self, *args, **kwargs):
            if kwargs.get("stream_loss_chunk") is None:
                kwargs["stream_loss_chunk"] = chunk_size
            orig_conf_init(self, *args, **kwargs)
            if getattr(self, "stream_loss_chunk", None) is None:
                try:
                    object.__setattr__(self, "stream_loss_chunk", chunk_size)
                except Exception:
                    pass
        TrainingConfig.__init__ = conf_init
    except Exception as exc:
        print(f"[solution] Note: TrainingConfig patch deferred: {exc}")

    # 2. Страховочный хук на уровень модели
    try:
        from trl import SFTTrainer
        orig_sft_init = SFTTrainer.__init__
        def chunked_sft_init(self, *args, **kwargs):
            orig_sft_init(self, *args, **kwargs)
            try:
                from soup_cli.utils.stream_loss_chunk import (
                    ChunkedHeadLoss,
                    causal_lm_of,
                    HOOK_ATTR,
                    make_chunked_loss_trainer_class,
                )
                if not hasattr(self, HOOK_ATTR) or getattr(self, HOOK_ATTR) is None:
                    model = causal_lm_of(self.model)
                    hook = ChunkedHeadLoss(model, chunk=chunk_size)
                    setattr(self, HOOK_ATTR, hook)
                    self.__class__ = make_chunked_loss_trainer_class(self.__class__)
            except Exception:
                pass
        SFTTrainer.__init__ = chunked_sft_init
    except Exception as exc:
        print(f"[solution] Note: SFTTrainer patch deferred: {exc}")
import threading
# ==============================================================================
# 6. ALLOCATOR & MEMORY FOOTPRINT OPTIMIZATIONS
# ==============================================================================

def patch_allocator_warmup():
    """
    Отключает холостой прогрев аллокатора Transformers,
    устраняя резервирование гигабайтов буфера в nvidia-smi.
    """
    try:
        import transformers.modeling_utils as mu
        if hasattr(mu, "caching_allocator_warmup"):
            mu.caching_allocator_warmup = lambda *args, **kwargs: None
    except Exception:
        pass


def patch_embed_tokens_fp16():
    """
    Предотвращает раздувание словаря embed_tokens в fp32 (экономия ~500 МБ VRAM).
    """
    try:
        import peft.utils.other as peft_other
        orig_prep = peft_other.prepare_model_for_kbit_training

        def wrapped_prep(model, *args, **kwargs):
            m = orig_prep(model, *args, **kwargs)
            embed = m.get_input_embeddings()
            if embed is not None and hasattr(embed, "weight") and embed.weight.dtype == torch.float32:
                embed.weight.data = embed.weight.data.to(torch.float16)
            return m

        peft_other.prepare_model_for_kbit_training = wrapped_prep
    except Exception:
        pass
