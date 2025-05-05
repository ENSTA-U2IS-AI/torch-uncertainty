import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


def packed_linear(
    inputs: Tensor,
    weight: Tensor,
    num_groups: int,
    implementation: str,
    bias: Tensor | None = None,
) -> Tensor:
    r"""Applies a packed linear transformation to the incoming data using the given implementation.

    Args:
        inputs (Tensor): :math:`(\star, \text{in\_features})` where :math:`\star` is any number of
            additional dimensions including none.
        weight (Tensor): :math:(\text{num\_groups}, \frac{\text{out\_features}}{\text{num\_groups}}, \frac{\text{in\_features}}{\text{num\_groups}})`.
        num_groups (int): number of groups to split the input.
        implementation (str): the implementation of the packed linear operation. Three
            implementations are currently supported:
            - "full": creates a block diagonal matrix from the weight tensor and applies the linear
                transformation using `torch.nn.functional.linear`.
            - "sparse": uses a sparse weight tensor directly to apply the linear transformation.
            - "einsum": uses `torch.einsum` to apply the packed linear transformation.
        rearrange (bool, optional): _description_. Defaults to True.
        bias (Tensor | None, optional): _description_. Defaults to None.

    Returns:
        Tensor:
    """
    if implementation == "full":
        block_diag = torch.block_diag(*weight)
        return F.linear(inputs, block_diag, bias)
    if implementation == "sparse":
        out = inputs @ weight.transpose(0, 1)
        if bias is not None:
            out += bias
        return out
    if implementation == "einsum":
        out = torch.einsum(
            "...ki,kij->...kj",
            rearrange(inputs, "... (m d) -> ... m d", m=num_groups),
            weight.transpose(1, 2),
        ).flatten(start_dim=-2)
        if bias is not None:
            out += bias
        return out
    raise ValueError(f"Unknown implementation: {implementation}")


def packed_in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    num_groups: int,
    implementation: str = "full",
    b_q: Tensor | None = None,
    b_k: Tensor | None = None,
    b_v: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    emb_q, emb_k, emb_v = q.size(-1), k.size(-1), v.size(-1)
    assert w_q.shape == (
        num_groups,
        emb_q // num_groups,
        emb_q // num_groups,
    ), f"expecting query weights shape of {(emb_q, emb_q)}, but got {w_q.shape}"
    assert w_k.shape == (
        num_groups,
        emb_q // num_groups,
        emb_k // num_groups,
    ), f"expecting key weights shape of {(emb_q, emb_k)}, but got {w_k.shape}"
    assert w_v.shape == (
        num_groups,
        emb_q // num_groups,
        emb_v // num_groups,
    ), f"expecting value weights shape of {(emb_q, emb_v)}, but got {w_v.shape}"
    assert b_q is None or b_q.shape == (
        emb_q,
    ), f"expecting query bias shape of {(emb_q,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (
        emb_q,
    ), f"expecting key bias shape of {(emb_k,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (
        emb_q,
    ), f"expecting value bias shape of {(emb_v,)}, but got {b_v.shape}"

    return (
        packed_linear(q, w_q, num_groups, implementation, b_q),
        packed_linear(k, w_k, num_groups, implementation, b_k),
        packed_linear(v, w_v, num_groups, implementation, b_v),
    )


def packed_in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    num_groups: int,
    implementation: str = "full",
    b: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    emb = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            proj = packed_linear(
                inputs=q, weight=w, num_groups=num_groups, implementation=implementation, bias=b
            )
            # reshape to 3, emb and not emb, 3 is deliberate for better memory
            # coalescing and keeping same order as chunk()
            proj = (
                proj.unflatten(-1, (3, emb)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
            )
            return proj[0], proj[1], proj[2]

        # encoder-decoder attention
        _tmp_dim = w.size(-1)
        w_q, w_kv = w.split([_tmp_dim, 2 * _tmp_dim], dim=1)
        if b is None:
            b_q = b_kv = None
        else:
            b_q, b_kv = b.split([emb, 2 * emb])
        q_proj = packed_linear(
            inputs=q, weight=w_q, num_groups=num_groups, implementation=implementation, bias=b_q
        )
        kv_proj = packed_linear(
            inputs=k, weight=w_kv, num_groups=num_groups, implementation=implementation, bias=b_kv
        )
        # reshape to 2, emb and not emb, 2 is deliberate for better memory
        # coalescing and keeping same order as chunk()
        kv_proj = (
            kv_proj.unflatten(-1, (2, emb)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
        )
        return q_proj, kv_proj[0], kv_proj[1]

    w_q, w_k, w_v = w.chunk(3, dim=1)
    if b is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = b.chunk(3)
    return (
        packed_linear(
            inputs=q, weight=w_q, num_groups=num_groups, implementation=implementation, bias=b_q
        ),
        packed_linear(
            inputs=k, weight=w_k, num_groups=num_groups, implementation=implementation, bias=b_k
        ),
        packed_linear(
            inputs=v, weight=w_v, num_groups=num_groups, implementation=implementation, bias=b_v
        ),
    )


def packed_multi_head_attention_forward(  # noqa: D417
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    num_groups: int,
    in_proj_weight: Tensor | None,
    in_proj_bias: Tensor | None,
    bias_k: Tensor | None,
    bias_v: Tensor | None,
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor | None,
    implementation: str = "einsum",
    training: bool = True,
    key_padding_mask: Tensor | None = None,
    need_weights: bool = False,  # TODO: add support
    attn_mask: Tensor | None = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Tensor | None = None,
    k_proj_weight: Tensor | None = None,
    v_proj_weight: Tensor | None = None,
    static_k: Tensor | None = None,
    static_v: Tensor | None = None,
    average_attn_weights: bool = True,  # TODO: add support  # noqa: ARG001
    is_causal: bool = False,
) -> tuple[Tensor, Tensor | None]:
    r"""Parallel Multihead Attention (pMHA) with packed inputs.

    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        implementation (str, optional): the implementation of the packed linear operation. Three
            implementations are currently supported:
            - ``"full"``: creates a block diagonal matrix from the weight tensor and applies the
                linear transformation using `torch.nn.functional.linear`.
            - ``"sparse"``: uses a sparse weight tensor directly to apply the linear
                transformation.
            - ``"einsum"``: uses `torch.einsum` to apply the packed linear transformation.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
            Default: `True`
            Note: `needs_weight` defaults to `True`, but should be set to `False`
            For best performance when attention weights are not needed.
            *Setting needs_weights to `True`
            leads to a significant performance degradation.*
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        is_causal: If specified, applies a causal mask as attention mask, and ignores
            attn_mask for computing scaled dot product attention.
            Default: ``False``.
            .. warning::
                is_causal is provides a hint that the attn_mask is the
                causal mask.Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
        average_attn_weights: If ``True``, indicates that the returned ``attn_weights`` should be averaged across heads.
            Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
            when ``need_weights=True.``. Default to ``True``.

    Shape:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a FloatTensor is provided, it will be directly added to the value.
          If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.

    References:
        Implementation of the packed multi-head attention is based on the PyTorch implementation of the
        `torch.nn.MultiheadAttention` module. The implementation is adapted to support packed inputs.
    """
    is_batched = F._mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    key_padding_mask = F._canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_type=F._none_or_dtype(attn_mask),
        other_name="attn_mask",
        target_type=query.dtype,
    )

    if is_causal and attn_mask is None:
        raise RuntimeError(
            "Need attn_mask if specifying the is_causal hint. "
            "You may use the Transformer module method "
            "`generate_square_subsequent_mask` to create this mask."
        )

    if is_causal and key_padding_mask is None and not need_weights:
        # when we have a kpm or need weights, we need attn_mask
        # Otherwise, we use the is_causal hint go as is_causal
        # indicator to SDPA.
        attn_mask = None
    else:
        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        if key_padding_mask is not None:
            # We have the attn_mask, and use that to merge kpm into it.
            # Turn off use of is_causal hint, as the merged mask is no
            # longer causal.
            is_causal = False

    assert (
        embed_dim == embed_dim_to_check
    ), f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
    else:
        head_dim = embed_dim // num_heads
    assert (
        head_dim * num_heads == embed_dim
    ), f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert (
            key.shape[:2] == value.shape[:2]
        ), f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert (
            key.shape == value.shape
        ), f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        assert (
            in_proj_weight is not None
        ), "use_separate_proj_weight is False but in_proj_weight is None"
        q, k, v = packed_in_projection_packed(
            q=query, k=key, v=value, w=in_proj_weight, num_groups=num_groups, b=in_proj_bias
        )
    else:
        assert (
            q_proj_weight is not None
        ), "use_separate_proj_weight is True but q_proj_weight is None"
        assert (
            k_proj_weight is not None
        ), "use_separate_proj_weight is True but k_proj_weight is None"
        assert (
            v_proj_weight is not None
        ), "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)

        q, k, v = packed_in_projection(
            q=query,
            k=key,
            v=value,
            w_q=q_proj_weight,
            w_k=k_proj_weight,
            w_v=v_proj_weight,
            num_groups=num_groups,
            implementation=implementation,
            b_q=b_q,
            b_k=b_k,
            b_v=b_v,
        )

    # prep attention mask
    if attn_mask is not None:
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                )
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                # unreachable code due to the check above (F._mha_shape_check, l.274)
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                )
        else:
            # unreachable code due to the check above (F._mha_shape_check, l.274)
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)], dim=0)
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)], dim=0)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None, "bias_k is not None"
        assert bias_v is None, "bias_v is not None"

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = rearrange(q, "l b (h d) -> b h l d", h=num_heads)
    k = rearrange(k, "s b (h d) -> b h s d", h=num_heads)
    v = rearrange(v, "s b (h d) -> b h s d", h=num_heads)

    if add_zero_attn:
        zero_attn_shape = (bsz, num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=2)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=2)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    src_len = k.size(2)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (
            bsz,
            src_len,
        ), f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = (
            key_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, num_heads, -1, -1)
            .reshape(bsz * num_heads, 1, src_len)
        )
        attn_mask = key_padding_mask if attn_mask is None else attn_mask + key_padding_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    if need_weights:
        raise NotImplementedError("need_weights is not supported yet")

    # attn_mask can be either (L,S) or (N*num_key_heads, L, S)
    # if attn_mask's shape is (1, L, S) we need to unsqueeze to (1, 1, L, S)
    # in order to match the input for SDPA of (N, num_key_heads, L, S)
    if attn_mask is not None:
        if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(0)
        else:
            attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)

    attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)

    attn_output = rearrange(attn_output, "b h l d -> (l b) (h d)")

    attn_output = packed_linear(
        attn_output, out_proj_weight, num_groups, implementation, out_proj_bias
    )

    attn_output = rearrange(attn_output, "(l b) d -> l b d", l=tgt_len)
    if not is_batched:
        # squeeze the output if input was unbatched
        attn_output = attn_output.squeeze(1)
    return attn_output, None
