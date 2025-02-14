from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from diffusers.models.attention import Attention
from diffusers.models.embeddings import apply_rotary_emb


def enhance_score(query_image, key_image, head_dim, num_frames, enhance_weight, is_return_attention: bool = False):
    scale = head_dim**-0.5
    query_image = query_image * scale
    attn_temp = query_image @ key_image.transpose(-2, -1)  # translate attn to float32
    attn_temp = attn_temp.to(torch.float32)
    attn_temp = attn_temp.softmax(dim=-1)

    # Reshape to [batch_size * num_tokens, num_frames, num_frames]
    attn_temp = attn_temp.reshape(-1, num_frames, num_frames)

    # Create a mask for diagonal elements
    diag_mask = torch.eye(num_frames, device=attn_temp.device).bool()
    diag_mask = diag_mask.unsqueeze(0).expand(attn_temp.shape[0], -1, -1)

    # Zero out diagonal elements
    attn_wo_diag = attn_temp.masked_fill(diag_mask, 0)

    # Calculate mean for each token's attention matrix
    # Number of off-diagonal elements per matrix is n*n - n
    num_off_diag = num_frames * num_frames - num_frames
    mean_scores = attn_wo_diag.sum(dim=(1, 2)) / num_off_diag

    enhance_scores = mean_scores.mean() * (num_frames + enhance_weight)
    enhance_scores = enhance_scores.clamp(min=1)

    # if enhance_scores > 1:
    #     print(f"ENHANCE! {enhance_scores}")

    if is_return_attention:
        return enhance_scores, attn_temp
    else:
        return enhance_scores


class HunyuanAttnProcessor2_0_EnhanceAVideo:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the HunyuanDiT model. It applies a s normalization layer and rotary embedding on query and key vector.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def get_enhance_scores(self, attn, query, key, num_frames, enhance_weight, is_return_attention=False):
        batch_size, num_heads, ST, head_dim = query.shape
        spatial_dim = ST / num_frames
        spatial_dim = int(spatial_dim)

        query_image = rearrange(
            query, "B N (T S) C -> (B S) N T C", T=num_frames, S=spatial_dim, N=num_heads, C=head_dim
        )
        key_image = rearrange(key, "B N (T S) C -> (B S) N T C", T=num_frames, S=spatial_dim, N=num_heads, C=head_dim)

        return enhance_score(query_image, key_image, head_dim, num_frames, enhance_weight, is_return_attention)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        # ========== Enhance-A-Video ==========
        enhance_a_video_enabled: bool = False,
        enhance_a_video_weight: float = 0.0,
        num_frames: int = 1,
        # ========== Enhance-A-Video ==========
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # ========== Enhance-A-Video ==========
        if enhance_a_video_enabled:
            enhance_scores = self.get_enhance_scores(attn, query, key, num_frames, enhance_a_video_weight)
        # ========== Enhance-A-Video ==========

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = apply_rotary_emb(key, image_rotary_emb)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        # ========== Enhance-A-Video ==========
        if enhance_a_video_enabled:
            hidden_states = hidden_states * enhance_scores
        # ========== Enhance-A-Video ==========

        return hidden_states
