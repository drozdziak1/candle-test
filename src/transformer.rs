use core::f64;

use anyhow::{Result, anyhow, bail};
use candle_core::{D, DType, Device, Tensor};
use candle_flash_attn::flash_attn;
use candle_nn::{
    self as nn, Embedding, LayerNorm, LayerNormConfig, Linear, Module, Sequential, VarBuilder,
};

use crate::num_helpers::check_nan_and_inf;

pub const FLOAT_DTYPE: DType = DType::BF16;
pub const INT_DTYPE: DType = DType::U32;

#[derive(Clone, Debug)]
pub struct TransformerConfig {
    pub ctx_size: usize,
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_blocks: usize,
    pub ff_dim: usize,
    pub is_causal: bool,
}

pub struct Transformer {
    pub config: TransformerConfig,
    pub embed: Embedding,
    pub pos_embed: Tensor,
    pub blocks: Sequential,
}

impl Transformer {
    pub fn new(cfg: &TransformerConfig, vb: VarBuilder, dev: &Device) -> Result<Self> {
        let mut blocks = nn::seq();

        for i in 0..cfg.num_blocks {
            blocks = blocks.add(TBlock::new(cfg, vb.pp(format!("block{}", i)), dev)?);
        }
	let w_embed_unembed = vb.get_with_hints(
		(cfg.vocab_size, cfg.embed_dim),
		"w_embed_unembed",
		nn::init::DEFAULT_KAIMING_UNIFORM,
	    )?;

	let embed = Embedding::new(w_embed_unembed, cfg.embed_dim);

	check_nan_and_inf(embed.embeddings(), "embed init")?;

        Ok(Self {
            config: cfg.clone(),
            embed,
            pos_embed: vb.get_with_hints(
                (cfg.ctx_size, cfg.embed_dim),
                "pos_embed",
                nn::init::DEFAULT_KAIMING_UNIFORM,
            )?,
            blocks,
        })
    }
}

impl Module for Transformer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor, candle_core::Error> {
        let x_embed = self.embed.forward(xs)?;

	check_nan_and_inf(&x_embed, "model x_embed")?;

        let x_pos_embed = x_embed.add(&self.pos_embed.expand(x_embed.shape())?)?;

	check_nan_and_inf(&x_pos_embed, "model x_pos_embed")?;

        let x_blocks = self.blocks.forward(&x_pos_embed)?;

	check_nan_and_inf(&x_blocks, "model x_blocks")?;

        let x_unembed = x_blocks.broadcast_matmul(&self.embed.embeddings().t()?)?;

	check_nan_and_inf(&x_unembed, "model x_unembed")?;

        let x_out = nn::ops::softmax(&x_unembed, D::Minus1)?;

	check_nan_and_inf(&x_out, "model x_out")?;

        Ok(x_out)
    }
}

pub struct TBlock {
    config: TransformerConfig,
    ln1: LayerNorm,
    atn: MHSA,
    ln2: LayerNorm,
    ff1: Linear,
    ff2: Linear,
}

impl TBlock {
    pub fn new(cfg: &TransformerConfig, vb: VarBuilder, dev: &Device) -> Result<Self> {
        Ok(Self {
            config: cfg.clone(),
            ln1: nn::layer_norm(cfg.embed_dim, LayerNormConfig::default(), vb.pp("ln1"))?,
            atn: MHSA::new(cfg, vb.pp("atn"), dev)?,
            ln2: nn::layer_norm(cfg.embed_dim, LayerNormConfig::default(), vb.pp("ln2"))?,
            ff1: nn::linear(cfg.embed_dim, cfg.ff_dim, vb.pp("ff1"))?,
            ff2: nn::linear(cfg.ff_dim, cfg.embed_dim, vb.pp("ff2"))?,
        })
    }
}

impl Module for TBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor, candle_core::Error> {
        let x_ln1 = self.ln1.forward(xs)?;
	check_nan_and_inf(&x_ln1, "x_ln1")?;

        let x_atn = self.atn.forward(&x_ln1)?;
	check_nan_and_inf(&x_atn, "x_atn")?;

        let xs_with_atn = (xs + x_atn)?;
	check_nan_and_inf(&xs_with_atn, "xs_with_atn")?;

        let x_ln2 = self.ln2.forward(&xs_with_atn)?;
	check_nan_and_inf(&x_ln2, "x_ln2")?;

        let x_ff1 = self.ff1.forward(&x_ln2)?.gelu()?;
	check_nan_and_inf(&x_ff1, "x_ff1")?;

        let x_ff2 = self.ff2.forward(&x_ff1)?;
	check_nan_and_inf(&x_ff2, "x_ff2")?;


        Ok((xs_with_atn + x_ff2)?)
    }
}

pub struct MHSA {
    config: TransformerConfig,
    head_size: usize,
    mask: Tensor,
    w_qkv: Linear,
    w_atn_out: Linear,
}

impl MHSA {
    pub fn new(cfg: &TransformerConfig, vb: VarBuilder, dev: &Device) -> Result<Self> {
        if cfg.embed_dim % cfg.num_heads != 0 {
            return Err(anyhow!(
                "num_heads must divide embed_dim evenly ({} % {} = {})",
                cfg.embed_dim,
                cfg.num_heads,
                cfg.embed_dim % cfg.num_heads
            ));
        }

        let zeros = Tensor::zeros((cfg.ctx_size, cfg.ctx_size), FLOAT_DTYPE, dev)?;

        let mask = if cfg.is_causal {
            let infinities = Tensor::full(f64::NEG_INFINITY, zeros.shape(), dev)?.to_dtype(FLOAT_DTYPE)?;

            Tensor::tril2(cfg.ctx_size, INT_DTYPE, dev)?
                .where_cond(&infinities, &zeros)?
                .to_dtype(FLOAT_DTYPE)?
        } else {
            zeros
        };

        Ok(Self {
            config: cfg.clone(),
            head_size: cfg.embed_dim / cfg.num_heads,
            mask,
            w_qkv: nn::linear(cfg.embed_dim, cfg.embed_dim * 3, vb.pp("w_qkv"))?,
            w_atn_out: nn::linear(cfg.embed_dim, cfg.embed_dim, vb.pp("w_atn_out"))?,
        })
    }
}

impl Module for MHSA {
    fn forward(&self, xs: &Tensor) -> Result<Tensor, candle_core::Error> {
        let xs_shape = xs.shape().dims();

        let x_qkv_chunks = self.w_qkv.forward(xs)?.chunk(3, D::Minus1)?;

        let x_q = x_qkv_chunks[0]
            .reshape((
                xs_shape[0],
                xs_shape[1],
                self.config.num_heads,
                self.head_size,
            ))?
            .transpose(1, 2)?
            .contiguous()?;

        let x_k = x_qkv_chunks[1]
            .reshape((
                xs_shape[0],
                xs_shape[1],
                self.config.num_heads,
                self.head_size,
            ))?
            .transpose(1, 2)?
            .contiguous()?;

        let x_v = x_qkv_chunks[2]
            .reshape((
                xs_shape[0],
                xs_shape[1],
                self.config.num_heads,
                self.head_size,
            ))?
            .transpose(1, 2)?
            .contiguous()?;

        let xqxk = x_q.matmul(&x_k.t()?)?;

        let xqxk_masked = xqxk.broadcast_add(&self.mask)?;

        let xqxk_scaled = xqxk_masked.affine(1.0 / (self.head_size as f64).sqrt(), 0.0)?;

        let x_atn = nn::ops::softmax_last_dim(&xqxk_scaled)?.matmul(&x_v)?;

        let x_atn_reshaped = x_atn.transpose(1, 2)?.reshape(xs_shape)?;

        let x_out = self.w_atn_out.forward(&x_atn_reshaped)?;

        Ok(x_out)
    }
}

#[cfg(test)]
mod test {
    use candle_core::DType;
    use candle_nn::VarMap;

    use super::*;

    #[test]
    fn test_transformer_happy_path() -> Result<()> {
        let cfg = TransformerConfig {
            ctx_size: 32,
            vocab_size: 16,
            embed_dim: 64,
            num_heads: 4,
            num_blocks: 4,
            ff_dim: 96,
            is_causal: true,
        };

        let device = Device::new_cuda(0)?;

        let varmap = VarMap::new();

        let vb = VarBuilder::from_varmap(&varmap, FLOAT_DTYPE, &device);

        let model = Transformer::new(&cfg, vb, &device)?;

        let input =
            Tensor::rand(0f32, cfg.vocab_size as f32, (10, 16), &device)?.to_dtype(INT_DTYPE)?;

        let _output = model.forward(&input)?;

        Ok(())
    }

    #[test]
    fn test_tblock_happy_path() -> Result<()> {
        let cfg = TransformerConfig {
            ctx_size: 32,
            vocab_size: 16,
            embed_dim: 64,
            num_heads: 4,
            num_blocks: 4,
            ff_dim: 96,
            is_causal: true,
        };

        let dev = Device::new_cuda(0)?;

        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F16, &dev);

        let model = TBlock::new(&cfg, vs, &dev)?;

        let input = Tensor::rand(0.0f32, 10.0f32, (10, cfg.ctx_size, cfg.embed_dim), &dev)?
            .to_dtype(DType::F16)?;

        let _output = model.forward(&input)?;

        Ok(())
    }

    #[test]
    fn test_mhsa_happy_path() -> Result<()> {
        let cfg = TransformerConfig {
            ctx_size: 32,
            vocab_size: 16,
            embed_dim: 64,
            num_heads: 4,
            num_blocks: 4,
            ff_dim: 96,
            is_causal: true,
        };

        let dev = Device::new_cuda(0)?;

        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F16, &dev);

        let model = MHSA::new(&cfg, vs, &dev)?;

        let input = Tensor::rand(0.0f32, 10.0f32, (10, cfg.ctx_size, cfg.embed_dim), &dev)?
            .to_dtype(DType::F16)?;

        let _output = model.forward(&input)?;

        Ok(())
    }
}
