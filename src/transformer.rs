use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor};
use candle_flash_attn::flash_attn;
use candle_nn::{
    self as nn, Embedding, LayerNorm, LayerNormConfig, Linear, Module, Sequential, VarBuilder,
};

pub const FLOAT_DTYPE: DType = DType::F16;

#[derive(Clone, Debug)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_blocks: usize,
    pub ff_dim: usize,
}

pub struct Transformer {
    pub config: TransformerConfig,
    pub embed: Embedding,
    pub blocks: Sequential,
    pub unembed: Tensor,
}

impl Transformer {
    pub fn new(cfg: &TransformerConfig, vb: VarBuilder, dev: &Device) -> Result<Self> {
        let mut blocks = nn::seq();

        for i in 0..cfg.num_blocks {
            blocks = blocks.add(TBlock::new(cfg, vb.pp(format!("block{}", i)), dev)?);
        }

        Ok(Self {
            config: cfg.clone(),
            embed: nn::embedding(cfg.vocab_size, cfg.embed_dim, vb.pp("embed"))?,
            blocks,
            unembed: vb.get_with_hints_dtype(
                (cfg.embed_dim, cfg.vocab_size),
                "unembed",
                nn::init::DEFAULT_KAIMING_NORMAL,
                FLOAT_DTYPE,
            )?,
        })
    }
}

impl Module for Transformer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor, candle_core::Error> {

	let x_embed = self.embed.forward(xs)?;

	let x_blocks = self.blocks.forward(&x_embed)?;

	let x_unembed = x_blocks.broadcast_matmul(&self.unembed)?;
	
	Ok(x_unembed)
    }
}

pub struct TBlock {
    config: TransformerConfig,
    head_size: usize,
    ln1: LayerNorm,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    ln2: LayerNorm,
    ff1: Linear,
    ff2: Linear,
}

impl TBlock {
    pub fn new(cfg: &TransformerConfig, vb: VarBuilder, dev: &Device) -> Result<Self> {
        if cfg.embed_dim % cfg.num_heads != 0 {
            return Err(anyhow!(
                "num_heads must divide embed_dim evenly ({} % {} = {})",
                cfg.embed_dim,
                cfg.num_heads,
                cfg.embed_dim % cfg.num_heads
            ));
        }
        let w_q = vb.get_with_hints_dtype(
            (cfg.embed_dim, cfg.embed_dim),
            "w_q",
            nn::init::DEFAULT_KAIMING_NORMAL,
            FLOAT_DTYPE,
        )?;
        let w_k = vb.get_with_hints_dtype(
            (cfg.embed_dim, cfg.embed_dim),
            "w_k",
            nn::init::DEFAULT_KAIMING_NORMAL,
            FLOAT_DTYPE,
        )?;
        let w_v = vb.get_with_hints_dtype(
            (cfg.embed_dim, cfg.embed_dim),
            "w_v",
            nn::init::DEFAULT_KAIMING_NORMAL,
            FLOAT_DTYPE,
        )?;

        Ok(Self {
            config: cfg.clone(),
            head_size: cfg.embed_dim / cfg.num_heads,
            ln1: nn::layer_norm(cfg.embed_dim, LayerNormConfig::default(), vb.pp("ln1"))?,
            w_q,
            w_k,
            w_v,
            ln2: nn::layer_norm(cfg.embed_dim, LayerNormConfig::default(), vb.pp("ln2"))?,
            ff1: nn::linear(cfg.embed_dim, cfg.ff_dim, vb.pp("ff1"))?,
            ff2: nn::linear(cfg.ff_dim, cfg.embed_dim, vb.pp("ff2"))?,
        })
    }
}

impl Module for TBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor, candle_core::Error> {
        let x_shape = xs.shape().dims();
        let x_ln1 = self.ln1.forward(xs)?;

        let qkv_shape = (
            x_shape[0],
            x_shape[1],
            self.config.num_heads,
            self.head_size,
        );

        let q = x_ln1.broadcast_matmul(&self.w_q)?.reshape(qkv_shape)?;
        let k = x_ln1.broadcast_matmul(&self.w_k)?.reshape(qkv_shape)?;
        let v = x_ln1.broadcast_matmul(&self.w_v)?.reshape(qkv_shape)?;

        let x_atn = flash_attn(&q, &k, &v, 1.0, true)?.reshape(x_shape)?;

        let xs_with_atn = (xs + x_atn)?;

        let x_ln2 = self.ln2.forward(&xs_with_atn)?;
        let x_ff1 = self.ff1.forward(&x_ln2)?.gelu()?;
        let x_ff2 = self.ff2.forward(&x_ff1)?;

        Ok((xs_with_atn + x_ff2)?)
    }
}

#[cfg(test)]
mod test {
    use candle_core::DType;
    use candle_nn::VarMap;

    use super::*;

    pub const INT_DTYPE: DType = DType::U32;


    #[test]
    fn test_transformer_happy_path() -> Result<()> {
        let cfg = TransformerConfig {
            vocab_size: 16,
            embed_dim: 64,
            num_heads: 4,
            num_blocks: 4,
            ff_dim: 96,
        };

	let device = Device::new_cuda(0)?;

	let varmap = VarMap::new();

	let vb = VarBuilder::from_varmap(&varmap, FLOAT_DTYPE, &device);

	let model = Transformer::new(&cfg, vb, &device)?; 

	let input = Tensor::rand(0f32, cfg.vocab_size as f32, (10, 16), &device)?.to_dtype(INT_DTYPE)?;

	let _output = model.forward(&input)?;

	Ok(())
    }

    #[test]
    fn test_tblock_happy_path() -> Result<()> {
        let cfg = TransformerConfig {
            vocab_size: 16,
            embed_dim: 64,
            num_heads: 4,
            num_blocks: 4,
            ff_dim: 96,
        };

        let dev = Device::new_cuda(0)?;

        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F16, &dev);

        let model = TBlock::new(&cfg, vs, &dev)?;

        let input =
            Tensor::rand(0.0f32, 10.0f32, (10, 16, cfg.embed_dim), &dev)?.to_dtype(DType::F16)?;

        let output = model.forward(&input)?;

        Ok(())
    }
}
