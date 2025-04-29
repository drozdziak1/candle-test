mod data_helpers;
mod transformer;

use anyhow::{Result, anyhow, bail};
use candle_core::{D, DType, Device, Tensor, shape::Dim};
use candle_nn::{AdamW, Linear, Module, Optimizer, VarBuilder, VarMap};
use hf_hub::api::sync::{Api, ApiRepo};
use polars::prelude::*;
use tokenizers::Tokenizer;

use std::path::PathBuf;

use data_helpers::{FWBatchIter, MultiParquetIter};
use transformer::{Transformer, TransformerConfig};

pub fn load_dataset(rng_seed: u64) -> Result<MultiParquetIter> {
    let api = Api::new()?;

    let repo = api.dataset("huggingface/fineweb".to_string());

    Ok(MultiParquetIter::new(
        repo,
        PathBuf::try_from("sample/100BT/")?,
        rng_seed,
    ))
}

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;

    let tokenizer = Tokenizer::from_pretrained("gpt2", None).map_err(|e| anyhow!(e.to_string()))?;

    let varmap = VarMap::new();

    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let cfg = TransformerConfig {
        vocab_size: tokenizer.get_vocab_size(true),
        embed_dim: 768,
        num_heads: 12,
        num_blocks: 12,
        ff_dim: 1024,
    };

    let model = Transformer::new(&cfg, vb, &device)?;

    let mut opt = AdamW::new_lr(varmap.all_vars(), 15e-4)?;

    let mut ds_iter = load_dataset(0xdeadbeef)?;

    let batch_iter = FWBatchIter::new(8, 8, ds_iter, tokenizer)?;

    Ok(())
}
