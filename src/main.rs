mod data_helpers;
mod my_loss;
mod num_helpers;
mod transformer;

use anyhow::{Result, anyhow, bail};
use candle_core::{D, DType, Device, IndexOp, Tensor, shape::Dim};
use candle_nn::{self as nn, AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use clap::Parser;
use hf_hub::api::sync::{Api, ApiRepo};
use log::{LevelFilter, info};
use polars::prelude::*;
use tokenizers::Tokenizer;

use std::{collections::BTreeSet, path::PathBuf};

use data_helpers::{BatchIter, MultiParquetIter};
use transformer::{FLOAT_DTYPE, INT_DTYPE, Transformer, TransformerConfig};

pub fn load_dataset(rng_seed: u64) -> Result<MultiParquetIter> {
    let api = Api::new()?;

    let repo = api.dataset("HuggingFaceFW/fineweb".to_string());

    Ok(MultiParquetIter::new(
        repo,
        PathBuf::try_from("sample/100BT/")?,
        rng_seed,
    ))
}

#[derive(Parser)]
#[command(version, about)]
pub struct Cli {
    #[arg(short, long, default_value_t = 128)]
    batch_size: usize,
    #[arg(short, long, default_value_t = 4)]
    minibatch_size: usize,
    #[arg(short, long, default_value_t = 1024)]
    ctx_size: usize,
    #[arg(short, long, default_value_t = 50)]
    val_interval: usize,
    #[arg(short, long, default_value_t = 600_000)]
    n_training_steps: usize,

    #[arg(short, long, default_value_t = 1e-5)]
    epsilon: f64,

    #[arg(short, long, default_value_t = 0xdeadbeef)]
    rng_seed: u64,
}

pub fn t_v_step(
    model: &Transformer,
    minibatch: &Tensor,
    eps: f64,
) -> Result<(Tensor, Tensor, BTreeSet<u32>)> {
    let last_dim_len = minibatch
        .shape()
        .dims()
        .last()
        .expect("INTERNAL: There must be at least one dimension in X shape")
        .clone();

    let x = minibatch.i((.., ..last_dim_len - 1))?;
    let y_gt = minibatch.i((.., 1..))?;

    let y_hat = model.forward(&x)?;

    let loss = my_loss::cross_entropy(
        &y_hat.reshape(((), model.config.vocab_size))?,
        &y_gt.flatten_all()?,
	eps,
    )?;

    let acc_scores = y_hat.argmax(D::Minus1)?.eq(&y_gt)?.to_dtype(INT_DTYPE)?;

    let misses = Tensor::zeros_like(&y_gt)?.affine(0.0, (model.config.vocab_size + 1) as f64)?;

    let mut acc_hits_by_id: BTreeSet<u32> = acc_scores
        .where_cond(&y_gt, &misses)?
        .to_dtype(INT_DTYPE)?
        .flatten_all()?
        .to_vec1()?
        .into_iter()
        .collect();

    acc_hits_by_id.remove(&(model.config.vocab_size as u32 + 1));

    Ok((loss.to_dtype(DType::F32)?, acc_scores.to_dtype(DType::F32)?, acc_hits_by_id))
}

fn main() -> Result<()> {
    init_logging();

    let cli = Cli::parse();

    let device = Device::new_cuda(1)?;

    let tokenizer = Tokenizer::from_pretrained("gpt2", None).map_err(|e| anyhow!(e.to_string()))?;

    let varmap = VarMap::new();

    let vb = VarBuilder::from_varmap(&varmap, FLOAT_DTYPE, &device);

    let cfg = TransformerConfig {
        ctx_size: cli.ctx_size,
        vocab_size: tokenizer.get_vocab_size(true),
        embed_dim: 768,
        num_heads: 12,
        num_blocks: 2,
        ff_dim: 768 * 4,
        is_causal: true,
    };

    let model = Transformer::new(&cfg, vb, &device)?;

    let optim_params = ParamsAdamW {
        lr: 6e-4,
        eps: cli.epsilon,
        weight_decay: 0.1,
        ..Default::default()
    };

    let mut optim = AdamW::new(varmap.all_vars(), optim_params)?;

    let mut ds_iter = load_dataset(cli.rng_seed)?;

    let mut batch_iter = BatchIter::new(
        cli.batch_size,
        cli.minibatch_size,
        cli.ctx_size + 1,
        ds_iter,
        &tokenizer,
    )?;

    let v_batch = batch_iter
        .next()
        .expect("Not a single batch could be extracted from dataset");

    let v_batch_tensor = Tensor::from_vec(v_batch, (cli.minibatch_size, cli.ctx_size + 1), &device)?;

    let mut embed_sum = 0.0f64;

    for (idx, t_batch) in batch_iter.enumerate() {
        let t_minibatch_tensor =
            Tensor::from_vec(t_batch, (cli.minibatch_size, cli.ctx_size + 1), &device)?;

        let (t_loss, t_acc_scores, t_acc_hits_by_id) = t_v_step(&model, &t_minibatch_tensor, cli.epsilon)?;

        let t_acc = t_acc_scores.mean_all()?;

        let grad_store = t_loss.backward()?;

        optim.step(&grad_store)?;

        let t_unique_acc_hit_cnt = t_acc_hits_by_id.len();

        let t_acc_hits_decoded = tokenizer
            .decode(
                Vec::from_iter(t_acc_hits_by_id.into_iter()).as_slice(),
                true,
            )
            .map_err(|e| anyhow!(e.to_string()))?;


	let new_embed_sum: f64 = model.embed.embeddings().to_dtype(DType::F64)?.sum_all()?.to_scalar()?;

        info!(
            "T Step {:6} of {:6} | Delta {:8.4} | T loss {:2.6} | T acc {:1.5} ({:5} of {:6}) | {:5} unique hits: {:?}",
            idx + 1,
            cli.n_training_steps,
	    new_embed_sum - embed_sum,
            t_loss.to_scalar::<f32>()?,
            t_acc.to_scalar::<f32>()?,
            t_acc_scores
                .to_dtype(INT_DTYPE)?
                .sum_all()?
                .to_scalar::<u32>()?,
            cli.batch_size * cli.minibatch_size * cli.ctx_size,
            t_unique_acc_hit_cnt,
            t_acc_hits_decoded,
        );

	embed_sum = new_embed_sum;

        if idx % cli.val_interval == 0 {
            let (v_loss, v_acc_scores, v_acc_hits_by_id) = t_v_step(&model, &v_batch_tensor, cli.epsilon)?;

            let v_acc = v_acc_scores.mean_all()?;

            let v_unique_acc_hit_cnt = v_acc_hits_by_id.len();

            let v_acc_hits_decoded = tokenizer
                .decode(
                    Vec::from_iter(v_acc_hits_by_id.into_iter()).as_slice(),
                    true,
                )
                .map_err(|e| anyhow!(e.to_string()))?;

            info!(
            "V Step {:6} of {:6} | V loss {:2.4} | V acc {:1.5} ({:5} of {:6}) | {:5} unique hits: {:?}",
                idx + 1,
                cli.n_training_steps,
                v_loss.to_scalar::<f32>()?,
                v_acc.to_scalar::<f32>()?,
                v_acc_scores
                    .to_dtype(INT_DTYPE)?
                    .sum_all()?
                    .to_scalar::<u32>()?,
                cli.batch_size * cli.minibatch_size * cli.ctx_size,
                v_unique_acc_hit_cnt,
                v_acc_hits_decoded,
            );
        }
    }

    Ok(())
}

pub fn init_logging() {
    if std::env::var("RUST_LOG").is_ok() {
        env_logger::init();
    } else {
        env_logger::builder().filter_level(LevelFilter::Info).init();
    }
}
