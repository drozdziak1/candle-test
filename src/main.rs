mod data_helpers;
mod my_loss;
mod my_ops;
mod num_helpers;
mod transformer;

use anyhow::{Result, anyhow, bail};
use candle_core::{backprop::GradStore, shape::Dim, DType, Device, IndexOp, Tensor, D};
use candle_nn::{self as nn, AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use clap::Parser;
use hf_hub::api::sync::{Api, ApiRepo};
use log::{LevelFilter, info};
use polars::prelude::*;
use tokenizers::Tokenizer;

use core::f64;
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
    #[arg(short, long, default_value_t = 1)]
    batch_size: usize,
    #[arg(short, long, default_value_t = 10)]
    minibatch_size: usize,
    #[arg(short, long, default_value_t = 1024)]
    ctx_size: usize,
    #[arg(short, long, default_value_t = 50)]
    val_interval: usize,
    #[arg(short, long, default_value_t = 600_000)]
    n_training_steps: usize,

    #[arg(long, default_value_t = 6e-4)]
    max_lr: f64,

    #[arg(long, default_value_t = 6e-5)]
    min_lr: f64,

    #[arg(long, default_value_t = 2_000)]
    warmup_rounds: usize,

    #[arg(short, long, default_value_t = 1e-5)]
    epsilon: f64,

    #[arg(short, long, default_value_t = 0xdeadbeef)]
    rng_seed: u64,
}

pub fn get_lr(iter: usize, cli: &Cli) -> f64 {
    match iter {
	n if n < cli.warmup_rounds => cli.max_lr * (n + 1) as f64 / (cli.warmup_rounds + 1) as f64,
	n if n > cli.n_training_steps => cli.min_lr,
	other => {
	    let decay_ratio = (other - cli.warmup_rounds) as f64 / (cli.n_training_steps - cli.warmup_rounds) as f64;
	    let coeff = 0.5f64 * (1.0 + (f64::consts::PI * decay_ratio).cos());

	    cli.min_lr + coeff * (cli.max_lr - cli.min_lr)
	}
    }
}

pub fn t_v_step(
    model: &Transformer,
    batch: &[Vec<u32>],
    cli: &Cli,
) -> Result<(Tensor, Tensor, BTreeSet<u32>)> {
    let mut total_loss = Tensor::zeros((), FLOAT_DTYPE, model.embed.embeddings().device())?;

    let mut total_acc_scores = Tensor::zeros(
        (cli.minibatch_size, cli.ctx_size),
        INT_DTYPE,
        model.embed.embeddings().device(),
    )?;
    let mut total_acc_hits_by_id = BTreeSet::new();

    for mbatch in batch {
        let mbatch_tensor = Tensor::from_slice(
            mbatch.as_slice(),
            (cli.minibatch_size, cli.ctx_size + 1),
            model.embed.embeddings().device(),
        )?;

        let last_dim_len = mbatch_tensor
            .shape()
            .dims()
            .last()
            .expect("INTERNAL: There must be at least one dimension in X shape")
            .clone();

        let x = mbatch_tensor.i((.., ..last_dim_len - 1))?;
        let y_gt = mbatch_tensor.i((.., 1..))?;

        let y_hat = model.forward(&x)?;

        let loss = my_loss::cross_entropy(
            &y_hat.reshape(((), model.config.vocab_size))?,
            &y_gt.flatten_all()?,
            cli.epsilon,
        )?.affine(1.0 / batch.len() as f64, 0.0)?;


        let acc_scores = y_hat.argmax(D::Minus1)?.eq(&y_gt)?.to_dtype(INT_DTYPE)?;

        let misses =
            Tensor::zeros_like(&y_gt)?.affine(0.0, (model.config.vocab_size + 1) as f64)?;

        let mut acc_hits_by_id: BTreeSet<u32> = acc_scores
            .where_cond(&y_gt, &misses)?
            .to_dtype(INT_DTYPE)?
            .flatten_all()?
            .to_vec1()?
            .into_iter()
            .collect();

        total_loss = total_loss.add(&loss)?;

        total_acc_scores = total_acc_scores.add(&acc_scores)?;

        total_acc_hits_by_id.append(&mut acc_hits_by_id);
    }

    total_acc_hits_by_id.remove(&(model.config.vocab_size as u32 + 1));

    Ok((
        total_loss.to_dtype(DType::F32)?,
        total_acc_scores.to_dtype(DType::F32)?,
        total_acc_hits_by_id,
    ))
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
        eps: cli.epsilon,
    };

    let model = Transformer::new(&cfg, vb, &device)?;

    let optim_params = ParamsAdamW {
        lr: get_lr(0, &cli),
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

    for (idx, t_batch) in batch_iter.enumerate() {
	
        let (t_loss, t_acc_scores, t_acc_hits_by_id) = t_v_step(&model, t_batch.as_slice(), &cli)?;

        let t_acc = t_acc_scores.mean_all()?;

        {
	    let grad_store = t_loss.backward()?;

            optim.step(&grad_store)?;
	}
	
        let t_unique_acc_hit_cnt = t_acc_hits_by_id.len();

        let t_acc_hits_decoded = tokenizer
            .decode(
                Vec::from_iter(t_acc_hits_by_id.into_iter()).as_slice(),
                true,
            )
            .map_err(|e| anyhow!(e.to_string()))?;

        info!(
            "T Step {:6} of {:6} | T loss {:2.6} | T acc {:1.5} ({:5} of {:6}) | {:5} unique hits: {:?}",
            idx + 1,
            cli.n_training_steps,
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

        if idx % cli.val_interval == 0 {
            let (v_loss, v_acc_scores, v_acc_hits_by_id) =
                t_v_step(&model, v_batch.as_slice(), &cli)?;

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

	optim.set_learning_rate(get_lr(idx, &cli));
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
