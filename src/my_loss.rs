//! Loss Calculations
//!
//! Taken verbatim from `candle_nn::loss` upstream.  ~drozdziak1

use candle_core::{Result, Tensor, D};

use crate::num_helpers::check_nan_and_inf;

/// The negative log likelihood loss.
///
/// Arguments
///
/// * [inp]: The input tensor of dimensions `N, C` where `N` is the batch size and `C` the number
///   of categories. This is expected to contain log probabilities.
/// * [target]: The ground truth labels as a tensor of u32 of dimension `N`.
///
/// The resulting tensor is a scalar containing the average value over the batch.
pub fn nll(inp: &Tensor, target: &Tensor, eps: f64) -> Result<Tensor> {
    let b_sz = match target.dims() {
        &[b_sz] => b_sz,
        dims => candle_core::bail!("the target tensor should have a single dimension ({dims:?})"),
    };
    match inp.dims() {
        &[inp_b_sz, _] => {
            if inp_b_sz != b_sz {
                candle_core::bail!(
                    "batch size mismatch between inp ({inp_b_sz}) and target ({b_sz})"
                )
            }
        }
        dims => candle_core::bail!("the target tensor should have two dimensions ({dims:?})"),
    }
    let gathered = inp.gather(&target.unsqueeze(1)?, 1)?;

    check_nan_and_inf(&gathered, "gathered")?;

    dbg!(&inp);
    dbg!(&target);
    dbg!(&gathered.flatten_all()?.max(0));
    dbg!(&gathered.flatten_all()?.min(0));

    let summed = gathered.sum_all()?;

    check_nan_and_inf(&summed, "summed")?;

    summed.affine(-1f64 / (b_sz as f64) + eps, 0.)
}

/// The cross-entropy loss.
///
/// Arguments
///
/// * [inp]: The input tensor of dimensions `N, C` where `N` is the batch size and `C` the number
///   of categories. This is expected to raw logits.
/// * [target]: The ground truth labels as a tensor of u32 of dimension `N`.
///
/// The resulting tensor is a scalar containing the average value over the batch.
pub fn cross_entropy(inp: &Tensor, target: &Tensor, eps: f64) -> Result<Tensor> {
    if inp.rank() != 2 {
        candle_core::bail!("cross_entropy expects an input tensor of rank 2")
    }
    let inp = candle_nn::ops::log_softmax(inp, 1)?;

    check_nan_and_inf(&inp, "log_softmax")?;

    let nll_out = nll(&inp, target, eps)?;

    check_nan_and_inf(&nll_out, "nll")?;

    Ok(nll_out)
}

/// The mean squared error loss.
pub fn mse(inp: &Tensor, target: &Tensor) -> Result<Tensor> {
    (inp - target)?.sqr()?.mean_all()
}

/// The binary cross-entropy with logit loss.
///
/// Arguments
///
/// * [inp]: The input tensor of dimensions `N, C` where `N` is the batch size and `C` the number
///   of categories. This is expected to raw logits.
/// * [target]: The ground truth labels as a tensor of u32 of dimension `N, C` where `N` is the batch size and `C` the number
///   of categories.
///
/// The resulting tensor is a scalar containing the average value over the batch.
pub fn binary_cross_entropy_with_logit(inp: &Tensor, target: &Tensor) -> Result<Tensor> {
    let inp = candle_nn::ops::sigmoid(inp)?;

    let left_side = target * inp.log()?;
    let right_side = (target.affine(-1., 1.))? * inp.affine(-1., 1.)?.log()?;

    let loss = left_side? + right_side?;
    let loss = loss?.neg()?.mean_all()?;

    Ok(loss)
}
