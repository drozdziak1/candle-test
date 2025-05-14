use candle_core::{Result, Tensor, backprop::GradStore};

/// Taken verbatim from candle_core::nn::ops and modified for stability
pub fn softmax<D: candle_core::shape::Dim>(xs: &Tensor, dim: D, eps: f64) -> Result<Tensor> {
    let dim = dim.to_index(xs.shape(), "softmax")?;
    let max = xs.max_keepdim(dim)?;
    let diff = xs.broadcast_sub(&max)?;
    let num = diff.exp()?;
    let den = num.sum_keepdim(dim)?;
    num.broadcast_div(&den)
}
