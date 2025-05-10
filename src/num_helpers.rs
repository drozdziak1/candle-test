//! Numerical stability helpers

use candle_core::{DType, Result, Tensor, bail};
use log::trace;

pub fn check_nan_and_inf(t: &Tensor, comment: &str) -> Result<()> {
    // inf unchanged by small mul + add
    let inf_check = t
        .affine(2.0, 1.0)?
        .eq(t)?
        .to_dtype(DType::U32)?
        .sum_all()?
        .to_scalar::<u32>()?;

    // nan not equal even to itself
    let nan_check = t
        .ne(t)?
        .to_dtype(DType::U32)?
        .sum_all()?
        .to_scalar::<u32>()?;

    if nan_check > 0 || inf_check > 0 {
        let maybe_comment = if !comment.is_empty() {
            format!("{}: ", comment)
        } else {
            String::new()
        };
	trace!("Tensor:\n{}", t);
        bail!(
            "{}NaNs and/or infs detected - inf: {}, nan: {}",
            maybe_comment,
            inf_check,
            nan_check
        );
    }

    Ok(())
}

#[cfg(test)]
pub mod test {
    use core::f32;

    use candle_core::Device;

    use super::*;

    #[test]
    pub fn test_check_nan_and_inf_detects_inf() -> Result<()> {
	let t = Tensor::full(f32::INFINITY, (1,), &Device::Cpu)?;

	assert!(check_nan_and_inf(&t, "le teste").is_err());

	Ok(())
    }

    #[test]
    pub fn test_check_nan_and_inf_detects_neg_inf() -> Result<()> {
	let t = Tensor::full(f32::NEG_INFINITY, (1,), &Device::Cpu)?;

	assert!(check_nan_and_inf(&t, "le teste").is_err());

	Ok(())
    }

    #[test]
    pub fn test_check_nan_and_inf_detects_nan() -> Result<()> {
	let t = Tensor::full(f32::NAN, (1,), &Device::Cpu)?;

	assert!(check_nan_and_inf(&t, "le teste").is_err());

	Ok(())
    }
}
