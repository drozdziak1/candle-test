//! Numerical stability helpers

use candle_core::{DType, Result, Tensor, bail};
use log::{trace, debug};

#[cfg(feature = "check_num_stability")]
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

    trace!("Tensor {:?}:\n{}", comment, t);
    if nan_check > 0 || inf_check > 0 {
	debug!("{:?}: fuggg :-------DDDDD", comment);
        bail!(
            "{:?}: NaNs and/or infs detected - inf: {}, nan: {}",
            comment,
            inf_check,
            nan_check
        );
    } else {
	debug!("{:?}: OK", comment);
    }

    Ok(())
}

#[cfg(not(feature = "check_num_stability"))]
pub fn check_nan_and_inf(_t: &Tensor, _comment: &str) -> Result<()> {
    Ok(())
}

#[cfg(test)]
pub mod test {
    use core::f32;

    use candle_core::Device;

    use super::*;

    #[cfg(feature = "check_num_stability")]
    #[test]
    pub fn test_check_nan_and_inf_detects_inf() -> Result<()> {
	let t = Tensor::full(f32::INFINITY, (1,), &Device::Cpu)?;

	assert!(check_nan_and_inf(&t, "le teste").is_err());

	Ok(())
    }

    #[cfg(feature = "check_num_stability")]
    #[test]
    pub fn test_check_nan_and_inf_detects_neg_inf() -> Result<()> {
	let t = Tensor::full(f32::NEG_INFINITY, (1,), &Device::Cpu)?;

	assert!(check_nan_and_inf(&t, "le teste").is_err());

	Ok(())
    }

    #[cfg(feature = "check_num_stability")]
    #[test]
    pub fn test_check_nan_and_inf_detects_nan() -> Result<()> {
	let t = Tensor::full(f32::NAN, (1,), &Device::Cpu)?;

	assert!(check_nan_and_inf(&t, "le teste").is_err());

	Ok(())
    }
}
