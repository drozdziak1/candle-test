use anyhow::Result;
use candle_core::{D, DType, Device, Tensor, shape::Dim};
use candle_nn::{AdamW, Linear, Module, Optimizer, VarBuilder, VarMap};

mod transformer;

// xs: [1024, 64, 1924], c Tensor[dims 128, 64, 8; f32, cuda:0] Conv1dConfig { padding: 0, stride: 4, dilation: 1, groups: 1 }
fn main() -> Result<()> {
    let device = Device::new_cuda(1)?;

    let varmap = VarMap::new();

    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let mut opt = AdamW::new_lr(varmap.all_vars(), 15e-4)?;

    Ok(())
}
