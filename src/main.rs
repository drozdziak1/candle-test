use anyhow::Result;
use candle_core::{D, DType, Device, Tensor, shape::Dim};
use candle_nn::{AdamW, Linear, Module, Optimizer, VarBuilder, VarMap};

pub struct MLP {
    ln1: Linear,
    ln2: Linear,
    ln3: Linear,
}

impl MLP {
    pub fn new(vs: VarBuilder) -> Result<Self> {
        let ln1 = candle_nn::linear(10, 100, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(100, 100, vs.pp("ln2"))?;
        let ln3 = candle_nn::linear(100, 1, vs.pp("ln3"))?;

        Ok(Self { ln1, ln2, ln3 })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.ln1.forward(xs)?;
        let xs = self.ln2.forward(&xs)?;
        Ok(self.ln3.forward(&xs)?)
    }
}

pub fn gen_train_case(n: usize, d: &Device) -> Result<(Tensor, Tensor)> {
    let x = Tensor::randn(1000.0f32, 1000.0f32, (n, 10), d)?;

    let y = x.sum(D::Minus1)?;

    Ok((x, y))
}

// xs: [1024, 64, 1924], c Tensor[dims 128, 64, 8; f32, cuda:0] Conv1dConfig { padding: 0, stride: 4, dilation: 1, groups: 1 }
fn main() -> Result<()> {
    let device = Device::new_cuda(1)?;

    let varmap = VarMap::new();

    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = MLP::new(vs.clone())?;

    let mut opt = AdamW::new_lr(varmap.all_vars(), 15e-4)?;

    for i in 1..=2_000 {
        let (x, y_gt) = gen_train_case(10_000, &device)?;

        let y_hat = model.forward(&x)?;

        let y_gt = y_gt.unsqueeze(D::Minus1)?;

        let diff = (&y_hat - &y_gt)?;

        let loss = diff.sqr()?.mean_all()?; // MSE

        let acc_scores = (y_hat / y_gt)?
            .broadcast_sub(&Tensor::from_vec(vec![1.0f32], 1, &device)?)?
            .abs()?
            .broadcast_lt(&Tensor::from_vec(vec![0.001f32], 1, &device)?)?
            .to_dtype(DType::F32)?;

        let acc = acc_scores.mean_all()?;

        println!(
            "Step {}: Loss {} | Acc {}",
            i,
            loss.to_scalar::<f32>()?,
            acc.to_scalar::<f32>()?
        );

        opt.backward_step(&loss)?;
    }

    let final_x = Tensor::from_vec((1..=10).map(|i| 100.0 * -i as f32).collect::<Vec<_>>(), 10, &device)?
        .unsqueeze(0)?;

    let final_y = model.forward(&final_x)?.sum_all()?;

    println!(
        "Sum({:?}) = {}",
        final_x.flatten(0, D::Minus1)?.to_vec1::<f32>()?,
        final_y.to_scalar::<f32>()?
    );

    Ok(())
}
