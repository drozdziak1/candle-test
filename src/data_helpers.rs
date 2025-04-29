use anyhow::{Result, anyhow, bail};
use hf_hub::api::sync::ApiRepo;
use polars::prelude::*;
use tokenizers::Tokenizer;

use std::path::PathBuf;

pub struct MultiParquetIter {
    repo: ApiRepo,
    dir: PathBuf,
    rng_seed: u64,
    next_df_i: usize,
    next_df_j: usize,
    cur_parquet_df: Option<DataFrame>,
}

impl MultiParquetIter {
    pub fn next_parquet_df(&mut self) -> Result<()> {
        let mut inc_i_tried = false;
        loop {
            let mut filename = self.dir.clone();

            filename.push(format!(
                "{:03}_{:05}.parquet",
                self.next_df_i, self.next_df_j
            ));

            if let Ok(local_path) = self.repo.get(
                filename
                    .to_str()
                    .expect("INTERNAL: filename PathBuf::to_str() should work"),
            ) {
                let df = LazyFrame::scan_parquet(local_path, Default::default())?.collect()?;

                self.cur_parquet_df = Some(df);

                self.next_df_j += 1;

                return Ok(());
            } else if !inc_i_tried {
                self.next_df_i += 1;
                self.next_df_j = 0;
                inc_i_tried = true;
            } else {
                bail!("It appears we've reached the end");
            }
        }
    }
    pub fn new(repo: ApiRepo, dir: PathBuf, rng_seed: u64) -> Self {
        Self {
            repo,
            dir,
            rng_seed,
            next_df_i: 0,
            next_df_j: 0,
            cur_parquet_df: None,
        }
    }
}

impl Iterator for MultiParquetIter {
    type Item = String;
    fn next(&mut self) -> Option<Self::Item> {
        if self.cur_parquet_df.is_none() {
            self.next_parquet_df().ok()?;
        }

        let mut cur_parquet_df = self
            .cur_parquet_df
            .as_mut()
            .expect("INTERNAL: We've literally just filled cur_parquet_df");

        match cur_parquet_df.sample_n_literal(
            1,
            false,
            true,
            Some(self.rng_seed + self.next_df_i as u64 + self.next_df_j as u64),
        ) {
            Ok(record) => todo!(),
            Err(e) => todo!(),
        }
    }
}

pub struct FWBatchIter {
    batch_size: usize,
    batch_item_size: usize,
    separator: Vec<u32>,
    multi_parquet_iter: MultiParquetIter,
    tokenizer: Tokenizer,
    buffers: Vec<Vec<u32>>,
}

impl FWBatchIter {
    pub fn new(
        batch_size: usize,
        batch_item_size: usize,
        multi_parquet_iter: MultiParquetIter,
        tokenizer: Tokenizer,
    ) -> Result<Self> {
        let separator = tokenizer
            .encode("<|end_of_text|>", true)
            .map_err(|e| anyhow!(e.to_string()))?
            .get_ids()
            .to_owned();

        Ok(Self {
            batch_size,
            batch_item_size,
            separator,
            multi_parquet_iter,
            tokenizer,
            buffers: vec![vec![]; batch_size],
        })
    }
}

impl Iterator for FWBatchIter {
    type Item = Vec<Vec<u32>>;
    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = vec![Vec::with_capacity(self.batch_item_size); self.batch_size];

        for (batch_item, buf) in batch.iter_mut().zip(self.buffers.iter_mut()) {
            while buf.len() < self.batch_item_size {
                let new_chunk = self
                    .tokenizer
                    .encode(self.multi_parquet_iter.next()?, true)
                    .ok()?
                    .get_ids()
                    .to_owned();

                buf.extend_from_slice(self.separator.as_slice());
                buf.extend_from_slice(new_chunk.as_slice());
            }

            batch_item.extend_from_slice(&buf[..self.batch_item_size]);
        }

        Some(batch)
    }
}
