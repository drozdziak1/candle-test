use anyhow::{Result, anyhow, bail};
use hf_hub::api::sync::ApiRepo;
use log::{debug, warn};
use polars::prelude::*;
use tokenizers::Tokenizer;

use std::path::PathBuf;

/// Visits subsequent parquet files of the fineweb dataset; tracks state for
/// returning subsequent records
pub struct MultiParquetIter {
    repo: ApiRepo,
    dir: PathBuf,
    rng_seed: u64,
    next_df_i: usize,
    next_df_j: usize,
    cur_parquet_column: Option<Column>,
    cur_column_idx: usize,
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

                self.cur_parquet_column = Some(df.column("text")?.shuffle(Some(self.rng_seed)));

                self.next_df_j += 1;

                self.cur_column_idx = 0;
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
            cur_parquet_column: None,
            cur_column_idx: 0,
        }
    }
}

impl Iterator for MultiParquetIter {
    type Item = String;
    fn next(&mut self) -> Option<Self::Item> {
        if self.cur_parquet_column.is_none() {
            self.next_parquet_df().ok()?;
        }

        let mut cur_parquet_column = self
            .cur_parquet_column
            .as_mut()
            .expect("INTERNAL: We've literally just filled cur_parquet_df");

        // Try to move on to the next parquet file if we've reached the end of this one
        if self.cur_column_idx >= cur_parquet_column.len() {
            self.next_parquet_df().ok()?;
            cur_parquet_column = self.cur_parquet_column.as_mut()?;
        }

        match cur_parquet_column.get(self.cur_column_idx) {
            Ok(value) => {
                self.cur_column_idx += 1;
                value.get_str().map(|s| s.to_string())
            }
            Err(e) => {
                warn!(
                    "Could not get element {} of cur_parquet_column: {}",
                    self.cur_column_idx,
                    e.to_string()
                );
                None
            }
        }
    }
}

/// Builds batches of desired size from underlying parquet file
/// iterator, making sure that every sample comes from a unique record
/// of the dataset. Note: The iterator items are a flat single-rank n*m vector
pub struct BatchIter<'a> {
    batch_size: usize,
    mbatch_size: usize,
    sample_size: usize,
    separator: Vec<u32>,
    multi_parquet_iter: MultiParquetIter,
    tokenizer: &'a Tokenizer,
    buffers: Vec<Vec<Vec<u32>>>,
}

impl<'a> BatchIter<'a> {
    pub fn new(
        batch_size: usize,
        mbatch_size: usize,
        sample_size: usize,
        multi_parquet_iter: MultiParquetIter,
        tokenizer: &'a Tokenizer,
    ) -> Result<Self> {
        let separator = tokenizer
            .encode("<|endoftext|>", true)
            .map_err(|e| anyhow!(e.to_string()))?
            .get_ids()
            .to_owned();

        Ok(Self {
            batch_size,
            mbatch_size,
            sample_size,
            separator,
            multi_parquet_iter,
            tokenizer,
            buffers: vec![vec![vec![]; mbatch_size]; batch_size],
        })
    }
}

impl<'a> Iterator for BatchIter<'a> {
    // Tensor::from_vec() expects flat vecs
    type Item = Vec<Vec<u32>>;
    fn next(&mut self) -> Option<Self::Item> {
        let mut flat_mbatches = vec![Vec::with_capacity(self.mbatch_size * self.sample_size); self.batch_size];

        for (mbatch_bufs, flat_mbatch) in self.buffers.iter_mut().zip(flat_mbatches.iter_mut()) {
            for buf in mbatch_bufs.iter_mut() {
                // Ensure there is enough tokens in current buffer
                while buf.len() < self.sample_size {
                    let new_chunk_txt = self.multi_parquet_iter.next()?;

                    let new_chunk_encoded = self
                        .tokenizer
                        .encode(new_chunk_txt, true)
                        .ok()?
                        .get_ids()
                        .to_owned();

                    buf.extend_from_slice(self.separator.as_slice());
                    buf.extend_from_slice(new_chunk_encoded.as_slice());
                }

                // Use first self.sample_size buffer tokens to fill the current minibatch item
                let (new_batch_item, new_buf) = buf.split_at(self.sample_size);

                flat_mbatch.extend_from_slice(new_batch_item);

                *buf = new_buf.to_vec();
            }
        }

        Some(flat_mbatches)
    }
}
