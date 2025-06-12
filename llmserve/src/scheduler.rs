use std::cmp::min;
use std::collections::{HashMap, VecDeque};

use tracing::info;

use crate::block_manager::BlockManager;
use crate::infer_task::{InferInput, InferOutput, InferTask};
use crate::pb::worker::BlockMapping;
use crate::sequence::{SeqStatus, Sequence};

struct BatchEntry {
    pub input_len: usize,
    pub chunked: bool,
}

pub struct Scheduler {
    pub max_batch_size: usize,
    pub max_seq_len: usize,
    pub max_num_batched_tokens: usize,
    block_manager: BlockManager,
    watermark_blocks: f32,
    waiting: VecDeque<InferTask>,
    allocated: VecDeque<InferTask>,

    running_batch: HashMap<u64, BatchEntry>,

    last_log_time: u64,
}

impl Scheduler {
    pub fn new(
        max_batch_size: usize,
        max_seq_len: usize,
        max_num_batched_tokens: usize,
        block_size: usize,
        num_blocks: usize,
    ) -> Scheduler {
        Scheduler {
            max_batch_size,
            max_seq_len,
            max_num_batched_tokens,
            block_manager: BlockManager::new(block_size, num_blocks),
            watermark_blocks: 0.97,
            waiting: VecDeque::new(),
            allocated: VecDeque::new(),
            running_batch: HashMap::new(),
            last_log_time: 0,
        }
    }

    pub fn add(&mut self, infer_task: InferTask) {
        self.waiting.push_back(infer_task);
    }

    fn can_alloc_seq(&self, seq: &Sequence, watermark: f32) -> bool {
        let num_blocks = self.block_manager.get_num_required_blocks(seq);
        self.block_manager.can_alloc_blocks(num_blocks, watermark)
    }

    fn try_reserve_infer_task(&mut self, infer_task: &mut InferTask, watermark: f32) -> bool {
        let mut seqs = infer_task.get_active_seqs_mut();
        let mut seq_iter = seqs.iter_mut();

        let Some(head_seq) = seq_iter.next() else {
            return false;
        };

        if !self.can_alloc_seq(head_seq, watermark) {
            return false;
        }

        self.block_manager.reserve_blocks(head_seq);
        head_seq.status = SeqStatus::ALLOCATED;

        for seq in seq_iter {
            self.block_manager
                .share_blocks(head_seq, seq)
                .unwrap_or_else(|e| panic!("Failed to share KV block: {e}"));
            if self.can_alloc_seq(seq, watermark) {
                self.block_manager.reserve_blocks(seq);
                seq.status = SeqStatus::ALLOCATED;
            }
        }

        true
    }

    fn try_extend_infer_task(&mut self, infer_task: &mut InferTask, watermark: f32) -> bool {
        let mut num_alloc_seqs = 0;
        let seqs = infer_task.get_active_seqs_mut();
        for seq in seqs {
            if self.can_alloc_seq(seq, watermark) {
                self.block_manager.reserve_blocks(seq);
                seq.status = SeqStatus::ALLOCATED;
                num_alloc_seqs += 1;
            }
        }

        num_alloc_seqs > 0
    }

    fn preempt_infer_task(&mut self, infer_task: &mut InferTask) {
        let seqs = infer_task.get_active_seqs_mut();
        for seq in seqs {
            self.block_manager
                .free(seq)
                .unwrap_or_else(|e| panic!("Failed to preempt sequence: {e}"));
            seq.status = SeqStatus::WAITING;
        }
    }

    fn dispatch(
        &mut self,
    ) -> (
        HashMap<u64, BatchEntry>,
        Vec<InferInput>,
        Vec<BlockMapping>,
        Vec<BlockMapping>,
    ) {
        let mut running_batch: HashMap<u64, BatchEntry> = HashMap::new();
        let mut infer_inputs: Vec<InferInput> = Vec::new();
        let mut fetch_block_mappings: Vec<BlockMapping> = Vec::new();
        let mut write_back_block_mappings: Vec<BlockMapping> = Vec::new();

        let mut num_seqs = 0;
        let mut token_budget = self.max_num_batched_tokens;
        'outer: for infer_task in self.allocated.iter() {
            for seq in infer_task.get_seqs(SeqStatus::ALLOCATED) {
                if num_seqs >= self.max_batch_size || token_budget == 0 {
                    break 'outer;
                }

                let total = seq.token_ids.len();
                let mut filled = match seq.cached {
                    true => {
                        let cached_len = seq.token_ids.len() - 1;
                        self.block_manager.update_filled_len(seq.seq_id, cached_len);
                        cached_len
                    }
                    false => self.block_manager.get_filled_token_len(seq.seq_id),
                };
                // Although all tokens are filled, we use last token to generate an output token.
                filled = min(filled, total - 1);
                let input_len = min(total - filled, token_budget);

                if input_len > 0 {
                    let input_ids = seq.token_ids[filled..filled + input_len].to_vec();
                    let (_, block_ids) =
                        self.block_manager
                            .get_block_ids_range(seq.seq_id, 0, filled + input_len);

                    let chunked = total > (filled + input_len);

                    let entry = BatchEntry {
                        input_len: input_ids.len(),
                        chunked,
                    };

                    infer_inputs.push(InferInput::new(
                        seq.seq_id,
                        input_ids,
                        filled,
                        block_ids.clone(),
                    ));

                    let block_mapping = BlockMapping {
                        seq_id: seq.seq_id,
                        block_ids,
                    };

                    if seq.cached {
                        fetch_block_mappings.push(block_mapping.clone());
                    }
                    write_back_block_mappings.push(block_mapping);

                    num_seqs += 1;
                    token_budget -= input_len;

                    running_batch.insert(seq.seq_id, entry);
                }
            }
        }

        (
            running_batch,
            infer_inputs,
            fetch_block_mappings,
            write_back_block_mappings,
        )
    }

    pub fn schedule(&mut self) -> (Vec<InferInput>, Vec<BlockMapping>, Vec<BlockMapping>) {
        let mut allocated: VecDeque<InferTask> = VecDeque::new();

        while let Some(mut infer_task) = self.allocated.pop_front() {
            if self.try_extend_infer_task(&mut infer_task, 1.0) {
                allocated.push_back(infer_task);
            } else {
                if let Some(mut preempt_task) = self.allocated.pop_back() {
                    self.preempt_infer_task(&mut preempt_task);
                    self.waiting.push_front(preempt_task);

                    self.allocated.push_front(infer_task);
                    continue;
                } else {
                    self.waiting.push_front(infer_task);
                    break;
                }
            }
        }

        while let Some(mut infer_task) = self.waiting.pop_front() {
            if self.try_reserve_infer_task(&mut infer_task, self.watermark_blocks) {
                allocated.push_back(infer_task);
            } else {
                self.waiting.push_front(infer_task);
                break;
            }
        }

        self.allocated = allocated;

        let (running_batch, infer_inputs, fetch_block_mappings, write_back_block_mappings) =
            self.dispatch();

        self.running_batch = running_batch;

        {
            // Logging
            let now = utils::time::now_ns();

            if now > self.last_log_time + 5e9 as u64 {
                info!(
                    "Running: {} seqs, {:}",
                    infer_inputs.len(),
                    self.get_log_text()
                );
                self.last_log_time = now;
            }
        }

        (
            infer_inputs,
            fetch_block_mappings,
            write_back_block_mappings,
        )
    }

    pub fn commit(&mut self, infer_outputs: HashMap<u64, InferOutput>) -> Vec<InferTask> {
        let mut finished_tasks: Vec<InferTask> = Vec::new();
        let mut still_allocated_tasks = VecDeque::with_capacity(self.allocated.len());
        let now = utils::time::now_ns();

        for mut task in self.allocated.drain(..) {
            for seq in task.get_seqs_mut(SeqStatus::ALLOCATED) {
                let (output, entry) = match (
                    infer_outputs.get(&seq.seq_id),
                    self.running_batch.get(&seq.seq_id),
                ) {
                    (Some(output), Some(entry)) => (output, entry),
                    _ => continue,
                };

                self.block_manager
                    .update_filled_len(seq.seq_id, entry.input_len as usize);

                if entry.chunked {
                    continue;
                }

                seq.cached = false;
                seq.append_output_id(
                    output.output_id,
                    output.prob,
                    output.output_word.clone(),
                    now,
                );

                let reached_max_len = seq
                    .max_output_len
                    .map_or(false, |max| seq.output_len >= max);

                if (!seq.ignore_eos && output.is_eos) || reached_max_len {
                    seq.status = SeqStatus::FINISHED;
                }
            }

            if task.is_finished() {
                finished_tasks.push(task);
            } else {
                still_allocated_tasks.push_back(task);
            }
        }

        for task in finished_tasks.iter() {
            self.remove_task(task);
        }

        self.allocated = still_allocated_tasks;
        self.running_batch.clear();

        finished_tasks
    }

    fn remove_task(&mut self, infer_task: &InferTask) {
        for seq in infer_task.get_seqs(SeqStatus::FINISHED) {
            self.block_manager.free(seq).unwrap_or_else(|e| {
                panic!(
                    "Failed to free KV blocks for sequence {}: {}",
                    seq.seq_id, e
                )
            });
        }
    }

    fn get_log_text(&self) -> String {
        let num_allocated_reqs: usize = self.allocated.len();
        let num_waiting_reqs: usize = self.waiting.len();
        let kv_block_usage: f32 = self.block_manager.get_block_usage();

        format!(
            "Allocated: {} reqs, Waiting: {} reqs, KV cache usage: {:.2} %",
            num_allocated_reqs,
            num_waiting_reqs,
            kv_block_usage * 100.0
        )
    }
}
