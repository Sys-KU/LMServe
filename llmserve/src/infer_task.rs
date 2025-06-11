pub use crate::pb::worker::{InferInput, InferOutput};
use crate::sequence::{SeqStatus, Sequence};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum InferTaskStatus {
    WAITING,
    DECODE,
    PREFILL,
}

#[allow(dead_code)]
pub struct InferTask {
    session_id: String,
    seqs: Vec<Sequence>,
    status: InferTaskStatus,
    arrival_time: u64,
    num_samples: u16,
}

impl InferTask {
    pub fn new(session_id: String, seqs: Vec<Sequence>, arrival_time: u64) -> InferTask {
        let num_samples = seqs.len() as u16;
        InferTask {
            session_id,
            seqs,
            status: InferTaskStatus::WAITING,
            arrival_time,
            num_samples,
        }
    }

    pub fn get_active_seqs(&self) -> Vec<&Sequence> {
        self.seqs.iter().filter(|seq| seq.is_active()).collect()
    }

    pub fn get_active_seqs_mut(&mut self) -> Vec<&mut Sequence> {
        self.seqs.iter_mut().filter(|seq| seq.is_active()).collect()
    }

    pub fn get_seqs(&self, status: SeqStatus) -> Vec<&Sequence> {
        self.seqs
            .iter()
            .filter(|seq| seq.status == status)
            .collect()
    }

    pub fn get_seqs_mut(&mut self, status: SeqStatus) -> Vec<&mut Sequence> {
        self.seqs
            .iter_mut()
            .filter(|seq| seq.status == status)
            .collect()
    }

    pub fn get_status(&self) -> InferTaskStatus {
        self.status
    }

    pub fn set_prefill(&mut self) {
        self.status = InferTaskStatus::PREFILL;
    }

    pub fn set_decode(&mut self) {
        self.status = InferTaskStatus::DECODE;
    }

    pub fn is_finished(&mut self) -> bool {
        self.seqs.iter().all(|seq| seq.is_finished())
    }

    pub fn get_session_id(&self) -> String {
        self.session_id.clone()
    }

    pub fn get_arrival_time(&self) -> u64 {
        self.arrival_time
    }
}

impl InferInput {
    pub fn new(
        seq_id: u64,
        input_ids: Vec<u32>,
        filled_token_len: usize,
        block_ids: Vec<u32>,
    ) -> InferInput {
        let input_len = input_ids.len() as u64;
        let context_len = input_ids.len() + filled_token_len;

        InferInput {
            seq_id,
            input_ids,
            input_len,
            filled_token_len: filled_token_len as u64,
            context_len: context_len as u64,
            block_ids,
        }
    }
}

impl InferOutput {
    pub fn new(output_id: u32, prob: f32, output_word: String, is_eos: bool) -> InferOutput {
        InferOutput {
            output_id,
            prob,
            output_word,
            is_eos,
        }
    }
}
