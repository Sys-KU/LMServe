#[derive(PartialEq, Eq)]
pub enum SeqStatus {
    WAITING,
    ALLOCATED,
    FINISHED,
    CANCELLED,
}

#[derive(PartialEq, Eq)]
pub enum StopReason {
    EOS,
    MAXLENGTH,
}

#[allow(dead_code)]
pub struct Sequence {
    pub seq_id: u64,
    pub session_id: String,

    pub token_ids: Vec<u32>,

    pub max_output_len: Option<usize>,

    pub status: SeqStatus,
    pub stop_reason: Option<StopReason>,

    pub prompt_len: usize,
    pub output_len: usize,
    output_words: Vec<String>,
    output_token_probs: Vec<f32>,
    filled_token_ids: Vec<u32>,

    append_token_times: Vec<f32>,

    pub ignore_eos: bool,

    pub cached: bool,
}

impl Sequence {
    pub fn new(
        session_id: String,
        token_ids: Vec<u32>,
        max_output_len: Option<usize>,
        ignore_eos: bool,
    ) -> Sequence {
        let seq_id = utils::random::generate_seq_id();
        let prompt_len = token_ids.len();

        Sequence {
            seq_id,
            session_id,
            token_ids,
            max_output_len,
            status: SeqStatus::WAITING,
            stop_reason: None,
            prompt_len,
            output_len: 0,
            output_words: Vec::new(),
            output_token_probs: Vec::new(),
            filled_token_ids: Vec::new(),
            append_token_times: Vec::new(),
            ignore_eos,
            cached: false,
        }
    }

    pub fn is_active(&self) -> bool {
        matches!(self.status, SeqStatus::WAITING | SeqStatus::ALLOCATED)
    }

    pub fn is_finished(&self) -> bool {
        matches!(self.status, SeqStatus::FINISHED | SeqStatus::CANCELLED)
    }

    pub fn get_token_ids(&self) -> &[u32] {
        self.token_ids.as_ref()
    }

    pub fn get_output_ids(&self) -> &[u32] {
        self.token_ids[self.prompt_len..].as_ref()
    }

    pub fn get_token_probs(&self) -> Vec<f32> {
        self.output_token_probs.clone()
    }

    pub fn get_output_words(&self) -> Vec<String> {
        self.output_words.clone()
    }

    pub fn append_output_id(&mut self, output_id: u32, prob: f32, output_word: String) {
        self.token_ids.push(output_id);
        self.output_len += 1;
        self.output_token_probs.push(prob);
        self.output_words.push(output_word);
    }
}
