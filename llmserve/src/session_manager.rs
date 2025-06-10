use std::collections::HashMap;

struct Session {
    seq_ids: Vec<u64>,
}

pub struct SessionManager {
    session_table: HashMap<String, Session>,
}

impl SessionManager {
    pub fn new() -> Self {
        SessionManager {
            session_table: HashMap::new(),
        }
    }

    pub fn get_seq_ids(&self, session_id: String) -> Option<Vec<u64>> {
        let session = self.session_table.get(&session_id)?;

        Some(session.seq_ids.clone())
    }

    pub fn add_session(&mut self, session_id: String, seq_ids: Vec<u64>) {
        let session = Session { seq_ids };

        self.session_table.insert(session_id, session);
    }
}
