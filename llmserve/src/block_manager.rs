use std::cmp::min;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fmt;
use std::hash::{Hash, Hasher};

use ahash::AHasher;

use crate::sequence::Sequence;
use utils::collections::FifoSet;

#[derive(PartialEq, Eq, Debug)]
enum BlockStatus {
    FREE,
    USED,
}

#[derive(Debug)]
struct Block {
    id: u32,
    block_size: usize,
    ref_cnt: i32,
    status: BlockStatus,
    token_ids: Vec<u32>,
    hash: u64,
}

fn compute_hash(token_ids: &[u32]) -> u64 {
    let mut hasher = AHasher::default();
    token_ids.hash(&mut hasher);
    hasher.finish()
}

impl Block {
    fn new(id: u32, block_size: usize) -> Self {
        Self {
            id,
            block_size,
            ref_cnt: 0,
            status: BlockStatus::FREE,
            token_ids: Vec::new(),
            hash: Default::default(),
        }
    }

    fn append_tokens(&mut self, new_token_ids: &mut Vec<u32>) {
        assert!(self.token_ids.len() + new_token_ids.len() <= self.block_size as usize);

        self.token_ids.append(new_token_ids);
    }

    fn get_num_empty_slots(&self) -> usize {
        self.block_size - self.token_ids.len()
    }

    fn get_hash(&self) -> u64 {
        self.hash
    }

    fn set_hash(&mut self, hash: u64) {
        self.hash = hash;
    }
}

#[derive(Debug)]
pub enum BlockAllocError {
    NotEnoughBlocks,
    EmptyTokenIds { seq_id: u64 },
    NotFoundBlockTable { seq_id: u64 },
}

impl fmt::Display for BlockAllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BlockAllocError::NotEnoughBlocks => write!(f, "Not enough blocks available"),
            BlockAllocError::EmptyTokenIds { seq_id } => {
                write!(f, "token_ids is empty for seq_id {}", seq_id)
            }
            BlockAllocError::NotFoundBlockTable { seq_id } => {
                write!(f, "Not found block table for seq_id {}", seq_id)
            }
        }
    }
}

impl std::error::Error for BlockAllocError {}

#[allow(dead_code)]
struct BlockAllocator {
    block_size: usize,
    num_total_blocks: usize,
    block_pool: Vec<Block>,
    free_block_ids: FifoSet<u32>,
}

// Make BlockAllocator methods return `Option<T>` instead of panicking.
impl BlockAllocator {
    pub fn new(block_size: usize, num_blocks: usize) -> Self {
        let block_pool: Vec<Block> = (0..num_blocks as u32)
            .map(|id| Block::new(id, block_size))
            .collect();
        let free_block_ids: FifoSet<u32> = block_pool.iter().map(|b| b.id).collect();

        Self {
            block_size,
            num_total_blocks: num_blocks,
            block_pool,
            free_block_ids,
        }
    }

    fn get_block(&self, block_id: u32) -> &Block {
        self.block_pool
            .get(block_id as usize)
            .expect(&format!("Block ID {} is out of bounds", block_id))
    }

    fn get_block_mut(&mut self, block_id: u32) -> &mut Block {
        self.block_pool
            .get_mut(block_id as usize)
            .expect(&format!("Block ID {} is out of bounds", block_id))
    }

    fn can_alloc_blocks(&self, num_required_blocks: usize, watermark: f32) -> bool {
        assert!(
            0.0 < watermark && watermark <= 1.0,
            "watermark must be in (0, 1]"
        );
        let num_free_blocks = self.free_block_ids.len();
        let num_used_blocks = self.num_total_blocks - num_free_blocks;

        num_used_blocks + num_required_blocks <= (watermark * self.num_total_blocks as f32) as usize
    }

    fn alloc_block(&mut self) -> &mut Block {
        let block_id = self
            .free_block_ids
            .pop()
            .expect("Cannot allocate block: all blocks are already in use.");

        let block = self
            .block_pool
            .get_mut(block_id as usize)
            .expect(&format!("Block ID {} is out of bounds", block_id));

        if block.status == BlockStatus::USED {
            panic!("Cannot allocate block {block_id}: alread in used.");
        }

        block.token_ids.clear();
        block.hash = Default::default();

        block.status = BlockStatus::USED;
        block.ref_cnt = 1;
        block
    }

    fn free_block(&mut self, block_id: u32) {
        let block = self.get_block_mut(block_id);
        assert!(
            block.status != BlockStatus::FREE || block.ref_cnt <= 0,
            "Double free detected for block {}.",
            block_id
        );
        block.ref_cnt -= 1;
        if block.ref_cnt == 0 {
            block.status = BlockStatus::FREE;
            self.free_block_ids.insert(block_id);
        }
    }

    fn alloc_block_by_id(&mut self, block_id: u32) -> &mut Block {
        let block = self
            .block_pool
            .get_mut(block_id as usize)
            .expect(&format!("Block ID {} is out of bounds", block_id));

        if block.status == BlockStatus::FREE {
            assert!(
                self.free_block_ids.remove(&block_id),
                "BrokenAllocatorError: block {block_id} is free but not in free list"
            );
            block.status = BlockStatus::USED;
        }
        block.ref_cnt += 1;
        block
    }

    fn get_block_usage(&self) -> f32 {
        let num_free_blocks = self.free_block_ids.len();
        let num_use_blocks = self.num_total_blocks - num_free_blocks;

        num_use_blocks as f32 / self.num_total_blocks as f32
    }
}

struct BlockMap {
    block_ids: Vec<u32>,
    num_allocated_slots: usize,
    filled_token_len: usize,
}

impl BlockMap {
    fn new(block_ids: Vec<u32>, num_allocated_slots: usize, filled_token_len: usize) -> Self {
        Self {
            block_ids,
            num_allocated_slots,
            filled_token_len,
        }
    }

    fn append_block_ids(&mut self, block_ids: &mut Vec<u32>, num_allocated_slots: usize) {
        self.block_ids.append(block_ids);
        self.num_allocated_slots += num_allocated_slots;
    }
}

pub struct BlockManager {
    pub block_size: usize,
    seq_block_mapping_table: HashMap<u64, BlockMap>,
    hash_block_table: HashMap<u64, u32>, // HashMap<hash value, block id>
    block_allocator: BlockAllocator,
}

impl BlockManager {
    pub fn new(block_size: usize, num_blocks: usize) -> Self {
        let block_allocator = BlockAllocator::new(block_size, num_blocks);

        Self {
            block_size,
            seq_block_mapping_table: HashMap::new(),
            hash_block_table: HashMap::new(),
            block_allocator,
        }
    }

    pub fn get_block_ids(&self, seq: &Sequence) -> Vec<u32> {
        self.seq_block_mapping_table
            .get(&seq.seq_id)
            .map_or(Vec::new(), |block_map| block_map.block_ids.clone())
    }

    pub fn get_block_ids_range(&self, seq_id: u64, start: usize, end: usize) -> (usize, Vec<u32>) {
        let block_offset = start / self.block_size;
        let mut block_end = end / self.block_size + 1;

        if let Some(block_map) = self.seq_block_mapping_table.get(&seq_id) {
            let block_ids = &block_map.block_ids;
            if block_offset >= block_end || block_offset >= block_ids.len() {
                return (0, Vec::new());
            }

            block_end = block_end.min(block_ids.len());

            (block_offset, (&block_ids[block_offset..block_end]).to_vec())
        } else {
            (0, Vec::new())
        }
    }

    pub fn get_filled_token_len(&self, seq_id: u64) -> usize {
        self.seq_block_mapping_table
            .get(&seq_id)
            .map_or(0, |block_map| block_map.filled_token_len)
    }

    pub fn get_num_required_blocks(&self, seq: &Sequence) -> usize {
        let filled_token_len = self.get_filled_token_len(seq.seq_id);
        let total_token_len = seq.token_ids.len();
        let num_required_slots = total_token_len.saturating_sub(filled_token_len);
        num_required_slots.div_ceil(self.block_size)
    }

    fn get_last_block_id(&self, seq_id: u64) -> Option<u32> {
        let block_map = self.seq_block_mapping_table.get(&seq_id)?;
        let last_block_id = block_map.block_ids.last()?;
        Some(last_block_id.clone())
    }

    fn get_num_allocated_slots(&self, seq_id: u64) -> usize {
        self.seq_block_mapping_table
            .get(&seq_id)
            .map_or(0, |block_map| block_map.num_allocated_slots)
    }

    pub fn can_alloc_blocks(&self, num_blocks: usize, watermark: f32) -> bool {
        self.block_allocator.can_alloc_blocks(num_blocks, watermark)
    }

    fn lookup_block_id(&mut self, hash: u64) -> Option<u32> {
        match self.hash_block_table.entry(hash) {
            Entry::Occupied(entry) => {
                let block_id = *entry.get();
                let block = self.block_allocator.get_block(block_id);
                if block.get_hash() == hash {
                    return Some(block_id);
                } else {
                    entry.remove_entry();
                }
            }
            Entry::Vacant(_) => {}
        }

        None
    }

    fn get_prefix_reusable_blocks(&mut self, seq: &Sequence) -> Vec<u32> {
        let token_ids = &seq.token_ids;
        let total_token_len = token_ids.len();

        let mut block_ids: Vec<u32> = Vec::new();
        for token_offset in (0..total_token_len).step_by(self.block_size) {
            let token_end = min(token_offset + self.block_size, total_token_len);
            let sub_token_ids = &token_ids[token_offset..token_end];

            if sub_token_ids.len() < self.block_size {
                break;
            }

            let hash = compute_hash(&token_ids[..token_end]);

            if let Some(block_id) = self.lookup_block_id(hash) {
                block_ids.push(block_id);
            } else {
                break;
            }
        }

        block_ids
    }

    fn alloc_blocks(&mut self, seq: &Sequence) -> usize {
        let mut num_allocated_slots = self.get_num_allocated_slots(seq.seq_id);

        let token_ids = &seq.token_ids;
        let total_token_len = token_ids.len();
        let num_alloc_tokens = total_token_len - num_allocated_slots;

        let mut reused_token_len = 0;
        let mut add_block_ids: Vec<u32> = Vec::new();

        if num_allocated_slots == 0 {
            for block_id in self.get_prefix_reusable_blocks(seq) {
                self.block_allocator.alloc_block_by_id(block_id);

                num_allocated_slots += self.block_size;
                reused_token_len += self.block_size;

                add_block_ids.push(block_id);
            }
        } else if num_allocated_slots > 0 {
            let last_block_id = self
                .get_last_block_id(seq.seq_id)
                .expect("Not found a last block");
            let last_block = self.block_allocator.get_block_mut(last_block_id);

            let num_append_slots = min(last_block.get_num_empty_slots(), num_alloc_tokens);
            if num_append_slots > 0 {
                let token_end = num_allocated_slots + num_append_slots;
                let sub_token_ids = &mut (token_ids[num_allocated_slots..token_end]).to_vec();
                last_block.append_tokens(sub_token_ids);

                if last_block.get_num_empty_slots() == 0 {
                    let hash = compute_hash(&token_ids[..token_end]);
                    last_block.set_hash(hash);
                    self.hash_block_table.insert(hash, last_block.id);
                }

                num_allocated_slots += num_append_slots;
            }
        }

        if total_token_len > num_allocated_slots {
            for token_offset in (num_allocated_slots..total_token_len).step_by(self.block_size) {
                let token_end = min(token_offset + self.block_size, total_token_len);
                let sub_token_ids = &token_ids[token_offset..token_end];

                let block = self.block_allocator.alloc_block();

                // We are going to refresh (update) the newly allocated block,
                // so we need to remove the entry of the new block that remains in the hash table
                self.hash_block_table.remove(&block.get_hash());

                block.append_tokens(&mut sub_token_ids.to_vec());

                if block.get_num_empty_slots() == 0 {
                    let hash = compute_hash(&token_ids[..token_end]);
                    block.set_hash(hash);
                    self.hash_block_table.insert(hash, block.id);
                }

                add_block_ids.push(block.id);
            }
        }

        self.seq_block_mapping_table
            .entry(seq.seq_id)
            .and_modify(|block_map| {
                block_map.append_block_ids(&mut add_block_ids, num_alloc_tokens)
            })
            .or_insert(BlockMap::new(
                add_block_ids,
                num_alloc_tokens,
                reused_token_len,
            ));

        num_alloc_tokens
    }

    fn copy_block(&mut self, src_block_id: u32) -> &mut Block {
        let token_ids = {
            let src_block = self.block_allocator.get_block(src_block_id);
            src_block.token_ids.clone()
        };
        let new_block = self.block_allocator.alloc_block();

        self.hash_block_table.remove(&new_block.get_hash());

        new_block.token_ids = token_ids.clone();
        new_block
    }

    pub fn reserve_blocks(&mut self, seq: &Sequence) -> usize {
        self.alloc_blocks(seq)
    }

    pub fn extend_blocks(
        &mut self,
        seq: &Sequence,
        blocks_to_copy: &mut HashMap<u32, Vec<u32>>,
    ) -> usize {
        // TODO(jinu): Remove copy blocks.
        let copy_block_id = self
            .get_last_block_id(seq.seq_id)
            .filter(|block_id| {
                let block = self.block_allocator.get_block(*block_id);
                block.ref_cnt > 1 && block.get_num_empty_slots() > 0
            })
            .map(|block_id| block_id);

        if let Some(copy_block_id) = copy_block_id {
            let new_block_id = self.copy_block(copy_block_id).id;
            self.block_allocator.free_block(copy_block_id);

            self.seq_block_mapping_table
                .entry(seq.seq_id)
                .and_modify(|block_map| {
                    if let Some(last_block_id) = block_map.block_ids.last_mut() {
                        *last_block_id = new_block_id;
                    }
                });

            blocks_to_copy
                .entry(copy_block_id)
                .and_modify(|block_ids| block_ids.push(new_block_id))
                .or_insert(vec![new_block_id]);
        }

        self.alloc_blocks(seq)
    }

    pub fn share_blocks(
        &mut self,
        src_seq: &Sequence,
        dst_seq: &Sequence,
    ) -> Result<Vec<u32>, BlockAllocError> {
        let mut share_block_ids: Vec<u32> = Vec::new();
        let src_block_map = self.seq_block_mapping_table.get(&src_seq.seq_id).ok_or(
            BlockAllocError::NotFoundBlockTable {
                seq_id: src_seq.seq_id,
            },
        )?;

        let dst_token_ids = &dst_seq.token_ids;
        let dst_seq_len = dst_token_ids.len();
        let max_share_token_len = min(
            dst_seq_len - 1,
            self.get_num_allocated_slots(src_seq.seq_id),
        );

        let mut shared_token_len = 0;
        let mut iter = dst_token_ids.chunks_exact(self.block_size);
        for block_id in src_block_map.block_ids.iter() {
            let dst_sub_token_ids = match iter.next() {
                Some(token_ids) => token_ids,
                None => break,
            };

            let dst_sub_token_len = dst_sub_token_ids.len();
            if shared_token_len + dst_sub_token_len > max_share_token_len {
                break;
            }

            let block = self.block_allocator.get_block_mut(*block_id);
            if block.token_ids != dst_sub_token_ids {
                break;
            }

            block.ref_cnt += 1;
            share_block_ids.push(*block_id);
            shared_token_len += dst_sub_token_len;
        }

        if share_block_ids.len() > 0 {
            self.seq_block_mapping_table.insert(
                dst_seq.seq_id,
                BlockMap::new(share_block_ids, shared_token_len, shared_token_len),
            );
        }

        Ok(dst_token_ids[0..shared_token_len].to_vec())
    }

    pub fn update_filled_len(&mut self, seq_id: u64, new_filled_token_len: usize) {
        self.seq_block_mapping_table
            .entry(seq_id)
            .and_modify(|block_map| {
                block_map.filled_token_len = min(
                    block_map.filled_token_len + new_filled_token_len,
                    block_map.num_allocated_slots,
                )
            });
    }

    pub fn free(&mut self, seq: &Sequence) -> Result<(), BlockAllocError> {
        let block_map = self
            .seq_block_mapping_table
            .remove(&seq.seq_id)
            .ok_or(BlockAllocError::NotFoundBlockTable { seq_id: seq.seq_id })?;

        for block_id in block_map.block_ids.iter() {
            self.block_allocator.free_block(*block_id);
        }

        Ok(())
    }

    pub fn get_block_usage(&self) -> f32 {
        self.block_allocator.get_block_usage()
    }
}
