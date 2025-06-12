use std::collections::{HashSet, VecDeque};
use std::hash::Hash;

pub struct FifoSet<T> {
    items: HashSet<T>,
    order: VecDeque<T>,
}

impl<T> FifoSet<T>
where
    T: Eq + Hash + Copy,
{
    pub fn new() -> Self {
        Self {
            items: HashSet::new(),
            order: VecDeque::new(),
        }
    }

    pub fn insert(&mut self, item: T) -> bool {
        if self.items.insert(item) {
            self.order.push_back(item);
            return true;
        } else {
            return false;
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        while let Some(item) = self.order.pop_front() {
            if self.items.remove(&item) {
                return Some(item);
            }
        }
        None
    }

    pub fn remove(&mut self, item: &T) -> bool {
        self.items.remove(item)
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }
}

impl<T> FromIterator<T> for FifoSet<T>
where
    T: Eq + Hash + Copy,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut s = HashSet::new();
        let mut q = VecDeque::new();
        for item in iter {
            if s.insert(item) {
                q.push_back(item);
            }
        }
        Self { items: s, order: q }
    }
}
