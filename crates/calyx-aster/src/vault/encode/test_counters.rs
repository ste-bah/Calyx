thread_local! {
    static SLOT_ENCODE_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static SLOT_HASH_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(super) fn increment_slot_encode_count() {
    SLOT_ENCODE_COUNT.with(|count| count.set(count.get() + 1));
}

pub(super) fn increment_slot_hash_count() {
    SLOT_HASH_COUNT.with(|count| count.set(count.get() + 1));
}

pub(crate) fn reset_slot_operation_counts() {
    SLOT_ENCODE_COUNT.with(|count| count.set(0));
    SLOT_HASH_COUNT.with(|count| count.set(0));
}

pub(crate) fn slot_operation_counts() -> (usize, usize) {
    (
        SLOT_ENCODE_COUNT.with(std::cell::Cell::get),
        SLOT_HASH_COUNT.with(std::cell::Cell::get),
    )
}
