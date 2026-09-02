//! The package's rayon thread pool.
//!
//! Every parallel binding runs inside `pool().install(..)`, so rayon calls
//! made by dependencies (petal-clustering, evoc-core) on that path land in
//! this pool rather than rayon's implicit global one. Rebuilding the pool is
//! how `shoal_threads(n)` takes effect at runtime; the global pool cannot be
//! resized once created, which is why the package owns one.

use rayon::{ThreadPool, ThreadPoolBuilder};
use std::sync::{Arc, OnceLock, RwLock};

static POOL: OnceLock<RwLock<Arc<ThreadPool>>> = OnceLock::new();

fn build(n: usize) -> Arc<ThreadPool> {
    // 0 lets rayon pick: RAYON_NUM_THREADS if set, else available parallelism.
    Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("shoal-{i}"))
            .build()
            .expect("failed to build the rayon thread pool"),
    )
}

fn slot() -> &'static RwLock<Arc<ThreadPool>> {
    POOL.get_or_init(|| RwLock::new(build(0)))
}

/// The current pool. Cloning the `Arc` means a concurrent `set_threads`
/// cannot drop it from under a running computation.
pub fn pool() -> Arc<ThreadPool> {
    slot().read().expect("thread pool lock poisoned").clone()
}

pub fn set_threads(n: usize) {
    let fresh = build(n);
    *slot().write().expect("thread pool lock poisoned") = fresh;
}

pub fn threads() -> usize {
    pool().current_num_threads()
}
