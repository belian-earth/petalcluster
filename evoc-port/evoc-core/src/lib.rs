//! Dependency-free port of the EVoC clustering pipeline.
//!
//! Ported from the Python reference implementation by Leland McInnes at the
//! Tutte Institute (<https://github.com/TutteInstitute/evoc>, BSD-2-Clause),
//! version 0.3.1, function by function. Where behaviour is surprising the
//! Python is mirrored deliberately — parity with the reference is checked by
//! the fixtures in `../fixtures`, and divergence is a bug here unless a
//! comment says otherwise.
//!
//! Two intentional divergences, both recorded when the fixtures were
//! generated:
//! - The minimum spanning tree is produced in canonical order (each edge as
//!   `(min, max)`, sorted by weight then endpoints). The reference's parallel
//!   Borůvka emits edges in an order that is not stable under tied weights.
//! - Floating-point stages the reference runs under numba `fastmath` are
//!   compared to fixtures at small tolerances rather than bitwise.

pub mod condense;
pub mod disjoint_set;
pub mod graph;
pub mod kdtree;
pub mod layers;
pub mod linkage;
pub mod mst;
pub mod persistence;
