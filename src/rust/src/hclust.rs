use kodama::{linkage, Method, Step};

/// Resolve a linkage method by name.
///
/// Names are validated on the R side; an unknown name here is a bug.
///
/// Note that kodama squares the dissimilarities for Ward, centroid and median
/// and takes the square root afterwards, matching fastcluster and SciPy. Ward
/// therefore corresponds to R's `"ward.D2"`, and centroid and median take plain
/// distances rather than the squared distances `stats::hclust()` expects.
pub fn method_from_name(name: &str) -> Method {
    match name {
        "single" => Method::Single,
        "complete" => Method::Complete,
        "average" => Method::Average,
        "weighted" => Method::Weighted,
        "ward" => Method::Ward,
        "centroid" => Method::Centroid,
        "median" => Method::Median,
        _ => panic!("Unknown linkage method: {name}"),
    }
}

/// Translate a kodama cluster label into R's `hclust` merge encoding.
///
/// kodama labels observations `0..n` and the cluster formed at step `i` as
/// `n + i` (SciPy's scheme). R encodes a singleton observation as its negated
/// 1-based index, and a previously formed cluster as the positive 1-based
/// index of the step that created it.
#[inline]
fn to_merge_entry(label: usize, n: usize) -> i32 {
    if label < n {
        -((label + 1) as i32)
    } else {
        (label - n + 1) as i32
    }
}

/// Compute the leaf ordering R uses to draw a dendrogram without crossings.
///
/// Depth-first from the root merge, emitting observations as they are reached.
/// Iterative rather than recursive so deep trees cannot overflow the stack.
fn leaf_order(steps: &[Step<f64>], n: usize) -> Vec<i32> {
    let mut order = Vec::with_capacity(n);

    if steps.is_empty() {
        if n == 1 {
            order.push(1);
        }
        return order;
    }

    // The root is the final step, in R's 1-based step numbering.
    let mut stack: Vec<i32> = vec![steps.len() as i32];

    while let Some(node) = stack.pop() {
        if node < 0 {
            order.push(-node);
        } else {
            let step = &steps[(node - 1) as usize];
            // Push the right child first so the left is popped, and emitted, first.
            stack.push(to_merge_entry(step.cluster2, n));
            stack.push(to_merge_entry(step.cluster1, n));
        }
    }

    order
}

/// The pieces of an `hclust` object.
pub struct Hclust {
    /// `(n - 1) x 2`, column-major, in R's merge encoding.
    pub merge: Vec<i32>,
    pub height: Vec<f64>,
    pub order: Vec<i32>,
}

/// Run hierarchical clustering over a condensed dissimilarity matrix.
///
/// `condensed` is consumed: kodama's `linkage` takes it by mutable reference and
/// destroys it while working, so callers must pass a copy they do not need.
pub fn hclust(mut condensed: Vec<f64>, n: usize, method: Method) -> Hclust {
    let dend = linkage(&mut condensed, n, method);
    let steps = dend.steps();

    let n_steps = steps.len();
    let mut merge = vec![0i32; 2 * n_steps];
    let mut height = Vec::with_capacity(n_steps);

    for (k, step) in steps.iter().enumerate() {
        // R matrices are column-major: (k, 0) at k, (k, 1) at n_steps + k.
        merge[k] = to_merge_entry(step.cluster1, n);
        merge[n_steps + k] = to_merge_entry(step.cluster2, n);
        height.push(step.dissimilarity);
    }

    Hclust {
        merge,
        height,
        order: leaf_order(steps, n),
    }
}
