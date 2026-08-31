//! Persistence barcode over minimum cluster size, peak finding and diverse
//! peak selection — ports of the corresponding functions in
//! `clustering_utilities.py`.

use crate::condense::CondensedTree;

/// Port of scipy-derived `find_peaks`: indices of strict local maxima, with
//  plateaus resolved to their midpoint.
pub fn find_peaks(x: &[f32]) -> Vec<i64> {
    let mut midpoints = Vec::new();
    if x.len() < 3 {
        return midpoints;
    }
    let i_max = x.len() - 1;
    let mut i = 1usize;
    while i < i_max {
        if x[i - 1] < x[i] {
            let mut i_ahead = i + 1;
            while i_ahead < i_max && x[i_ahead] == x[i] {
                i_ahead += 1;
            }
            if x[i_ahead] < x[i] {
                let left = i;
                let right = i_ahead - 1;
                midpoints.push(((left + right) / 2) as i64);
                i = i_ahead;
            }
        }
        i += 1;
    }
    midpoints
}

pub struct Barcode {
    pub births: Vec<f32>,
    pub deaths: Vec<f32>,
    pub parents: Vec<i32>,
    pub lambda_deaths: Vec<f32>,
}

/// Port of `min_cluster_size_barcode`. Relies on the pair structure of the
/// cluster tree: `condense_tree` appends split children two at a time, so
/// consecutive row pairs share a parent — hence the reverse step of 2.
pub fn min_cluster_size_barcode(cluster_tree: &CondensedTree, n_points: i64, min_size: i64) -> Barcode {
    let n_nodes = (cluster_tree.child[cluster_tree.len() - 1] - n_points + 1) as usize;

    let mut parents = vec![0i32; n_nodes];
    let mut lambda_deaths = vec![0.0f32; n_nodes];
    let mut size_deaths = vec![0.0f32; n_nodes];
    let mut size_births = vec![min_size as f32; n_nodes];
    lambda_deaths[0] = 0.0;
    size_deaths[0] = n_points as f32;
    parents[0] = n_points as i32;

    let n_rows = cluster_tree.len();
    let mut idx = n_rows as i64 - 1;
    while idx > 0 {
        let i = idx as usize;
        let out_idx = (cluster_tree.child[i] - n_points) as usize;
        let parent = cluster_tree.parent[i] as i32;
        let lambda_death = (-1.0f32 / cluster_tree.lambda_val[i]).exp();
        parents[out_idx - 1] = parent;
        parents[out_idx] = parent;
        lambda_deaths[out_idx - 1] = lambda_death;
        lambda_deaths[out_idx] = lambda_death;

        let death_size = cluster_tree.child_size[i - 1].min(cluster_tree.child_size[i]) as f32;
        size_deaths[out_idx - 1] = death_size;
        size_deaths[out_idx] = death_size;
        let parent_idx = (cluster_tree.parent[i] - n_points) as usize;
        size_births[parent_idx] = size_births[out_idx - 1]
            .max(size_births[out_idx])
            .max(death_size);

        idx -= 2;
    }

    Barcode {
        births: size_births,
        deaths: size_deaths,
        parents,
        lambda_deaths,
    }
}

/// Port of `compute_total_persistence`. The "binary searches" in the reference
/// are linear scans; they are reproduced as such because their tie behaviour
/// is part of the output.
pub fn compute_total_persistence(
    births: &[f32],
    deaths: &[f32],
    lambda_deaths: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let mut sizes: Vec<f32> = births.to_vec();
    sizes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sizes.dedup();

    let mut total_persistence = vec![0.0f32; sizes.len()];

    for i in 1..births.len() {
        let birth = births[i];
        let death = deaths[i];
        let lambda_death = lambda_deaths[i];

        if death <= birth {
            continue;
        }

        let mut birth_idx = 0usize;
        for (j, &s) in sizes.iter().enumerate() {
            if s >= birth {
                birth_idx = j;
                break;
            }
        }
        let mut death_idx = sizes.len();
        for (j, &s) in sizes.iter().enumerate() {
            if s >= death {
                death_idx = j;
                break;
            }
        }

        for k in birth_idx..death_idx {
            total_persistence[k] += (death - birth) * lambda_death;
        }
    }

    (sizes, total_persistence)
}

fn jaccard_similarity(a: &[usize], b: &[usize]) -> f64 {
    let set_a: std::collections::HashSet<usize> = a.iter().copied().collect();
    let mut union = set_a.clone();
    let mut intersection = 0usize;
    for &item in b {
        if set_a.contains(&item) {
            intersection += 1;
        } else {
            union.insert(item);
        }
    }
    if union.is_empty() {
        0.0
    } else {
        intersection as f64 / union.len() as f64
    }
}

fn estimate_cluster_similarity(births: &[f32], deaths: &[f32], birth_a: f32, birth_b: f32) -> f64 {
    let active = |at: f32| -> Vec<usize> {
        (0..births.len())
            .filter(|&i| births[i] <= at && deaths[i] > at)
            .collect()
    };
    jaccard_similarity(&active(birth_a), &active(birth_b))
}

/// Port of `select_diverse_peaks`: greedy pick by descending persistence,
/// rejecting peaks whose active-cluster set is too similar to one already
/// chosen.
///
/// The reference sorts with `np.argsort(...)[::-1]`, an unstable sort
/// reversed, so ties in persistence are implementation-defined there. This
/// port breaks ties by ascending peak index; if a fixture ever disagrees on a
/// tie, this is the place to look.
pub fn select_diverse_peaks(
    peaks: &[i64],
    total_persistence: &[f32],
    sizes: &[f32],
    births: &[f32],
    deaths: &[f32],
    min_similarity_threshold: f64,
    max_layers: usize,
) -> Vec<i64> {
    if peaks.is_empty() {
        return Vec::new();
    }

    let mut order: Vec<usize> = (0..peaks.len()).collect();
    order.sort_by(|&a, &b| {
        total_persistence[peaks[b] as usize]
            .partial_cmp(&total_persistence[peaks[a] as usize])
            .unwrap()
            .then(a.cmp(&b))
    });

    let mut selected_peaks = Vec::new();
    let mut selected_births: Vec<f32> = Vec::new();

    for &oi in &order {
        if selected_peaks.len() >= max_layers {
            break;
        }
        let peak = peaks[oi];
        let birth_size = sizes[peak as usize];

        let diverse = selected_births.iter().all(|&sel| {
            estimate_cluster_similarity(births, deaths, birth_size, sel) <= min_similarity_threshold
        });

        if diverse {
            selected_peaks.push(peak);
            selected_births.push(birth_size);
        }
    }
    selected_peaks
}
