//! Small xorshift RNG for the stochastic stages. The reference draws from
//! numpy's RandomState; seeds are not comparable across implementations, so
//! the stochastic half is validated by ARI bounds, not bitwise.

#[derive(Clone)]
pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Avoid the all-zero fixed point and decorrelate small seeds.
        Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    #[inline]
    pub fn unit_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / (1u64 << 24) as f32
    }

    /// Standard normal via Box-Muller.
    pub fn gauss(&mut self) -> f32 {
        let u1 = self.unit_f32().max(1e-12);
        let u2 = self.unit_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }
}
