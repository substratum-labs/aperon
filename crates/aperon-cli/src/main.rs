use aperon_core::AperonIndex;

fn main() {
    let index = AperonIndex::new(0);
    let stats = index.stats();

    println!(
        "aperon workspace initialized: dim={}, grains={}, vectors={}",
        stats.dim, stats.grains, stats.vectors
    );
}
