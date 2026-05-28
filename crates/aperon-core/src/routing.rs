use crate::{distance::l2_squared, Distance, GrainId};

/// Chosen grain and score for a query route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Route {
    pub grain_id: GrainId,
    pub distance: Distance,
}

/// Centroid router for assigning queries to local grains.
#[derive(Clone, Debug)]
pub struct CentroidRouter {
    dim: usize,
    centroids: Vec<(GrainId, Vec<f32>)>,
}

impl CentroidRouter {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            centroids: Vec::new(),
        }
    }

    pub fn add_centroid(
        &mut self,
        grain_id: GrainId,
        centroid: impl Into<Vec<f32>>,
    ) -> Result<(), String> {
        let centroid = centroid.into();
        if centroid.len() != self.dim {
            return Err(format!(
                "dimension mismatch: expected {}, got {}",
                self.dim,
                centroid.len()
            ));
        }

        self.centroids.push((grain_id, centroid));
        Ok(())
    }

    pub fn route(&self, query: &[f32]) -> Option<Route> {
        if query.len() != self.dim {
            return None;
        }

        self.centroids
            .iter()
            .filter_map(|(grain_id, centroid)| {
                l2_squared(query, centroid).map(|distance| Route {
                    grain_id: *grain_id,
                    distance,
                })
            })
            .min_by(|a, b| a.distance.total_cmp(&b.distance))
    }
}
