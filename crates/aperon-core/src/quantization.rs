/// Linear scalar quantizer used for local tangent coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quantizer {
    scale: f32,
}

impl Quantizer {
    pub fn new(scale: f32) -> Result<Self, QuantizationError> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(QuantizationError::InvalidScale);
        }

        Ok(Self { scale })
    }

    pub fn quantize_i16(self, value: f32) -> i16 {
        quantize_i16(value, self.scale)
    }

    pub fn dequantize_i16(self, value: i16) -> f32 {
        value as f32 * self.scale
    }

    pub fn scale(self) -> f32 {
        self.scale
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuantizationError {
    InvalidScale,
}

pub fn quantize_i16(value: f32, scale: f32) -> i16 {
    quantize_signed(value, scale, i16::MIN as i32, i16::MAX as i32) as i16
}

pub fn quantize_i8(value: f32, scale: f32) -> i8 {
    quantize_signed(value, scale, i8::MIN as i32, i8::MAX as i32) as i8
}

pub fn quantize_u16(value: f32, scale: f32) -> u16 {
    let scaled = scaled_value(value, scale);
    scaled.round_ties_even().clamp(0.0, u16::MAX as f32) as u16
}

pub fn scale_for_i16(abs_max: f32) -> f32 {
    if abs_max > 0.0 {
        abs_max / i16::MAX as f32
    } else {
        1.0
    }
}

pub fn scale_for_i8(abs_max: f32) -> f32 {
    if abs_max > 0.0 {
        abs_max / i8::MAX as f32
    } else {
        1.0
    }
}

pub fn scale_for_u16(max: f32) -> f32 {
    if max > 0.0 {
        max / u16::MAX as f32
    } else {
        1.0
    }
}

pub fn scaled_weight(component_scale: f64, unit: f64, max_weight: i64) -> i64 {
    if component_scale <= 0.0 || unit <= 0.0 {
        return 1;
    }

    (component_scale / unit)
        .round()
        .clamp(1.0, max_weight as f64) as i64
}

fn quantize_signed(value: f32, scale: f32, min: i32, max: i32) -> i32 {
    let scaled = scaled_value(value, scale);
    scaled.round_ties_even().clamp(min as f32, max as f32) as i32
}

fn scaled_value(value: f32, scale: f32) -> f32 {
    if scale > 0.0 {
        value / scale
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_scale() {
        assert_eq!(Quantizer::new(0.0), Err(QuantizationError::InvalidScale));
    }

    #[test]
    fn round_trips_on_scale_grid() {
        let quantizer = Quantizer::new(0.5).unwrap();
        assert_eq!(quantizer.quantize_i16(3.0), 6);
        assert_eq!(quantizer.dequantize_i16(6), 3.0);
    }

    #[test]
    fn uses_ties_to_even_like_cpp_lrint() {
        let quantizer = Quantizer::new(1.0).unwrap();
        assert_eq!(quantizer.quantize_i16(2.5), 2);
        assert_eq!(quantizer.quantize_i16(3.5), 4);
        assert_eq!(quantizer.quantize_i16(-2.5), -2);
    }
}
