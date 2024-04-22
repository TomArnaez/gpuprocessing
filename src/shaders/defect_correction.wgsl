const defect_conv_matrix = mat3x3f(
    0.125, 0.125, 0.125,
    0.125, 0,     0.125,
    0.125, 0.125, 0.125
);

@group(0) @binding(0) var inputImage: texture_2d<f32>;