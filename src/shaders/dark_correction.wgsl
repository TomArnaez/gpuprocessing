@group(0) @binding(0) var imageInput: texture_2d<f32>;
@group(0) @binding(1) var gainMap: texture_2d<f32>;
@group(0) @binding(2) var imageOutput: texture_storage_2d<r16unorm, write>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let imageValue = textureLoad(imageInput, global_id.xy, 0).r;
    let gainValue = textureLoad(darkMap, global_id.xy, 0).r;
    let correctedValue: f32 = imageValue * gainValue;
    let outputValue: f32 = clamp(correctedValue, 0.0, 1.0);
    textureStore(imageOutput, global_id.xy, vec4<f32>(outputValue, 0.0, 0.0, 1.0));
}