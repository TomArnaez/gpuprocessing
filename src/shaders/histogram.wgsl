@group(0) @binding(0) var myTexture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> histogram: array<u32, 256>;

let numBins: u32 = 256u;
let maxValue: u32 = 65535u;

var<workgroup> localHistograms: array<array<u32>, 256>, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn main(
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(num_subgroups) num_subgroups: u32,
        @builtin(subgroup_id) subgroup_id: u32,
        @builtin(subgroup_size) subgroup_size: u32,
        @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
        @builtin(num_workgroups) groups: vec3<u32>,
) {
    let this_histogram = array<u32, 256>(0u);

    let normedValue = textureLoad(inputImage, global_id.xy, 0).r;
    let actualValue = u32(normedValue * f32(maxValue));
    let binIndex = min(actualValue * numBins / (maxValue + 1), numBins - 1);

    // Per subgroup histogram
    for (i: u32 = 0; i < numBins; ++i) {
        subgroup_sum: u32 = subgroupAdd(this_histogram[i]);
        if (subgroupElect()) {
            localHistograms[subgroup_id][i] = subgroup_sum;
        }
    }

    workgroupBarrier();

    // Per workgroup histogram
    if (local_invocation_id.x < numBins) {
        bucket_sum: u32 = 0;
        for (i: u32 = 0; i < subgroup_size; ++i) {
            bucket_sum += localHistograms[i][local_invocation_id.x];
        }
        histogram[local_invocation_id.x] = bucket_sum;
    }
}
