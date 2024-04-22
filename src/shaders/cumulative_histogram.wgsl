@group(0) @binding(0) var<storage, read> histogram: array<u32>;
@group(0) @binding(1) var<storage, read_write> cumulative_histogram: array<u32>;

var<workgroup> shared_cumulative_histogram: array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
        @builtin(local_invocation_id) localId : vec3<u32>,
        @builtin(num_workgroups) numWorkgroups : vec3<u32>) {
    shared_cumulative_histogram[localId.x] = histogram[localId.x];
    workgroupBarrier();

    for (var offset: u32 = 1; offset < 256u; offset *= 2u) {
        if (localId.x >= offset) {
            shared_cumulative_histogram[localId.x] += shared_cumulative_histogram[localId.x - offset];
        }
        workgroupBarrier();
    }

    cumulative_histogram[localId.x] = shared_cumulative_histogram[localId.x];
}

fn calculateBinIndex(value: u32, numBins: u32) -> u32 {
    let maxValue = 1000u; // Adapt this to the range of your data
    return (value * numBins) / (maxValue + 1u);
}