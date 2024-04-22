@group(0) @binding(0) var<storage, read> inputData: array<u32>;
@group(0) @binding(1) var<storage, read_write> globalHistogram: array<atomic<u32>>;

var<workgroup> localHistogram: array<atomic<u32>, 256>; // Local histogram in shared memory

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) globalId : vec3<u32>,
        @builtin(local_invocation_id) localId : vec3<u32>,
        @builtin(num_workgroups) numWorkgroups : vec3<u32>) {
    let index = globalId.x;
    let numBins = 256u; // Adjust the number of bins based on your data

    // Initialize local histogram in shared memory
    for (var i = localId.x; i < numBins; i += 256u) {
        localHistogram[i] = 0u;
    }
    workgroupBarrier();

    // Calculate the local histogram
    if (index < arrayLength(&inputData)) {
        let value = inputData[index];
        let binIndex = calculateBinIndex(value, numBins);
        atomicAdd(&localHistogram[binIndex], 1u);
    }
    workgroupBarrier();

    // Reduce local histograms into the global histogram
    for (var i = localId.x; i < numBins; i += 256u) {
        atomicAdd(&globalHistogram[i], localHistogram[i]);
    }
}

fn calculateBinIndex(value: u32, numBins: u32) -> u32 {
    let maxValue = 1000u; // Adapt this to the range of your data
    return (value * numBins) / (maxValue + 1u);
}
