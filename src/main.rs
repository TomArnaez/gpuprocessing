use wgpu::util::DeviceExt;

pub struct WgpuState {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

#[tokio::main]
async fn main() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });

    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        force_fallback_adapter: false,
        compatible_surface: None,
    }).await.expect("Failed to get adapter");

    let (device, queue) = 
        adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::TEXTURE_FORMAT_16BIT_NORM,
                required_limits: wgpu::Limits::downlevel_defaults()
            },
            None,
        ).await.expect("Failed to get device");

    let wgpu_state = WgpuState {
        device,
        queue
    };

    test_dark_shader(&wgpu_state).await;
}

async fn test_dark_shader(wgpu: &WgpuState) {
    // Create shader
    let shader = wgpu.device.create_shader_module(wgpu::include_wgsl!("shaders/histogram.wgsl"));

    // Create compute pipeline
    let compute_pipeline = wgpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Histogram Compute Pipeline"),
        layout: None, // We'll let WebGPU infer the layout based on the shader
        module: &shader,
        entry_point: "main",
    });

    // Setup input data
    let input_data: Vec<u32> = vec![10u32; 3000*3000];
    let input_buffer = wgpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: bytemuck::cast_slice(&input_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Setup histogram buffer
    let num_bins = 256u32; // Adjust according to your needs

    let hist_size = (num_bins * std::mem::size_of::<u32>() as u32) as u64;

    let histogram_buffer = wgpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Histogram Buffer"),
        size: hist_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buffer = wgpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: hist_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create bind group
    let bind_group = wgpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Compute Bind Group"),
        layout: &compute_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(input_buffer.as_entire_buffer_binding()),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(histogram_buffer.as_entire_buffer_binding()),
            },
        ],
    });

    // Create a command buffer to dispatch the compute work
    let mut encoder = wgpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Compute Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(input_data.len() as u32 / 256, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&histogram_buffer, 0, &staging_buffer, 0, hist_size);

    wgpu.queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    wgpu.device.poll(wgpu::Maintain::wait()).panic_on_timeout();
    if let Ok(Ok(())) = receiver.recv_async().await {
        let data = buffer_slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        println!("{:?}", result);
        drop(data);
        staging_buffer.unmap();
    } else {
        panic!("failed to run compute on gpu!")
    }
}