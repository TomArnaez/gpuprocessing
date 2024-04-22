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

    let mut required_limits = wgpu::Limits::default();
    required_limits.max_texture_dimension_2d = 2560;

    let (device, queue) = 
        adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::TEXTURE_FORMAT_16BIT_NORM | wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
                required_limits
            },
            None,
        ).await.expect("Failed to get device");

    let wgpu = WgpuState {
        device,
        queue
    };

    let query_set = wgpu.device.create_query_set(&wgpu::QuerySetDescriptor {
        ty: wgpu::QueryType::Timestamp,
        count: 2,
        label: Some("Timestamp Query Set"),
    });

    const WIDTH: u32 = 2560;
    const HEIGHT: u32 = 2560;
    const FRAME_SIZE: usize = (WIDTH * HEIGHT) as usize;

    let size = wgpu::Extent3d {
        width: WIDTH,
        height: HEIGHT,
        depth_or_array_layers: 1,
    };

    let image_data = vec![10u16; FRAME_SIZE];

    let image_texture = wgpu.device.create_texture(
        &wgpu::TextureDescriptor { 
            label: Some("Image Texture"), 
            size, 
            mip_level_count: 1, 
            sample_count: 1, 
            dimension: wgpu::TextureDimension::D2, 
            format: wgpu::TextureFormat::R16Unorm, 
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[] 
            });
        
    wgpu.queue.write_texture(
        wgpu::ImageCopyTexture {
            aspect: wgpu::TextureAspect::All,
            texture: &image_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        bytemuck::cast_slice(&image_data),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(WIDTH * 2),
            rows_per_image: Some(HEIGHT),
        },
        size,
    );

    let gain_data = vec![5u16; FRAME_SIZE];
    let dark_data = vec![3u16; FRAME_SIZE];

    let gain_texture = wgpu.device.create_texture(
        &wgpu::TextureDescriptor { 
            label: Some("Image Texture"), 
            size, 
            mip_level_count: 1, 
            sample_count: 1, 
            dimension: wgpu::TextureDimension::D2, 
            format: wgpu::TextureFormat::R16Unorm, 
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[] 
    });
        
    wgpu.queue.write_texture(
        wgpu::ImageCopyTexture {
            aspect: wgpu::TextureAspect::All,
            texture: &gain_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        bytemuck::cast_slice(&gain_data),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(WIDTH * 2),
            rows_per_image: Some(HEIGHT),
        },
        size,
    );

    let dark_texture = wgpu.device.create_texture(
        &wgpu::TextureDescriptor { 
            label: Some("Image Texture"), 
            size, 
            mip_level_count: 1, 
            sample_count: 1, 
            dimension: wgpu::TextureDimension::D2, 
            format: wgpu::TextureFormat::R16Unorm, 
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[] 
    });
        
    wgpu.queue.write_texture(
        wgpu::ImageCopyTexture {
            aspect: wgpu::TextureAspect::All,
            texture: &dark_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        bytemuck::cast_slice(&dark_data),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(WIDTH * 2),
            rows_per_image: Some(HEIGHT),
        },
        size,
    );

    let output_texture = wgpu.device.create_texture(
        &wgpu::TextureDescriptor { 
            label: Some("Image Texture"), 
            size, 
            mip_level_count: 1, 
            sample_count: 1, 
            dimension: wgpu::TextureDimension::D2, 
            format: wgpu::TextureFormat::R16Unorm, 
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[] 
    });

    let output_buffer = &wgpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        mapped_at_creation: false,
        size: (FRAME_SIZE * 2) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ
    });

    let query_resolve_buffer = wgpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Timestamp Buffer"),
        size: 16,  // 2 timestamps, 8 bytes each
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
        mapped_at_creation: false,
    });

    let query_destination_buffer = wgpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Query Destination Buffer"),
        size: 16,  // 2 timestamps, 8 bytes each
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let gain_correction_compute = GainCorrectionCompute::new(&wgpu, &image_texture, &gain_texture, &output_texture);
    let dark_correction_compute = DarkCorrectionCompute::new(&wgpu, &image_texture, &dark_texture, &output_texture, 0.);

    let mut encoder = wgpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Compute Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Correction Compute Pass"),
            timestamp_writes: None
        });

        compute_pass.write_timestamp(&query_set, 0);
        //gain_correction_compute.run(&mut compute_pass);
        dark_correction_compute.run(&mut compute_pass);
        compute_pass.write_timestamp(&query_set, 1);

    }

    encoder.resolve_query_set(&query_set, 0..2, &query_resolve_buffer, 0);
    encoder.copy_buffer_to_buffer(&query_resolve_buffer, 0, &query_destination_buffer, 0, 16);
    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &output_texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: output_buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(WIDTH * 2),
                rows_per_image: Some(HEIGHT),
            }
        },
        size,
    );

    wgpu.queue.submit(Some(encoder.finish()));

    let query_dest_slice = query_destination_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    query_dest_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    let buffer_slice = output_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    wgpu.device.poll(wgpu::Maintain::wait()).panic_on_timeout();
    if let Ok(Ok(())) = receiver.recv_async().await {
        let data = buffer_slice.get_mapped_range();
        let result: Vec<u16> = bytemuck::cast_slice(&data).to_vec();
        println!("{:?}", &result[0..10]);
        drop(data);
        output_buffer.unmap();

        let query_data = query_dest_slice.get_mapped_range();
        let timestamps: Vec<f32> = bytemuck::cast_slice(&query_data).to_vec();
        let duration = timestamps[1] - timestamps[0];
        let timestamp_period = wgpu.queue.get_timestamp_period();  // This gives the period in nanoseconds per tick
        let duration_ns = duration * timestamp_period;
        println!("Duration: {} ns", duration_ns);
    } else {
        panic!("failed to run compute on gpu!")
    }
}

pub struct GainCorrectionCompute {
    compute_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup
}

impl GainCorrectionCompute {
    fn new(
        wgpu: &WgpuState, 
        image_texture: &wgpu::Texture, 
        gain_map_texture: &wgpu::Texture, 
        output_texture: &wgpu::Texture
    ) -> Self {
        let shader = wgpu.device.create_shader_module(wgpu::include_wgsl!("shaders/gain_correction.wgsl"));

        #[repr(C)]
        #[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]  
        struct GainUniformData {
            min_pixel_value: f32,
            max_pixel_value: f32,
        }

        let uniform_data = GainUniformData { 
            min_pixel_value: 0.,
            max_pixel_value: 1. 
        };

        let uniform_buffer = wgpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniform_data]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = wgpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture { 
                        access: wgpu::StorageTextureAccess::WriteOnly, 
                        format: wgpu::TextureFormat::R16Unorm, 
                        view_dimension: wgpu::TextureViewDimension::D2 
                    },
                    count: None
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { 
                        ty: wgpu::BufferBindingType::Uniform, 
                        has_dynamic_offset: false, 
                        min_binding_size: None, 
                    },
                    count: None,
                },
            ],
            label: Some("Gain Correction Bind Group Layout"),
        });

        let pipeline_layout = wgpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
    
        let compute_pipeline = wgpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Histogram Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        let bind_group = wgpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &compute_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&image_texture.create_view(&wgpu::TextureViewDescriptor::default())),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&gain_map_texture.create_view(&wgpu::TextureViewDescriptor::default())),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&output_texture.create_view(&wgpu::TextureViewDescriptor::default()))
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(uniform_buffer.as_entire_buffer_binding())
                }
            ],
        });

        GainCorrectionCompute {
            compute_pipeline,
            bind_group
        }
    }

    fn run<'cpass>(&'cpass self, cpass: &mut wgpu::ComputePass<'cpass>) {
        cpass.set_pipeline(&self.compute_pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch_workgroups(1024 / 256, 1, 1);
    }
}

pub struct DarkCorrectionCompute {
    compute_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup
}

impl DarkCorrectionCompute {
    fn new(
        wgpu: &WgpuState, 
        image_texture: &wgpu::Texture, 
        dark_map_texture: &wgpu::Texture, 
        output_texture: &wgpu::Texture,
        offset_value: f32,
    ) -> Self {
        let shader = wgpu.device.create_shader_module(wgpu::include_wgsl!("shaders/dark_correction.wgsl"));

        #[repr(C)]
        #[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]  
        struct DarkUniformData {
            offset_value: f32,
        }

        let uniform_data = DarkUniformData { 
            offset_value
        };

        let uniform_buffer = wgpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniform_data]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = wgpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture { 
                        access: wgpu::StorageTextureAccess::WriteOnly, 
                        format: wgpu::TextureFormat::R16Unorm, 
                        view_dimension: wgpu::TextureViewDimension::D2 
                    },
                    count: None
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { 
                        ty: wgpu::BufferBindingType::Uniform, 
                        has_dynamic_offset: false, 
                        min_binding_size: None, 
                    },
                    count: None,
                },
            ],
            label: Some("Gain Correction Bind Group Layout"),
        });

        let pipeline_layout = wgpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
    
        let compute_pipeline = wgpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Histogram Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        let bind_group = wgpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &compute_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&image_texture.create_view(&wgpu::TextureViewDescriptor::default())),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&dark_map_texture.create_view(&wgpu::TextureViewDescriptor::default())),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&output_texture.create_view(&wgpu::TextureViewDescriptor::default()))
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(uniform_buffer.as_entire_buffer_binding())
                }
            ],
        });

        DarkCorrectionCompute {
            compute_pipeline,
            bind_group
        }
    }

    fn run<'cpass>(&'cpass self, cpass: &mut wgpu::ComputePass<'cpass>) {
        cpass.set_pipeline(&self.compute_pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch_workgroups(1024 / 256, 1, 1);
    }
}