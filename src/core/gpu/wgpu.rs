// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2021-2022 Adrian <adrian.eddy at gmail>

use std::borrow::Cow;
use std::sync::atomic::{ AtomicUsize, Ordering::SeqCst };
use wgpu::Adapter;
use wgpu::BufferUsages;
use parking_lot::{ RwLock, Mutex };
use crate::gpu:: { Buffers, BufferSource };
use crate::stabilization::KernelParams;
use crate::stabilization::distortion_models::DistortionModel;
use super::wgpu_interop::*;

#[derive(Debug)]
pub enum WgpuError {
    RequestDevice(wgpu::RequestDeviceError),
    ParamCheck,
    NoAvailableAdapter,
}

/// Turns the templated `wgpu_undistort.wgsl` source into a real WGSL module.
/// Extracted so the offline shader-validation tests exercise the exact same
/// substitutions the runtime does.
pub fn preprocess_kernel(lens_model_functions: &str, scalar: &str, uses_textures: bool) -> String {
    let mut kernel = include_str!("wgpu_undistort.wgsl").to_string();
    kernel = kernel.replace("LENS_MODEL_FUNCTIONS;", lens_model_functions);
    kernel = kernel.replace("SCALAR", scalar);

    if uses_textures {
        while let Some(pos) = kernel.find("{buffer_input}") {
            kernel.replace_range(pos..kernel.find("{/buffer_input}").unwrap() + 15, "");
        }
    } else {
        while let Some(pos) = kernel.find("{texture_input}") {
            kernel.replace_range(pos..kernel.find("{/texture_input}").unwrap() + 16, "");
        }
    }
    kernel
}

enum PipelineType {
    None,
    Render(wgpu::RenderPipeline),
    Compute(wgpu::ComputePipeline)
}

pub struct WgpuWrapper  {
    staging_buffer: Option<wgpu::Buffer>,
    buf_matrices: Option<wgpu::Buffer>,
    buf_params: Option<wgpu::Buffer>,
    buf_mesh_data: Option<wgpu::Buffer>,
    buf_drawing: Option<wgpu::Buffer>,

    in_texture: TextureHolder,
    out_texture: TextureHolder,

    pipeline: PipelineType,
    bind_group: Option<wgpu::BindGroup>,

    queue: wgpu::Queue,
    pub device: wgpu::Device,

    pixel_format: wgpu::TextureFormat,
    padded_out_stride: u32,
    in_size: u64,
    out_size: u64,
    params_size: u64,
    drawing_size: u64,

    /// The chroma smoothing pass (`wgpu_chroma_filter.wgsl`): render path, CPU output only.
    chroma: Option<WgpuChroma>,
    chroma_warned: std::sync::atomic::AtomicBool,
}

struct WgpuChroma {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    params: wgpu::Buffer,
    out: wgpu::Texture,
    guide: Option<wgpu::Texture>,
}
impl Drop for WgpuWrapper {
    fn drop(&mut self) {
        // We need to delete all texture references and then call device.poll() to actually release them properly
        self.staging_buffer = None;
        self.buf_matrices = None;
        self.buf_params = None;
        self.buf_mesh_data = None;
        self.buf_drawing = None;
        self.in_texture = TextureHolder::default();
        self.out_texture = TextureHolder::default();
        self.pipeline = PipelineType::None;
        self.bind_group = None;

        let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }
}

lazy_static::lazy_static! {
    static ref INSTANCE: Mutex<wgpu::Instance> = Mutex::new(wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle()));
    static ref ADAPTERS: RwLock<Vec<Adapter>> = RwLock::new(Vec::new());
    static ref ADAPTER: AtomicUsize = AtomicUsize::new(0);
}

const EXCLUSIONS: &[&'static str] = &["Microsoft Basic Render Driver"];

impl WgpuWrapper {
    pub fn list_devices() -> Vec<String> {
        if ADAPTERS.read().is_empty() {
            let devices = std::panic::catch_unwind(|| -> Vec<Adapter> {
                pollster::block_on(INSTANCE.lock().enumerate_adapters(wgpu::Backends::all())).into_iter().filter(|x| !EXCLUSIONS.iter().any(|e| x.get_info().name.contains(e))).collect()
            });
            match devices {
                Ok(devices) => { *ADAPTERS.write() = devices; },
                Err(e) => {
                    if let Some(s) = e.downcast_ref::<&str>() {
                        log::error!("Failed to initialize wgpu {}", s);
                    } else if let Some(s) = e.downcast_ref::<String>() {
                        log::error!("Failed to initialize wgpu {}", s);
                    } else {
                        log::error!("Failed to initialize wgpu {:?}", e);
                    }
                }
            }
        }

        ADAPTERS.read().iter().map(|x| { let x = x.get_info(); format!("{} ({:?})", x.name, x.backend) }).collect()
    }

    pub fn set_device(index: usize) -> Option<()> {
        let mut i = 0;
        for a in ADAPTERS.read().iter() {
            if i == index {
                let info = a.get_info();
                log::debug!("WGPU adapter ({i}): {:?}", &info);

                ADAPTER.store(i, SeqCst);
                return Some(());
            }
            i += 1;
        }
        None
    }
    pub fn get_info() -> Option<String> {
        let lock = ADAPTERS.read();
        if let Some(ref adapter) = lock.get(ADAPTER.load(SeqCst)) {
            let info = adapter.get_info();
            Some(format!("{} ({:?})", info.name, info.backend))
        } else {
            None
        }
    }

    pub fn initialize_context() -> Option<(String, String)> {
        let instance = INSTANCE.lock();

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        })).ok()?;
        let info = adapter.get_info();
        log::debug!("WGPU adapter: {:?}", &info);
        if info.device_type == wgpu::DeviceType::Cpu {
            return None;
        }

        let name = info.name.clone();
        let list_name = format!("[wgpu] {} ({:?})", info.name, info.backend);

        ADAPTERS.write().push(adapter);
        ADAPTER.store(ADAPTERS.read().len() - 1, SeqCst);

        Some((name, list_name))
    }

    pub fn new(params: &KernelParams, wgpu_format: (wgpu::TextureFormat, &str, bool), distortion_model: DistortionModel, digital_lens: Option<DistortionModel>, buffers: &Buffers, mut drawing_len: usize, lut: Option<std::sync::Arc<crate::color::Lut3D>>) -> Result<Self, WgpuError> {
        let max_matrix_count = 14 * if (params.flags & 16) == 16 { params.width } else { params.height } as usize;

        if params.height < 4 || params.output_height < 4 || buffers.input.size.0 < 16 || buffers.input.size.2 < 16 || buffers.output.size.0 < 16 || buffers.output.size.2 < 16 || params.width > 16384 || params.output_width > 16384 {
            return Err(WgpuError::ParamCheck);
        }

        let output_height = buffers.output.size.1 as i32;
        let output_stride = buffers.output.size.2 as i32;

        let in_size = (buffers.input.size.2 * buffers.input.size.1) as wgpu::BufferAddress;
        let out_size = (buffers.output.size.2 * buffers.output.size.1) as wgpu::BufferAddress;
        let params_size = (max_matrix_count * super::MAX_GATHER_FRAMES * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

        let drawing_enabled = (params.flags & 8) == 8;

        let mut adapter_id = ADAPTER.load(SeqCst);
        let adapter_initialized = ADAPTERS.read().get(adapter_id).is_some();
        if !adapter_initialized { Self::initialize_context(); }
        let lock = ADAPTERS.read();
        adapter_id = ADAPTER.load(SeqCst);

        #[cfg(any(target_os = "windows", target_os = "linux"))]
        if let BufferSource::CUDABuffer { .. } = buffers.input.data {
            adapter_id = super::wgpu_interop_cuda::get_current_device_id_by_uuid(&lock);
        }

        if let Some(adapter) = lock.get(adapter_id) {
            log::debug!("WGPU initializing adapter #{adapter_id}: {:?}", adapter.get_info());
            let (device, queue) = match &buffers.input.data {
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                BufferSource::Metal { command_queue, .. } |
                BufferSource::MetalBuffer { command_queue, .. } if !(*command_queue).is_null() => {
                    unsafe {
                        use objc2::rc::Retained;
                        use objc2::runtime::ProtocolObject;
                        use objc2_metal::*;
                        use wgpu::hal::api::Metal;

                        let mtl_cq: Retained<ProtocolObject<dyn MTLCommandQueue>> =
                            Retained::retain(*command_queue as *mut ProtocolObject<dyn MTLCommandQueue>).unwrap();
                        let mtl_dev = mtl_cq.device();

                        let max_buffer_bits = if cfg!(any(target_os = "android", target_os = "ios")) { 29 } else { 31 };
                        let max_storage_buffer_bits = if cfg!(any(target_os = "android", target_os = "ios")) { 27 } else { 31 };

                        let adapter_limits = adapter.limits();
                        let mut limits = wgpu::Limits::default().using_resolution(adapter_limits.clone());

                        limits.max_storage_buffers_per_shader_stage = 6.min(adapter_limits.max_storage_buffers_per_shader_stage);
                        limits.max_storage_textures_per_shader_stage = 4.min(adapter_limits.max_storage_textures_per_shader_stage);
                        limits.max_buffer_size = ((1 << max_buffer_bits) - 1).min(adapter_limits.max_buffer_size);
                        limits.max_storage_buffer_binding_size = ((1 << max_storage_buffer_bits) - 1+5).min(adapter_limits.max_storage_buffer_binding_size);

                        adapter.create_device_from_hal(wgpu::hal::OpenDevice::<Metal> {
                            device: <Metal as wgpu::hal::Api>::Device::device_from_raw(mtl_dev, wgpu::Features::empty(), &limits),
                            queue: <Metal as wgpu::hal::Api>::Queue::queue_from_raw(mtl_cq, 1.0)
                        }, &wgpu::DeviceDescriptor {
                            label: None,
                            required_features: wgpu::Features::empty(),
                            required_limits: limits,
                            memory_hints: wgpu::MemoryHints::Performance,
                            trace: wgpu::Trace::Off,
                            experimental_features: wgpu::ExperimentalFeatures::disabled(),
                        }).map_err(|e| WgpuError::RequestDevice(e))?
                    }
                },
                _ => {
                    let max_buffer_bits = if cfg!(any(target_os = "android", target_os = "ios")) { 29 } else { 31 };
                    let max_storage_buffer_bits = if cfg!(any(target_os = "android", target_os = "ios")) { 27 } else { 31 };
                    let adapter_limits = adapter.limits();

                    let mut limits = wgpu::Limits::default().using_resolution(adapter_limits.clone());

                    limits.max_storage_buffers_per_shader_stage = 6.min(adapter_limits.max_storage_buffers_per_shader_stage);
                    limits.max_storage_textures_per_shader_stage = 4.min(adapter_limits.max_storage_textures_per_shader_stage);
                    limits.max_buffer_size = ((1 << max_buffer_bits) - 1).min(adapter_limits.max_buffer_size);
                    limits.max_storage_buffer_binding_size = ((1 << max_storage_buffer_bits) - 1+5).min(adapter_limits.max_storage_buffer_binding_size);

                    let mut result = Err(WgpuError::NoAvailableAdapter);
                    for _ in 0..4 {
                        let device = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                            label: None,
                            required_features: wgpu::Features::empty(),
                            required_limits: limits.clone(),
                            memory_hints: wgpu::MemoryHints::Performance,
                            trace: wgpu::Trace::Off,
                            experimental_features: wgpu::ExperimentalFeatures::disabled(),
                        }));
                        if let Err(e) = &device {
                            let e_str = format!("{e:?}");
                            let re = regex::Regex::new("FailedLimit \\{ name: \"(.*?)\", requested: [0-9]+, allowed: ([0-9]+)").unwrap();
                            if let Some(captures) = re.captures(&e_str) {
                                log::debug!("Catching wgpu limit error: {e_str}");
                                let (_, [name, allowed]) = captures.extract();
                                match name {
                                    "max_storage_buffers_per_shader_stage"  => { limits.max_storage_buffers_per_shader_stage  = allowed.parse().unwrap(); continue; },
                                    "max_storage_textures_per_shader_stage" => { limits.max_storage_textures_per_shader_stage = allowed.parse().unwrap(); continue; },
                                    "max_buffer_size"                       => { limits.max_buffer_size                       = allowed.parse().unwrap(); continue; },
                                    "max_storage_buffer_binding_size"       => { limits.max_storage_buffer_binding_size       = allowed.parse().unwrap(); continue; },
                                    _ => { }
                                }
                            }
                        }
                        result = device.map_err(|e| WgpuError::RequestDevice(e));
                        break;
                    }
                    result?
                }
            };

            device.on_uncaptured_error(std::sync::Arc::new(|e| {
                log::error!("Uncaptured device error: {e:?}");
            }));

            let mut lens_model_functions = distortion_model.wgsl_functions().to_string();
            let default_digital_lens = "fn digital_undistort_point(uv: vec2<f32>) -> vec2<f32> { return uv; }
                                        fn digital_distort_point  (uv: vec2<f32>) -> vec2<f32> { return uv; }";
            lens_model_functions.push_str(digital_lens.as_ref().map(|x| x.wgsl_functions()).unwrap_or(default_digital_lens));

            if !drawing_enabled {
                drawing_len = 16;
            }

            let backend = adapter.get_info().backend;
            // The centre frame is layer 0; any temporal references follow it, so the
            // texture (and, where the backend allocates one directly, the buffer) must
            // be sized for all of them up front - it is not resized per-frame. Sized for
            // the largest window rather than for this emission's reference count, which
            // varies across a clip while this wrapper is built once - see
            // `super::MAX_GATHER_FRAMES`.
            let in_array_layers = if buffers.references.is_empty() { 1 } else { super::MAX_GATHER_FRAMES as u32 };
            let in_texture = init_texture(&device, backend, &buffers.input, wgpu_format.0, true, in_array_layers);
            let out_texture = init_texture(&device, backend, &buffers.output, wgpu_format.0, false, 1);

            let uses_textures = in_texture.wgpu_texture.is_some();
            let kernel = preprocess_kernel(&lens_model_functions, wgpu_format.1, uses_textures);
            // log::info!("Using kernel: {kernel}");

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(kernel)),
                label: None
            });

            let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as i32;
            let padding = (align - output_stride % align) % align;
            let padded_out_stride = output_stride + padding;
            let staging_size = padded_out_stride * output_height;

            let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor { size: staging_size as u64, usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST, label: None, mapped_at_creation: false });
            let buf_matrices  = device.create_buffer(&wgpu::BufferDescriptor { size: params_size, usage: BufferUsages::STORAGE | BufferUsages::COPY_DST, label: None, mapped_at_creation: false });
            let buf_params = device.create_buffer(&wgpu::BufferDescriptor { size: std::mem::size_of::<KernelParams>() as u64, usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST, label: None, mapped_at_creation: false });
            let buf_drawing = device.create_buffer(&wgpu::BufferDescriptor { size: drawing_len as u64, usage: BufferUsages::STORAGE | BufferUsages::COPY_DST, label: None, mapped_at_creation: false });
            let coeffs_data = crate::stabilization::build_coeffs_buffer(lut.as_deref());
            let buf_coeffs = device.create_buffer(&wgpu::BufferDescriptor {
                size: (coeffs_data.len() * std::mem::size_of::<f32>()) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                label: None, mapped_at_creation: false
            });
            queue.write_buffer(&buf_coeffs, 0, bytemuck::cast_slice(&coeffs_data));
            let buf_mesh_data = device.create_buffer(&wgpu::BufferDescriptor { size: (crate::gyro_source::splines::MAX_BUFFER_SIZE * std::mem::size_of::<f32>()).max(4096) as _, usage: BufferUsages::STORAGE | BufferUsages::COPY_DST, label: None, mapped_at_creation: false });

            let bind_group_layout = if uses_textures {
                let sample_type = match wgpu_format.1 {
                    "f32" => wgpu::TextureSampleType::Float { filterable: false },
                    "u32" => wgpu::TextureSampleType::Uint,
                    _ => { log::error!("Unknown texture scalar: {:?}", wgpu_format); wgpu::TextureSampleType::Float { filterable: false } }
                };
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<KernelParams>() as _) }, count: None },
                        wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: wgpu::BufferSize::new(params_size as _) }, count: None },
                        wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: wgpu::BufferSize::new((crate::stabilization::COEFFS.len() * std::mem::size_of::<f32>()) as _) }, count: None },
                        wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: wgpu::BufferSize::new(4096) }, count: None },
                        wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: wgpu::BufferSize::new(drawing_len as _) }, count: None },
                        wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type, view_dimension: wgpu::TextureViewDimension::D2Array, multisampled: false }, count: None },
                    ],
                    label: None,
                })
            } else {
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<KernelParams>() as _) }, count: None },
                        wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: wgpu::BufferSize::new(params_size as _) }, count: None },
                        wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: wgpu::BufferSize::new((crate::stabilization::COEFFS.len() * std::mem::size_of::<f32>()) as _) }, count: None },
                        wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: wgpu::BufferSize::new(4096) }, count: None },
                        wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: wgpu::BufferSize::new(drawing_len as _) }, count: None },
                        wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: wgpu::BufferSize::new(in_size as _) }, count: None },
                        wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: wgpu::BufferSize::new(out_size as _) }, count: None },
                    ],
                    label: None,
                })
            };

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0
            });

            let compilation_options = wgpu::PipelineCompilationOptions {
                constants: &[
                    ("100", params.interpolation     as f64),
                    ("101", params.pix_element_count as f64),
                    ("102", params.bytes_per_pixel   as f64),
                    ("103", params.flags             as f64),
                ],
                ..Default::default()
            };

            let pipeline = if uses_textures {
                PipelineType::Render(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: None,
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("undistort_vertex"),
                        buffers: &[],
                        compilation_options: compilation_options.clone(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("undistort_fragment"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu_format.0,
                            blend: None,
                            write_mask: wgpu::ColorWrites::default(),
                        })],
                        compilation_options,
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleStrip,
                        ..wgpu::PrimitiveState::default()
                    },
                    multiview_mask: None,
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    cache: Default::default()
                }))
            } else {
                PipelineType::Compute(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    module: &shader,
                    entry_point: Some("undistort_compute"),
                    label: None,
                    layout: Some(&pipeline_layout),
                    compilation_options,
                    cache: Default::default()
                }))
            };

            let bind_group = match &pipeline {
                PipelineType::None => None,
                PipelineType::Render(p) => {
                    Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &p.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: buf_params.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: buf_matrices.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 2, resource: buf_coeffs.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 3, resource: buf_mesh_data.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 4, resource: buf_drawing.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&in_texture.wgpu_texture.as_ref().unwrap().create_view(&wgpu::TextureViewDescriptor {
                                // The kernel declares `texture_2d_array` so it can gather temporal
                                // reference frames from layers 1... A plain 2D input texture is a
                                // one-layer array, so this view is valid before Task 7 uploads any.
                                dimension: Some(wgpu::TextureViewDimension::D2Array),
                                ..Default::default()
                            })) },
                        ],
                    }))
                },
                PipelineType::Compute(p) => {
                    Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &p.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: buf_params.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: buf_matrices.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 2, resource: buf_coeffs.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 3, resource: buf_mesh_data.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 4, resource: buf_drawing.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 5, resource: in_texture.wgpu_buffer.as_ref().unwrap().as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 6, resource: out_texture.wgpu_buffer.as_ref().unwrap().as_entire_binding() },
                        ],
                    }))
                }
            };

            // Chroma smoothing: render path + CPU output only (see the spec).
            let chroma = if uses_textures && matches!(buffers.output.data, BufferSource::Cpu { .. }) {
                let src_tex = out_texture.wgpu_texture.as_ref().unwrap();
                let size = wgpu::Extent3d { width: buffers.output.size.0 as u32, height: buffers.output.size.1 as u32, depth_or_array_layers: 1 };
                let out = device.create_texture(&wgpu::TextureDescriptor { label: None, size, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: wgpu_format.0, usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC, view_formats: &[] });
                // The guide is the same scalar type as the plane, one channel.
                let guide_format = match wgpu_format.0 {
                    wgpu::TextureFormat::R8Unorm | wgpu::TextureFormat::Rg8Unorm => Some(wgpu::TextureFormat::R8Unorm),
                    wgpu::TextureFormat::R16Uint | wgpu::TextureFormat::Rg16Uint => Some(wgpu::TextureFormat::R16Uint),
                    _ => None,
                };
                let guide = match (&buffers.guide, guide_format) {
                    (Some(g), Some(f)) => Some(device.create_texture(&wgpu::TextureDescriptor { label: None, size: wgpu::Extent3d { width: g.size.0 as u32, height: g.size.1 as u32, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: f, usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST, view_formats: &[] })),
                    _ => None,
                };
                let params = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: std::mem::size_of::<crate::denoise::chroma_filter::ChromaFilterParams>() as u64, usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST, mapped_at_creation: false });
                let src = include_str!("wgpu_chroma_filter.wgsl")
                    .replace("SCALAR", wgpu_format.1)
                    .replace("CF_ROUND", if wgpu_format.1 == "u32" { "round" } else { "" });
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Owned(src)) });
                let sample_type = if wgpu_format.1 == "u32" { wgpu::TextureSampleType::Uint } else { wgpu::TextureSampleType::Float { filterable: false } };
                let tex_entry = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None };
                let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[
                    wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    tex_entry(1), tex_entry(2),
                ] });
                let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
                let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: None, layout: Some(&pl),
                    vertex: wgpu::VertexState { module: &module, entry_point: Some("cf_vertex"), buffers: &[], compilation_options: Default::default() },
                    fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("cf_fragment"), targets: &[Some(wgpu::ColorTargetState { format: wgpu_format.0, blend: None, write_mask: wgpu::ColorWrites::default() })], compilation_options: Default::default() }),
                    primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, ..Default::default() },
                    multiview_mask: None, depth_stencil: None, multisample: wgpu::MultisampleState::default(), cache: Default::default(),
                });
                let src_view = src_tex.create_view(&Default::default());
                // RGB mode never reads the guide; bind the source again so the layout is satisfied.
                let guide_view = guide.as_ref().map(|g| g.create_view(&Default::default())).unwrap_or_else(|| src_tex.create_view(&Default::default()));
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout, entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&src_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&guide_view) },
                ] });
                Some(WgpuChroma { pipeline, bind_group, params, out, guide })
            } else { None };

            Ok(Self {
                chroma,
                chroma_warned: Default::default(),
                device,
                queue,
                staging_buffer: Some(staging_buffer),
                out_texture,
                in_texture,
                buf_matrices: Some(buf_matrices),
                buf_params: Some(buf_params),
                buf_drawing: Some(buf_drawing),
                buf_mesh_data: Some(buf_mesh_data),
                bind_group,
                pipeline,
                in_size,
                out_size,
                params_size,
                drawing_size: drawing_len as u64,
                pixel_format: wgpu_format.0,
                padded_out_stride: padded_out_stride as u32
            })
        } else {
            Err(WgpuError::NoAvailableAdapter)
        }
    }

    /// Records the chroma pass into `encoder` if it applies; returns whether it did. Skipping
    /// is the pass's only failure mode (see the spec): a reason to skip warns once.
    fn encode_chroma(&self, buffers: &Buffers, p: &crate::denoise::chroma_filter::ChromaFilterParams, encoder: &mut wgpu::CommandEncoder) -> bool {
        use crate::denoise::chroma_filter::MODE_PLANAR;
        if !p.is_active() { return false; }
        let skip = |why: &str| {
            if !self.chroma_warned.swap(true, std::sync::atomic::Ordering::Relaxed) { log::warn!("Chroma smoothing skipped: {why}"); }
            false
        };
        let Some(c) = &self.chroma else { return skip("this output, pixel format or pipeline is not supported"); };
        let mut kp = *p;
        kp.width = buffers.output.size.0 as i32;
        kp.height = buffers.output.size.1 as i32;
        kp.stride = buffers.output.size.2 as i32;
        if p.mode == MODE_PLANAR {
            let (Some(g), Some(gtex)) = (&buffers.guide, &c.guide) else { return skip("no luma guide for a chroma plane"); };
            let BufferSource::Cpu { buffer } = &g.data else { return skip("the luma guide is not a CPU buffer"); };
            let len = g.size.1 * g.size.2;
            if buffer.len() < len { return skip("the luma guide is smaller than its size says"); }
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: gtex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                &buffer[..len],
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(g.size.2 as u32), rows_per_image: None },
                wgpu::Extent3d { width: g.size.0 as u32, height: g.size.1 as u32, depth_or_array_layers: 1 },
            );
            kp.guide_width = g.size.0 as i32;
            kp.guide_height = g.size.1 as i32;
            kp.guide_stride = g.size.2 as i32;
        }
        self.queue.write_buffer(&c.params, 0, bytemuck::bytes_of(&kp));
        let view = c.out.create_view(&Default::default());
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None, depth_stencil_attachment: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view, resolve_target: None, depth_slice: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store } })],
        });
        rpass.set_pipeline(&c.pipeline);
        rpass.set_bind_group(0, &c.bind_group, &[]);
        rpass.draw(0..4, 0..1);
        true
    }

    pub fn undistort_image(&self, buffers: &mut Buffers, itm: &crate::stabilization::FrameTransform, drawing_buffer: &[u8]) -> bool {
        // Block layout the shader indexes into: frame f starts at
        // f * matrix_count * 14 floats. Centre first, then references in
        // temporal order.
        let mut all_matrices: Vec<[f32; 14]> = Vec::with_capacity(itm.matrices.len() * (1 + itm.reference_matrices.len()));
        all_matrices.extend_from_slice(&itm.matrices);
        for block in &itm.reference_matrices {
            debug_assert_eq!(block.len(), itm.matrices.len(), "every reference block must match the centre block length or the shader's stride arithmetic is wrong");
            all_matrices.extend_from_slice(block);
        }
        let matrices = bytemuck::cast_slice(&all_matrices);

        let in_size = (buffers.input.size.2 * buffers.input.size.1) as u64;
        let out_size = (buffers.output.size.2 * buffers.output.size.1) as u64;
        if self.in_size  != in_size  { log::error!("Buffer size mismatch! {} vs {}", self.in_size,  in_size);  return false; }
        if self.out_size != out_size { log::error!("Buffer size mismatch! {} vs {}", self.out_size, out_size); return false; }

        // The buffer path's frame offset is `frame * height * stride_px`, so every
        // concatenated reference frame must have identical stride and height to the
        // centre, or the shader silently samples garbage.
        for (i, r) in buffers.references.iter().enumerate() {
            debug_assert_eq!(r.size.1, buffers.input.size.1, "reference frame {i} has a different height ({}) than the centre frame ({}) - the shader's f*height*stride offset arithmetic would be wrong", r.size.1, buffers.input.size.1);
            debug_assert_eq!(r.size.2, buffers.input.size.2, "reference frame {i} has a different stride ({}) than the centre frame ({}) - the shader's f*height*stride offset arithmetic would be wrong", r.size.2, buffers.input.size.2);
        }

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let _temp_texture = handle_input_texture(&self.device, &buffers.input, &self.queue, &mut encoder, &self.in_texture, self.pixel_format, self.padded_out_stride);

        // How many references actually made it into the texture/buffer. The kernel walks
        // frames 1..reference_count contiguously, so a reference that fails to upload
        // truncates the window rather than merely being absent - everything after it would be
        // sampled from a layer or offset that was never written.
        let mut uploaded_references = 0usize;

        // Upload reference frames into the layers/offsets that follow the centre frame.
        if !buffers.references.is_empty() {
            if let Some(in_tex) = self.in_texture.wgpu_texture.as_ref() {
                for (i, r) in buffers.references.iter().enumerate() {
                    match &r.data {
                        BufferSource::Cpu { buffer } => {
                            self.queue.write_texture(
                                wgpu::TexelCopyTextureInfo {
                                    texture: in_tex,
                                    mip_level: 0,
                                    origin: wgpu::Origin3d { x: 0, y: 0, z: (i + 1) as u32 },
                                    aspect: wgpu::TextureAspect::All,
                                },
                                bytemuck::cast_slice(buffer),
                                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(r.size.2 as u32), rows_per_image: None },
                                wgpu::Extent3d { width: r.size.0 as u32, height: r.size.1 as u32, depth_or_array_layers: 1 },
                            );
                            uploaded_references += 1;
                        }
                        _ => {
                            // Stop rather than skip - see `uploaded_references`.
                            log::warn!("Reference frame {i} upload is not implemented for this backend yet; denoising will use the {uploaded_references} reference(s) before it");
                            break;
                        }
                    }
                }
            } else if let Some(in_buf) = self.in_texture.wgpu_buffer.as_ref() {
                let capacity = in_buf.size();
                // `read_input_at`'s buffer branch steps between frames by
                // `params.height * stride_px`, which is not the same as this plane's own
                // frame size on a subsampled plane - see `super::reference_frame_pitch`.
                let pitch = super::reference_frame_pitch(&itm.kernel_params, &buffers.input, in_size as usize).unwrap_or(0) as u64;
                let required = pitch * (1 + buffers.references.len()) as u64;
                if pitch == 0 || capacity < required {
                    log::error!("Reference frame buffer too small for {} references ({} < {})", buffers.references.len(), capacity, required);
                } else {
                    for (i, r) in buffers.references.iter().enumerate() {
                        if let BufferSource::Cpu { buffer } = &r.data {
                            self.queue.write_buffer(in_buf, (i as u64 + 1) * pitch, bytemuck::cast_slice(buffer));
                            uploaded_references += 1;
                        } else {
                            // Stop rather than skip - see `uploaded_references`.
                            log::warn!("Reference frame {i} upload is not implemented for this backend yet; denoising will use the {uploaded_references} reference(s) before it");
                            break;
                        }
                    }
                }
            }
        }

        if self.params_size < matrices.len() as u64    { log::error!("Buffer size mismatch! {} vs {}", self.params_size, matrices.len()); return false; }

        self.queue.write_buffer(self.buf_matrices.as_ref().unwrap(), 0, matrices);
        let mut kernel_params = itm.kernel_params;
        kernel_params.reference_count = (1 + uploaded_references) as f32;
        self.queue.write_buffer(self.buf_params.as_ref().unwrap(), 0, bytemuck::bytes_of(&kernel_params));
        if !drawing_buffer.is_empty() {
            if self.drawing_size < drawing_buffer.len() as u64 { log::error!("Buffer size mismatch! {} vs {}", self.drawing_size, drawing_buffer.len()); return false; }
            self.queue.write_buffer(self.buf_drawing.as_ref().unwrap(), 0, drawing_buffer);
        }
        if !itm.mesh_data.is_empty() {
            if self.buf_mesh_data.is_none() || (self.buf_mesh_data.as_ref().unwrap().size() as usize * 4) < itm.mesh_data.len() { log::error!("Buffer size mismatch buf_mesh_data! {} vs {}", self.buf_mesh_data.as_ref().unwrap().size() * 4, itm.mesh_data.len()); return false; }
            self.queue.write_buffer(self.buf_mesh_data.as_ref().unwrap(), 0, bytemuck::cast_slice(&itm.mesh_data));
        }

        match &self.pipeline {
            PipelineType::None => { },
            PipelineType::Compute(p) => {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                cpass.set_pipeline(p);
                cpass.set_bind_group(0, self.bind_group.as_ref(), &[]);
                cpass.dispatch_workgroups((buffers.output.size.0 as f32 / 8.0).ceil() as u32, (buffers.output.size.1 as f32 / 8.0).ceil() as u32, 1);
            },
            PipelineType::Render(p) => {
                let view = self.out_texture.wgpu_texture.as_ref().unwrap().create_view(&wgpu::TextureViewDescriptor::default());
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None
                    })],
                    multiview_mask: None,
                    depth_stencil_attachment: None,
                });
                rpass.set_pipeline(p);
                rpass.set_bind_group(0, self.bind_group.as_ref(), &[]);
                rpass.draw(0..6, 0..1);
            }
        }

        let chroma_ran = self.encode_chroma(buffers, &itm.chroma_filter, &mut encoder);
        let _temp_texture2 = if chroma_ran {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo { texture: &self.chroma.as_ref().unwrap().out, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyBufferInfo { buffer: self.staging_buffer.as_ref().unwrap(), layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(self.padded_out_stride), rows_per_image: None } },
                wgpu::Extent3d { width: buffers.output.size.0 as u32, height: buffers.output.size.1 as u32, depth_or_array_layers: 1 },
            );
            None
        } else {
            handle_output_texture(&self.device, &buffers.output, &self.queue, &mut encoder, &self.out_texture, self.pixel_format, self.staging_buffer.as_ref().unwrap(), self.padded_out_stride)
        };

        let sub_index = self.queue.submit(Some(encoder.finish()));

        match &mut buffers.output.data {
            BufferSource::Cpu { buffer, .. } => {
                let buffer_slice = self.staging_buffer.as_ref().unwrap().slice(..);
                let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
                buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

                let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

                if let Some(Ok(())) = pollster::block_on(receiver.receive()) {
                    if let Ok(data) = buffer_slice.get_mapped_range() {
                        if self.padded_out_stride == buffers.output.size.2 as u32 {
                            // Fast path
                            (&mut buffer[..buffers.output.size.1 * buffers.output.size.2]).copy_from_slice(data.as_ref());
                        } else {
                            // data.as_ref()
                            //     .chunks(self.padded_out_stride as usize)
                            //     .zip(output.chunks_mut(buffers.output_size.2))
                            //     .for_each(|(src, dest)| {
                            //         dest.copy_from_slice(&src[0..buffers.output_size.2]);
                            //     });
                            use rayon::prelude::{ ParallelSliceMut, ParallelSlice };
                            use rayon::iter::{ ParallelIterator, IndexedParallelIterator };
                            data.as_ref()
                                .par_chunks(self.padded_out_stride as usize)
                                .zip(buffer.par_chunks_mut(buffers.output.size.2))
                                .for_each(|(src, dest)| {
                                    dest.copy_from_slice(&src[0..buffers.output.size.2]);
                                });
                        }

                        // We have to make sure all mapped views are dropped before we unmap the buffer.
                        drop(data);
                    }
                    self.staging_buffer.as_ref().unwrap().unmap();
                } else {
                    // TODO change to Result
                    log::error!("failed to run compute on wgpu!");
                    return false;
                }
            }
            _ => { handle_output_texture_post(&self.device, &buffers.output, &self.out_texture, self.pixel_format, sub_index); }
        }

        true
    }
}

pub fn is_buffer_supported(buffers: &Buffers) -> bool {
    match buffers.input.data {
        BufferSource::None           => false,
        BufferSource::Cpu     { .. } => true,
        BufferSource::OpenGL  { .. } => false,
        #[cfg(target_os = "windows")]
        BufferSource::DirectX11 { .. } => true,
        #[cfg(feature = "use-opencl")]
        BufferSource::OpenCL  { .. } => false,
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        BufferSource::Vulkan  { .. } => true,
        #[cfg(any(target_os = "windows", target_os = "linux"))]
        BufferSource::CUDABuffer{ .. } => true,
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        BufferSource::Metal { .. } | BufferSource::MetalBuffer { .. } => true,
    }
}

/// Parses and validates `wgpu_undistort.wgsl` with naga, offline, with no GPU and no
/// running app. Every substitution variant the runtime can produce is covered, so a
/// syntax or type error in the shader fails `cargo test` instead of only showing up as a
/// runtime shader-compilation error once a video is loaded.
#[cfg(test)]
mod wgsl_validation_tests {
    use super::*;

    fn validate(scalar: &str, uses_textures: bool) {
        let mut lens_model_functions = DistortionModel::default().wgsl_functions().to_string();
        lens_model_functions.push_str("fn digital_undistort_point(uv: vec2<f32>) -> vec2<f32> { return uv; }
                                       fn digital_distort_point  (uv: vec2<f32>) -> vec2<f32> { return uv; }");

        let source = preprocess_kernel(&lens_model_functions, scalar, uses_textures);

        let module = match wgpu::naga::front::wgsl::parse_str(&source) {
            Ok(m) => m,
            Err(e) => panic!("WGSL parse failed (scalar={scalar}, textures={uses_textures}):\n{}", e.emit_to_string(&source)),
        };
        let mut validator = wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        );
        if let Err(e) = validator.validate(&module) {
            panic!("WGSL validation failed (scalar={scalar}, textures={uses_textures}):\n{}", e.emit_to_string(&source));
        }
    }

    #[test] fn wgsl_validates_f32_texture_input() { validate("f32", true); }
    #[test] fn wgsl_validates_f32_buffer_input()  { validate("f32", false); }
    #[test] fn wgsl_validates_u32_texture_input() { validate("u32", true); }
    #[test] fn wgsl_validates_u32_buffer_input()  { validate("u32", false); }

    /// The offset the shader reads LUT data at must match the Rust constant it mirrors.
    #[test]
    fn lut_data_offset_matches_rust() {
        let src = include_str!("wgpu_undistort.wgsl");
        let expected = format!("const LUT_DATA_OFFSET: u32 = {}u;", crate::stabilization::LUT_DATA_OFFSET);
        assert!(src.contains(&expected), "wgpu_undistort.wgsl must contain `{expected}`");
    }
}
