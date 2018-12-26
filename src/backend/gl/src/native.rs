use std::cell::{Cell, RefCell};
use std::sync::{Arc, Mutex, RwLock};
use std::fmt;

#[cfg(target_arch = "wasm32")]
use std::rc::Rc;

use hal::{format, image as i, pass, pso};
use hal::memory::Properties;
use hal::backend::FastHashMap;

use gl;
use {Backend, GlBuffer, GlBufferOwned, GlProgram, GlProgramOwned, GlShaderOwned, GlTextureOwned, GlSamplerOwned};


pub type Program     = GlProgram;
pub type FrameBuffer = gl::types::GLuint;
pub type Surface     = gl::types::GLuint;

pub type DescriptorSetLayout = Vec<pso::DescriptorSetLayoutBinding>;

pub const DEFAULT_FRAMEBUFFER: FrameBuffer = 0;

#[derive(Debug)]
pub struct Buffer {
    pub(crate) raw: GlBufferOwned,
    pub(crate) target: gl::types::GLenum,
    pub(crate) size: u64,
}

#[derive(Debug)]
pub struct BufferView;

#[derive(Debug)]
pub struct Fence(pub(crate) Cell<gl::types::GLsync>);
unsafe impl Send for Fence {}
unsafe impl Sync for Fence {}

impl Fence {
    pub(crate) fn new(sync: gl::types::GLsync) -> Self {
        Fence(Cell::new(sync))
    }
}

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum BindingTypes {
    Images,
    UniformBuffers,
}

#[derive(Clone, Debug)]
pub struct DescRemapData {
    bindings: FastHashMap<(BindingTypes, pso::DescriptorSetIndex, pso::DescriptorBinding), Vec<pso::DescriptorBinding>>,
    names: FastHashMap<String, (BindingTypes, pso::DescriptorSetIndex, pso::DescriptorBinding)>,
    next_binding: FastHashMap<BindingTypes, pso::DescriptorBinding>,
}

/// Stores where the descriptor bindings have been remaped too.
///
/// OpenGL doesn't support sets, so we have to flatten out the bindings.
impl DescRemapData {
    pub fn new() -> Self {
        DescRemapData {
            bindings: FastHashMap::default(),
            names: FastHashMap::default(),
            next_binding: FastHashMap::default(),
        }
    }

    pub fn insert_missing_binding_into_spare(
        &mut self,
        btype: BindingTypes,
        set: pso::DescriptorSetIndex,
        binding: pso::DescriptorBinding,
    ) -> &[pso::DescriptorBinding] {
        let nb = self.next_binding.entry(btype).or_insert(0);
        let val = self.bindings.entry((btype, set, binding)).or_insert(Vec::new());
        val.push(*nb);
        *nb += 1;
        &*val
    }

    pub fn reserve_binding(&mut self, btype: BindingTypes) -> pso::DescriptorBinding {
        let nb = self.next_binding.entry(btype).or_insert(0);
        *nb += 1;
        *nb - 1
    }

    pub fn insert_missing_binding(
        &mut self,
        nb: pso::DescriptorBinding,
        btype: BindingTypes,
        set: pso::DescriptorSetIndex,
        binding: pso::DescriptorBinding,
    ) -> &[pso::DescriptorBinding] {
        let val = self.bindings.entry((btype, set, binding)).or_insert(Vec::new());
        val.push(nb);
        &*val
    }

    pub fn get_binding(
        &self,
        btype: BindingTypes,
        set: pso::DescriptorSetIndex,
        binding: pso::DescriptorBinding,
    ) -> Option<&[pso::DescriptorBinding]> {
        self.bindings.get(&(btype, set, binding)).map(AsRef::as_ref)
    }
}

#[derive(Clone, Debug)]
pub struct GraphicsPipeline {
    pub(crate) program: GlProgramOwned,
    pub(crate) primitive: gl::types::GLenum,
    pub(crate) patch_size: Option<gl::types::GLint>,
    pub(crate) blend_targets: Vec<pso::ColorBlendDesc>,
    pub(crate) attributes: Vec<AttributeDesc>,
    pub(crate) vertex_buffers: Vec<Option<pso::VertexBufferDesc>>,
}

#[derive(Clone, Debug)]
pub struct ComputePipeline {
    pub(crate) program: GlProgramOwned,
}

#[derive(Clone, Debug)]
pub struct Image {
    pub(crate) kind: ImageKind,
    // Required for clearing operations
    pub(crate) channel: format::ChannelType,
}

#[derive(Clone, Debug)]
pub enum ImageKind {
    Surface(Surface),
    Texture(GlTextureOwned),
}

#[derive(Clone, Debug)]
/// Additionally storing the `SamplerInfo` for older OpenGL versions, which
/// don't support separate sampler objects.
pub enum FatSampler {
    Sampler(GlSamplerOwned),
    Info(i::SamplerInfo),
}

#[derive(Clone, Debug)]
pub enum ImageView {
    Surface(Surface),
    Texture(GlTextureOwned, i::Level),
    TextureLayer(GlTextureOwned, i::Level, i::Layer),
}

#[derive(Clone, Debug)]
pub(crate) enum DescSetBindings {
    Buffer {
        ty: BindingTypes,
        binding: pso::DescriptorBinding,
        buffer: GlBufferOwned,
        offset: gl::types::GLintptr,
        size: gl::types::GLsizeiptr
    },
    Texture(pso::DescriptorBinding, GlTextureOwned),
    Sampler(pso::DescriptorBinding, GlSamplerOwned),
    SamplerInfo(pso::DescriptorBinding, i::SamplerInfo),
}

#[derive(Clone, Debug)]
pub struct DescriptorSet {
    layout: DescriptorSetLayout,
    pub(crate) bindings: Arc<Mutex<Vec<DescSetBindings>>>,
}

#[derive(Debug)]
pub struct DescriptorPool {}

impl pso::DescriptorPool<Backend> for DescriptorPool {
    fn allocate_set(&mut self, layout: &DescriptorSetLayout) -> Result<DescriptorSet, pso::AllocationError> {
        Ok(DescriptorSet {
            layout: layout.clone(),
            bindings: Arc::new(Mutex::new(Vec::new())),
        })
    }

    fn free_sets<I>(&mut self, _descriptor_sets: I)
    where
        I: IntoIterator<Item = DescriptorSet>
    {
        // Poof!  Does nothing, because OpenGL doesn't have a meaningful concept of a `DescriptorSet`.
    }

    fn reset(&mut self) {
        // Poof!  Does nothing, because OpenGL doesn't have a meaningful concept of a `DescriptorSet`.
    }
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Debug)]
pub enum ShaderModule {
    Raw(GlShaderOwned),
    Spirv(Vec<u8>),
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Debug, Hash)]
pub enum ShaderModule {
    Raw(GlShaderOwned),
    Spirv(Vec<u8>),
}

pub struct Memory {
    pub(crate) properties: Properties,
    pub(crate) first_bound_buffer: RefCell<Option<GlBufferOwned>>,
    /// Allocation size
    pub(crate) size: u64,
    #[cfg(target_arch = "wasm32")]
    pub(crate) mapped_memory: RefCell<Option<*mut u8>>,
}

// TODO
impl fmt::Debug for Memory {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Memory")
    }
}

unsafe impl Send for Memory {}
unsafe impl Sync for Memory {}

impl Memory {
    pub fn can_upload(&self) -> bool {
        self.properties.contains(Properties::CPU_VISIBLE)
    }

    pub fn can_download(&self) -> bool {
        self.properties.contains(Properties::CPU_VISIBLE | Properties::CPU_CACHED)
    }

    pub fn map_flags(&self) -> gl::types::GLenum {
        let mut flags = 0;
        if self.can_download() {
web_sys::console::log_1(&format!("map_flags() can download {:?}", gl::MAP_READ_BIT).into());
            flags |= gl::MAP_READ_BIT;
        }
        if self.can_upload() {
web_sys::console::log_1(&format!("map_flags() can upload {:?}", gl::MAP_WRITE_BIT).into());
            flags |= gl::MAP_WRITE_BIT;
        }
web_sys::console::log_1(&format!("map_flags() result {:?}", flags).into());
        flags
    }
}

#[derive(Clone, Debug)]
pub struct RenderPass {
    pub(crate) attachments: Vec<pass::Attachment>,
    pub(crate) subpasses: Vec<SubpassDesc>,
}

#[derive(Clone, Debug)]
pub struct SubpassDesc {
    pub(crate) color_attachments: Vec<usize>,
}

impl SubpassDesc {
    /// Check if an attachment is used by this sub-pass.
    pub(crate) fn is_using(&self, at_id: pass::AttachmentId) -> bool {
        self.color_attachments.iter()
            .any(|id| *id == at_id)
    }
}

#[derive(Debug)]
pub struct PipelineLayout {
    pub(crate) desc_remap_data: Arc<RwLock<DescRemapData>>,
}

#[derive(Debug)]
// No inter-queue synchronization required for GL.
pub struct Semaphore;

#[derive(Debug, Clone, Copy)]
pub struct AttributeDesc {
    pub(crate) location: gl::types::GLuint,
    pub(crate) offset: u32,
    pub(crate) binding: gl::types::GLuint,
    pub(crate) size: gl::types::GLint,
    pub(crate) format: gl::types::GLenum,
    pub(crate) vertex_attrib_fn: VertexAttribFunction,
}

#[derive(Debug, Clone, Copy)]
pub enum VertexAttribFunction {
    Float, // glVertexAttribPointer
    Integer, // glVertexAttribIPointer
    Double, // glVertexAttribLPointer
}
