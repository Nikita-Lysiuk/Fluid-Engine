use std::sync::Arc;
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DispatchIndirectCommand};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};

/// A prepared compute pipeline and its descriptor set.
pub struct ComputePass {
    pipeline: Arc<ComputePipeline>,
    descriptor_set: Arc<DescriptorSet>,
    group_count: u32,
}

impl ComputePass {
    /// Prepares a pipeline with the descriptor bindings for its first set.
    pub fn new(
        pipeline: Arc<ComputePipeline>,
        allocator: Arc<StandardDescriptorSetAllocator>,
        bindings: impl IntoIterator<Item = WriteDescriptorSet>,
        group_count: u32,
    ) -> Self {
        let layout = pipeline.layout().set_layouts().first().expect("compute descriptor set layout 0 exists");
        let descriptor_set = DescriptorSet::new(allocator, layout.clone(), bindings, [])
            .expect("compute descriptor set bindings match the shader");
        Self { pipeline, descriptor_set, group_count }
    }

    /// Records the bound compute dispatch.
    pub fn bind<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>) {
        builder.bind_pipeline_compute(self.pipeline.clone()).expect("compute pipeline can be bound")
            .bind_descriptor_sets(PipelineBindPoint::Compute, self.pipeline.layout().clone(), 0, self.descriptor_set.clone())
            .expect("compute descriptor set can be bound");
    }

    /// Binds the pass and its push constants.
    pub fn bind_with_constants<Cb, Pc: BufferContents>(&self, builder: &mut AutoCommandBufferBuilder<Cb>, constants: Pc) {
        self.bind(builder);
        builder.push_constants(self.pipeline.layout().clone(), 0, constants).expect("push constants match the shader");
    }

    /// Dispatches after binding the pass and its push constants.
    pub fn dispatch_bound<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>) {
        // SAFETY: the caller has bound the pipeline, descriptor set, and required push constants.
        unsafe { builder.dispatch([self.group_count, 1, 1]).expect("compute dispatch is valid"); }
    }

    /// Dispatches indirectly after binding the pass and its push constants.
    pub fn dispatch_indirect_bound<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>, args: Subbuffer<[DispatchIndirectCommand]>) {
        // SAFETY: the caller bound the pass and constants, and the solver writes valid workgroup counts.
        unsafe { builder.dispatch_indirect(args).expect("indirect compute dispatch is valid"); }
    }

    /// Records a direct dispatch.
    pub fn dispatch<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>) {
        self.bind(builder);
        // SAFETY: the pipeline and all descriptor sets required by these shaders are bound above.
        unsafe { builder.dispatch([self.group_count, 1, 1]).expect("compute dispatch is valid"); }
    }

    /// Records a dispatch whose workgroup count is provided by the solver.
    pub fn dispatch_indirect<Cb>(&self, builder: &mut AutoCommandBufferBuilder<Cb>, args: Subbuffer<[DispatchIndirectCommand]>) {
        self.bind(builder);
        // SAFETY: the bound pipeline and descriptors match, and the solver writes a valid workgroup count.
        unsafe { builder.dispatch_indirect(args).expect("indirect compute dispatch is valid"); }
    }
}
