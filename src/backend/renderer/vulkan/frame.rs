#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum VulkanFrameState {
    #[default]
    Idle,
    Recording,
    Finished,
}

#[derive(Debug, Default)]
pub(crate) struct FrameStateMachine {
    pub(crate) state: VulkanFrameState,
}

impl FrameStateMachine {
    pub(crate) fn begin(&mut self) {
        self.state = VulkanFrameState::Recording;
    }

    pub(crate) fn finish(&mut self) {
        self.state = VulkanFrameState::Finished;
    }

    pub(crate) fn reset(&mut self) {
        self.state = VulkanFrameState::Idle;
    }
}
