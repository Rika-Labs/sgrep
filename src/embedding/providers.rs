use std::env;
use std::process::Command;

use ort::execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider, ExecutionProviderDispatch,
};

fn optimized_cpu_provider() -> ExecutionProviderDispatch {
    CPUExecutionProvider::default()
        .with_arena_allocator(true)
        .into()
}

pub fn select_execution_providers() -> Vec<ExecutionProviderDispatch> {
    configure_onnx_threading();

    let device = env::var("SGREP_DEVICE").unwrap_or_default().to_lowercase();

    match device.as_str() {
        "cpu" => vec![optimized_cpu_provider()],
        "coreml" => vec![
            CoreMLExecutionProvider::default().into(),
            optimized_cpu_provider(),
        ],
        "cuda" => vec![
            CUDAExecutionProvider::default().into(),
            optimized_cpu_provider(),
        ],
        _ => auto_detect_providers(),
    }
}

fn auto_detect_providers() -> Vec<ExecutionProviderDispatch> {
    if has_nvidia_gpu() {
        return vec![
            CUDAExecutionProvider::default().into(),
            optimized_cpu_provider(),
        ];
    }

    vec![optimized_cpu_provider()]
}

fn configure_onnx_threading() {
    let parallelism = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);

    let optimal_threads = (parallelism / 2).clamp(2, 8);

    if env::var_os("ORT_NUM_THREADS").is_none() {
        env::set_var("ORT_NUM_THREADS", optimal_threads.to_string());
    }

    if env::var_os("ORT_INTER_OP_NUM_THREADS").is_none() {
        env::set_var("ORT_INTER_OP_NUM_THREADS", "1");
    }
}

#[allow(dead_code)]
pub fn is_apple_silicon() -> bool {
    #[cfg(test)]
    if let Ok(val) = std::env::var("SGREP_TEST_APPLE") {
        return match val.as_str() {
            "1" => true,
            "0" => false,
            _ => cfg!(target_os = "macos") && cfg!(target_arch = "aarch64"),
        };
    }

    cfg!(target_os = "macos") && cfg!(target_arch = "aarch64")
}

fn has_nvidia_gpu() -> bool {
    #[cfg(test)]
    if std::env::var("SGREP_TEST_NVIDIA")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        return true;
    }

    Command::new("nvidia-smi")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn select_cpu_when_forced() {
        env::set_var("SGREP_DEVICE", "cpu");
        let eps = select_execution_providers();
        env::remove_var("SGREP_DEVICE");
        let joined = format!("{:?}", eps);
        assert!(joined.contains("CPUExecutionProvider"));
        assert_eq!(eps.len(), 1);
    }

    #[test]
    #[serial]
    fn select_coreml_when_forced() {
        env::set_var("SGREP_DEVICE", "coreml");
        let eps = select_execution_providers();
        env::remove_var("SGREP_DEVICE");
        let joined = format!("{:?}", eps);
        assert!(joined.contains("CoreMLExecutionProvider"));
        assert!(joined.contains("CPUExecutionProvider"));
    }

    #[test]
    #[serial]
    fn select_cuda_when_forced() {
        env::set_var("SGREP_DEVICE", "cuda");
        let eps = select_execution_providers();
        env::remove_var("SGREP_DEVICE");
        let joined = format!("{:?}", eps);
        assert!(joined.contains("CUDAExecutionProvider"));
        assert!(joined.contains("CPUExecutionProvider"));
    }

    #[test]
    #[serial]
    fn select_execution_providers_prefers_test_gpu_override() {
        env::remove_var("SGREP_DEVICE");
        env::set_var("SGREP_TEST_APPLE", "0");
        env::set_var("SGREP_TEST_NVIDIA", "1");
        let eps = select_execution_providers();
        env::remove_var("SGREP_TEST_NVIDIA");
        env::remove_var("SGREP_TEST_APPLE");
        let joined = format!("{:?}", eps);
        assert!(joined.contains("CUDAExecutionProvider"));
        assert!(joined.contains("CPUExecutionProvider"));
    }

    #[test]
    #[serial]
    fn select_execution_providers_defaults_to_cpu() {
        env::remove_var("SGREP_DEVICE");
        env::set_var("SGREP_TEST_APPLE", "1");
        env::set_var("SGREP_TEST_NVIDIA", "0");
        let eps = select_execution_providers();
        env::remove_var("SGREP_TEST_APPLE");
        env::remove_var("SGREP_TEST_NVIDIA");
        let joined = format!("{:?}", eps);
        assert!(joined.contains("CPUExecutionProvider"));
        assert!(!joined.contains("CoreMLExecutionProvider"));
    }
}
