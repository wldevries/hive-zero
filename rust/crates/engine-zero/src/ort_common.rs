//! Shared ONNX Runtime / QNN setup helpers used by all game-specific
//! inference engines. Path probing is hard-coded for the project's `.venv`
//! layout — see `find_onnxruntime_dylib` and friends.

use std::path::PathBuf;

/// Ensure the directory containing QnnHtp.dll is on PATH so ORT can find
/// onnxruntime_providers_qnn.dll. Needed for the 2.x unbundled layout where
/// QNN DLLs live in a separate onnxruntime_qnn/ directory.
pub fn prepend_qnn_dir_to_path() {
    let htp = find_qnn_htp_dll();
    if let Some(dir) = std::path::Path::new(&htp).parent() {
        let dir_str = dir.to_string_lossy();
        let current = std::env::var("PATH").unwrap_or_default();
        if !current.contains(dir_str.as_ref()) {
            // SAFETY: single-threaded startup, before any threads are spawned
            unsafe { std::env::set_var("PATH", format!("{};{}", dir_str, current)); }
        }
    }
}

pub fn find_onnxruntime_dylib() -> PathBuf {
    PathBuf::from(r".venv\Lib\site-packages\onnxruntime\capi\onnxruntime.dll")
}

pub fn find_qnn_htp_dll() -> String {
    // onnxruntime-qnn <=1.x bundles DLLs into onnxruntime/capi/ (alongside onnxruntime.dll).
    // onnxruntime-qnn 2.x+ puts them in a separate onnxruntime_qnn/ directory.
    let bundled = std::path::Path::new(r".venv\Lib\site-packages\onnxruntime\capi\QnnHtp.dll");
    let unbundled = std::path::Path::new(r".venv\Lib\site-packages\onnxruntime_qnn\QnnHtp.dll");
    let p = if bundled.exists() { bundled } else { unbundled };
    p.canonicalize()
        .unwrap_or_else(|_| p.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

/// onnxruntime-qnn 2.x ships the QNN EP as a standalone plugin library that must be registered
/// with the ORT environment before sessions can use it.
pub fn find_qnn_provider_dll() -> PathBuf {
    let p = std::path::Path::new(r".venv\Lib\site-packages\onnxruntime_qnn\onnxruntime_providers_qnn.dll");
    p.canonicalize().unwrap_or_else(|_| p.to_path_buf())
}

/// Register the QNN plugin EP library with the current ORT environment (onnxruntime-qnn ≥ 2.0).
/// Silently skips if the plugin DLL is absent or registration fails (e.g. no QNN hardware).
pub fn register_qnn_plugin() {
    let provider_dll = find_qnn_provider_dll();
    if !provider_dll.exists() {
        return;
    }
    if let Ok(env) = ort::environment::Environment::current() {
        let _ = env.register_ep_library("QNN", provider_dll);
    }
}
