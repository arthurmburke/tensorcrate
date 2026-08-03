use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn main() {
    println!("cargo:rerun-if-changed=metal/kernel.metal");
    println!("cargo:rerun-if-changed=metal/tensorops.metal");
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");

    if env::var_os("CARGO_FEATURE_METAL").is_none()
        || env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos")
    {
        return;
    }

    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let source = manifest_dir.join("metal/kernel.metal");
    let tensorops_source = manifest_dir.join("metal/tensorops.metal");
    let air = out_dir.join("tensorcrate.air");
    let tensorops_air = out_dir.join("tensorcrate_tensorops.air");
    let library = out_dir.join("tensorcrate.metallib");
    let tensorops_library = out_dir.join("tensorcrate_tensorops.metallib");

    run_xcrun("metal", &["-c", path(&source), "-o", path(&air)]);
    run_xcrun("metallib", &[path(&air), "-o", path(&library)]);
    run_xcrun(
        "metal",
        &[
            "-std=metal4.0",
            "-c",
            path(&tensorops_source),
            "-o",
            path(&tensorops_air),
        ],
    );
    run_xcrun(
        "metallib",
        &[path(&tensorops_air), "-o", path(&tensorops_library)],
    );
}

fn path(path: &Path) -> &str {
    path.to_str()
        .unwrap_or_else(|| panic!("Metal build path is not valid UTF-8: {}", path.display()))
}

fn run_xcrun(tool: &str, args: &[&str]) {
    let output = Command::new("xcrun")
        .args(["--sdk", "macosx", tool])
        .args(args)
        .output()
        .unwrap_or_else(|error| panic!("failed to run `xcrun {tool}`: {error}"));

    if !output.status.success() {
        fail(tool, output);
    }
}

fn fail(tool: &str, output: Output) -> ! {
    panic!(
        "Metal {tool} step failed with {}\n\
         Install the Metal toolchain with full Xcode, then select it with \
         `xcode-select` or set `DEVELOPER_DIR`.\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    )
}
