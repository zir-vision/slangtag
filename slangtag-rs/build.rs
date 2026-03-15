use std::env;
use std::ffi::OsStr;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is not set by Cargo"),
    );
    let shader_root = manifest_dir.join("shaders");
    let compute_dir = shader_root.join("compute");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is not set by Cargo"))
        .join("shaders")
        .join("compute");

    println!("cargo:rerun-if-env-changed=SLANGC");
    println!("cargo:rerun-if-changed={}", shader_root.display());

    fs::create_dir_all(&out_dir).expect("failed to create output shader directory");

    let slang_sources = collect_slang_sources(&compute_dir);
    if slang_sources.is_empty() {
        return;
    }

    let slangc = env::var("SLANGC").unwrap_or_else(|_| "slangc".to_string());

    for source in slang_sources {
        println!("cargo:rerun-if-changed={}", source.display());

        let source_text = fs::read_to_string(&source).unwrap_or_else(|err| {
            panic!("failed to read shader source {}: {err}", source.display());
        });

        // Support helper modules like `image_buffer.slang` that have no shader entry point.
        if !source_text.contains("[shader(\"compute\")]") {
            continue;
        }

        let relative = source.strip_prefix(&compute_dir).unwrap_or_else(|err| {
            panic!(
                "failed to strip compute dir prefix from {}: {err}",
                source.display()
            );
        });

        let mut output = out_dir.join(relative);
        output.set_extension("spv");

        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent).unwrap_or_else(|err| {
                panic!("failed to create output dir {}: {err}", parent.display());
            });
        }

        let status = Command::new(&slangc)
            .arg(&source)
            .arg("-g3")
            .arg("-entry")
            .arg("main")
            .arg("-stage")
            .arg("compute")
            .arg("-target")
            .arg("spirv")
            .arg("-I")
            .arg(&compute_dir)
            .arg("-I")
            .arg(&shader_root)
            .arg("-o")
            .arg(&output)
            .status();

        match status {
            Ok(status) if status.success() => {}
            Ok(_) => {
                panic!(
                    "slang shader compilation failed for {} (output: {})",
                    source.display(),
                    output.display()
                );
            }
            Err(err) if err.kind() == ErrorKind::NotFound => {
                println!(
                    "cargo:warning=`{slangc}` was not found; writing placeholder SPIR-V for {}",
                    source.display()
                );
                write_placeholder_spirv(&output);
            }
            Err(err) => {
                panic!("failed to run `{slangc}`: {err}");
            }
        }
    }
}

fn write_placeholder_spirv(output: &Path) {
    let placeholder: [u32; 5] = [0x0723_0203, 0x0001_0000, 0, 0, 0];
    let mut bytes = Vec::with_capacity(placeholder.len() * std::mem::size_of::<u32>());
    for word in placeholder {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    fs::write(output, bytes).unwrap_or_else(|err| {
        panic!(
            "failed to write placeholder SPIR-V {}: {err}",
            output.display()
        );
    });
}

fn collect_slang_sources(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    visit_dir(root, &mut files);
    files.sort();
    files
}

fn visit_dir(dir: &Path, files: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir).unwrap_or_else(|err| {
        panic!("failed to read shader directory {}: {err}", dir.display());
    });

    for entry in entries {
        let entry = entry.unwrap_or_else(|err| {
            panic!("failed to read entry in {}: {err}", dir.display());
        });
        let path = entry.path();
        if path.is_dir() {
            visit_dir(&path, files);
            continue;
        }
        if path.extension() == Some(OsStr::new("slang")) {
            files.push(path);
        }
    }
}
