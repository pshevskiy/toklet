use std::process::{Command, Output};

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_toklet"))
        .args(args)
        .output()
        .expect("Toklet CLI should run")
}

#[test]
fn help_is_available_without_a_model() {
    for args in [&[][..], &["--help"], &["-h"]] {
        let output = run(args);
        assert!(output.status.success());
        assert!(output.stderr.is_empty());
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(stdout.contains("Usage: toklet [OPTIONS]"));
        assert!(stdout.contains("not implemented yet"));
    }
}

#[test]
fn version_matches_the_package() {
    for arg in ["--version", "-V"] {
        let output = run(&[arg]);
        assert!(output.status.success());
        assert!(output.stderr.is_empty());
        assert_eq!(
            output.stdout,
            format!("toklet {}\n", env!("CARGO_PKG_VERSION")).as_bytes()
        );
    }
}

#[test]
fn invalid_arguments_fail_without_success_output() {
    for args in [
        &["generate"][..],
        &["--help", "extra"],
        &["--version", "extra"],
    ] {
        let output = run(args);
        assert_eq!(output.status.code(), Some(2));
        assert!(output.stdout.is_empty());
        assert!(String::from_utf8(output.stderr).unwrap().contains("error:"));
    }
}

#[cfg(unix)]
#[test]
fn non_unicode_arguments_return_an_error_instead_of_panicking() {
    use std::{ffi::OsString, os::unix::ffi::OsStringExt};

    let output = Command::new(env!("CARGO_BIN_EXE_toklet"))
        .arg(OsString::from_vec(vec![0xff]))
        .output()
        .expect("Toklet CLI should run");
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8(output.stderr).unwrap().contains("error:"));
}
