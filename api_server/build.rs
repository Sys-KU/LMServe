use std::fs;

fn main() {
    fs::create_dir_all("src/pb")
        .unwrap_or_else(|e| panic!("Failed to create directory for pb files {:?}", e));

    tonic_build::configure()
        .protoc_arg("--experimental_allow_proto3_optional")
        .build_client(true)
        .build_server(true)
        .out_dir("src/pb")
        .include_file("mod.rs")
        .compile_protos(&["../proto/llm.proto"], &["../proto"])
        .unwrap_or_else(|e| panic!("Failed to compile protos {:?}", e));
}
