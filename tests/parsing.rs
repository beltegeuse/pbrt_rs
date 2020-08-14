use pbrt_rs;
use std::io::prelude::*;
use tempfile;

fn create_file_and_parse(content: &str) -> pbrt_rs::Scene {
    let mut file = tempfile::NamedTempFile::new().expect("Impossible to create tempdir");
    file.write_all(content.as_bytes())
        .expect("Impossible to write tempfile content");

    let mut scene_info = pbrt_rs::Scene::default();
    let mut state = pbrt_rs::State::default();

    let path = file.path();
    let working_dir = path.parent().unwrap();

    pbrt_rs::read_pbrt_file(
        path.to_str().unwrap(),
        Some(&working_dir),
        &mut scene_info,
        &mut state,
    );

    scene_info
}

#[test]
fn sphere() {
    let scene = create_file_and_parse(
        r#"
        Shape "sphere" "float radius" [0.25]
    "#,
    );

    // FIXME: Write acutal test
    //  by testing the right parsing
}

#[test]
fn name_with_sharp() {
    let scene_with_sharp = create_file_and_parse(
        r#"   
    ### Object: Jardinera 1 ### 
    Texture "Map #594" "color" "imagemap"
             "string mapping" "uv"
             "string wrap" ["repeat"] 
             "string filename" ["textures/jardinera_1_color.png"]
             "string mapping" ["uv"]
             "float uscale" [1.0]
             "float vscale" [1.0]
             "float udelta" [0.0]
             "float vdelta" [0.0]
             "float maxanisotropy" [8.0]    
    "#,
    );
}
