use pbrt_rs;
use tempfile;
use env_logger;
use std::io::prelude::*;

fn create_file_and_parse(content: &str) -> pbrt_rs::Scene {
    let mut file = tempfile::NamedTempFile::new().expect("Impossible to create tempdir");
    file.write_all(content.as_bytes()).expect("Impossible to write tempfile content");
    
    let mut scene_info = pbrt_rs::Scene::default();
    let mut state = pbrt_rs::State::default();
    
    let path = file.path();
    let working_dir = path.parent().unwrap();

    env_logger::Builder::from_default_env()
            .format_timestamp(None)
            .parse_filters("info")
            .is_test(true)
            .init();

    pbrt_rs::read_pbrt_file(path.to_str().unwrap(), &working_dir, &mut scene_info, &mut state);

    scene_info
}

#[test]
fn sphere() {
    let scene = create_file_and_parse(r#"
        Shape "sphere" "float radius" [0.25]
    "#);

    // FIXME: Write acutal test
}

#[test]
fn multiline() {
    let scene = create_file_and_parse(r#"
    ### Object: Puerta Arco Grande Vidrio ### 
    MakeNamedMaterial "Vidrio" 
             "string type" ["glass"] 
             "color Kr" [1.0 1.0 1.0] "color Kt" [1.0 1.0 1.0] "float index" [1.5] 
    
    ### End Object: Puerta Arco Grande Vidrio ### 
    
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
    "#);

}

