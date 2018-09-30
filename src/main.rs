extern crate clap;
extern crate env_logger;
extern crate pbrt_rs;
#[macro_use]
extern crate log;
use clap::{App, Arg};

fn main() {
    let matches = App::new("scene_parsing")
        .version("0.0.1")
        .author("Adrien Gruson <adrien.gruson@gmail.com>")
        .about("A Rusty scene 3D parsing for rendering system")
        .arg(
            Arg::with_name("scene")
                .required(true)
                .takes_value(true)
                .index(1)
                .help("3D scene"),
        ).arg(Arg::with_name("debug").short("d").help("debug output"))
        .get_matches();
    let scene_path_str = matches
        .value_of("scene")
        .expect("no scene parameter provided");
    if matches.is_present("debug") {
        // FIXME: add debug flag?
        env_logger::Builder::from_default_env()
            .default_format_timestamp(false)
            .parse("debug")
            .init();
    } else {
        env_logger::Builder::from_default_env()
            .default_format_timestamp(false)
            .parse("info")
            .init();
    }

    // The parsing
    let mut scene_info = pbrt_rs::Scene::default();
    let working_dir = std::path::Path::new(scene_path_str.clone())
        .parent()
        .unwrap();
    pbrt_rs::read_pbrt_file(
        scene_path_str,
        &working_dir,
        &mut scene_info,
        pbrt_rs::State::default(),
    );

    // Print statistics
    info!("Scenes info: ");
    info!(" - BSDFS: {}", scene_info.materials.len());
    info!(" - Objects: ");
    info!("    * Unamed object: {}", scene_info.shapes.len());
    info!("    * Object: {}", scene_info.objects.len());
    info!("    * Object's instance: {}", scene_info.instances.len());

    let tri_sum: usize = scene_info
        .shapes
        .iter()
        .map(|v| match v.data {
            pbrt_rs::Shape::TriMesh(ref v) => v.points.len(),
            _ => 0,
        }).sum();
    let indices_sum: usize = scene_info
        .shapes
        .iter()
        .map(|v| match v.data {
            pbrt_rs::Shape::TriMesh(ref v) => v.indices.len() / 3,
            _ => 0,
        }).sum();
    info!("Total: ");
    info!(" - #triangles: {}", tri_sum);
    info!(" - #indices: {}", indices_sum);
}
