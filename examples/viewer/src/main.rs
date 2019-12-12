#![allow(dead_code)]

extern crate cgmath;
extern crate clap;
extern crate env_logger;
extern crate pbrt_rs;
#[macro_use]
extern crate log;

fn main() {
	let scene_path_str = "/home/muliana/projects/pbrt_rs/data/pbrt_rs_scenes/Spaceship/scene.pbrt";

	let mut scene_info = pbrt_rs::Scene::default();
    let mut state = pbrt_rs::State::default();
    let working_dir = std::path::Path::new(scene_path_str).parent().unwrap();
	pbrt_rs::read_pbrt_file(scene_path_str, &working_dir, &mut scene_info, &mut state);
	
	// Print statistics
	info!("Scenes info: ");
	info!(" - BSDFS: {}", scene_info.materials.len());
	info!(" - Objects: ");
	info!("    * Unamed object: {}", scene_info.shapes.len());
	info!("    * Object: {}", scene_info.objects.len());
	info!("    * Object's instance: {}", scene_info.instances.len()); 
}