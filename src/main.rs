extern crate cgmath;
extern crate clap;
extern crate env_logger;
extern crate pbrt_rs;
#[macro_use]
extern crate log;
use cgmath::*;
use clap::{App, Arg};
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

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
        )
        .arg(Arg::with_name("debug").short("d").help("debug output"))
        .arg(
            Arg::with_name("obj")
                .required(false)
                .takes_value(true)
                .short("o")
                .help("Output obj file"),
        )
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
    let mut state = pbrt_rs::State::default();
    let working_dir = std::path::Path::new(scene_path_str.clone())
        .parent()
        .unwrap();
    pbrt_rs::read_pbrt_file(scene_path_str, &working_dir, &mut scene_info, &mut state);

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
        })
        .sum();
    let indices_sum: usize = scene_info
        .shapes
        .iter()
        .map(|v| match v.data {
            pbrt_rs::Shape::TriMesh(ref v) => v.indices.len() / 3,
            _ => 0,
        })
        .sum();
    info!("Total: ");
    info!(" - #triangles: {}", tri_sum);
    info!(" - #indices: {}", indices_sum);

    // Camera information
    for cam in &scene_info.cameras {
        info!("Camera information: ");
        match cam {
            pbrt_rs::Camera::Perspective(ref c) => {
                info!(" - fov: {}", c.fov);
                info!(" - world_to_camera: {:?}", c.world_to_camera);
            }
        }
    }

    if let Some(obj_path) = matches.value_of("obj") {
        info!("Export in OBJ: {}", obj_path);

        let mut file = File::create(Path::new(obj_path)).unwrap();
        file.write(b"# OBJ EXPORTED USING pbrt_rs\n").unwrap();

        // Need to write manually the obj file
        // --- Write all uname shapes
        let mut offset_point = 1;
        let mut offset_normal = 1;
        let mut offset_uv = 1;
        for (i, shape) in scene_info.shapes.iter().enumerate() {
            match shape.data {
                pbrt_rs::Shape::TriMesh(ref data) => {
                    // Load the relevent data and make the transformation
                    let mat = shape.matrix;
                    let uv = if let Some(uv) = data.uv.clone() {
                        uv
                    } else {
                        vec![]
                    };
                    let normals = match data.normals {
                        Some(ref v) => v.iter().map(|n| mat.transform_vector(n.clone())).collect(),
                        None => Vec::new(),
                    };
                    let points = data
                        .points
                        .iter()
                        .map(|n| mat.transform_point(n.clone()))
                        .collect::<Vec<Point3<f32>>>();

                    // We only support trianbles, so it is much easier
                    // Moreover, normal, uv are aligned
                    writeln!(file, "o Unamed_{}", i).unwrap();
                    // --- Geometry
                    let mut number_channels = 1;
                    for p in &points {
                        writeln!(file, "v {} {} {}", p.x, p.y, p.z).unwrap();
                    }
                    file.write(b"\n").unwrap();
                    if !uv.is_empty() {
                        number_channels += 1;
                        for t in &uv {
                            writeln!(file, "vt {} {}", t.x, t.y).unwrap();
                        }
                        file.write(b"\n").unwrap();
                    }
                    if !normals.is_empty() {
                        number_channels += 1;
                        for n in &normals {
                            writeln!(file, "vn {} {} {}", n.x, n.y, n.z).unwrap();
                        }
                        file.write(b"\n").unwrap();
                    }

                    // --- Indicies
                    for index in data.indices.chunks(3) {
                        let i1 = index[0] as usize;
                        let i2 = index[1] as usize;
                        let i3 = index[2] as usize;

                        match number_channels {
                            1 => writeln!(
                                file,
                                "f {} {} {}",
                                i1 + offset_point,
                                i2 + offset_point,
                                i3 + offset_point
                            )
                            .unwrap(),
                            2 => {
                                if normals.is_empty() {
                                    writeln!(
                                        file,
                                        "f {}/{} {}/{} {}/{}",
                                        i1 + offset_point,
                                        i1 + offset_uv,
                                        i2 + offset_point,
                                        i2 + offset_uv,
                                        i3 + offset_point,
                                        i3 + offset_uv
                                    )
                                    .unwrap();
                                } else {
                                    writeln!(
                                        file,
                                        "f {}//{} {}//{} {}//{}",
                                        i1 + offset_point,
                                        i1 + offset_normal,
                                        i2 + offset_point,
                                        i2 + offset_normal,
                                        i3 + offset_point,
                                        i3 + offset_normal
                                    )
                                    .unwrap();
                                }
                            }
                            3 => writeln!(
                                file,
                                "f {}/{}/{} {}/{}/{} {}/{}/{}",
                                i1 + offset_point,
                                i1 + offset_uv,
                                i1 + offset_normal,
                                i2 + offset_point,
                                i2 + offset_uv,
                                i2 + offset_normal,
                                i3 + offset_point,
                                i3 + offset_uv,
                                i3 + offset_normal
                            )
                            .unwrap(),
                            _ => panic!("Unsupported number of channels"),
                        }
                    }
                    file.write(b"\n").unwrap();
                    offset_point += points.len();
                    offset_normal += normals.len();
                    offset_uv += uv.len();
                }
                _ => {
                    panic!("Ignore the type of mesh");
                }
            }
        }
    }
}
