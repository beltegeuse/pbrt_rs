#![allow(clippy::cognitive_complexity)]

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
use std::io::BufWriter;
use std::path::Path;

/*
 Export PBRT files to Obj
*/
fn export_obj(scene_info: pbrt_rs::Scene, file: &mut File, mat_file: &mut File) {
    let mut file = BufWriter::new(file);
    let mut mat_file = BufWriter::new(mat_file);

    let normalize_rgb = |rgb: &mut pbrt_rs::parser::RGB| {
        let max = rgb.r.max(rgb.b.max(rgb.g));
        if max > 1.0 {
            rgb.r /= max;
            rgb.g /= max;
            rgb.b /= max;
        }
    };

    let default_mat = |f: &mut BufWriter<&mut File>| {
        writeln!(f, "Ns 1.0").unwrap();
        writeln!(f, "Ka 1.000000 1.000000 1.000000").unwrap();
        writeln!(f, "Kd 0.8 0.8 0.8").unwrap();
        writeln!(f, "Ke 0.000000 0.000000 0.000000").unwrap();
        writeln!(f, "Ni 1.000000").unwrap();
        writeln!(f, "d 1.000000").unwrap();
        writeln!(f, "illum 1").unwrap();
    };
    let emission_mat = |id_light: u32,
                        shape_name: String,
                        shape_emission: &Option<pbrt_rs::parser::Spectrum>,
                        f_obj: &mut BufWriter<&mut File>,
                        f_mat: &mut BufWriter<&mut File>| {
        info!("Exporting emission:");
        info!(" - shape_name: {}", shape_name);

        match shape_emission {
            Some(pbrt_rs::parser::Spectrum::RGB(ref rgb)) => {
                info!(" - emission: [{}, {}, {}]", rgb.r, rgb.g, rgb.b);
                writeln!(f_obj, "usemtl light_{}", id_light).unwrap();
                // Write the material file because the light is special materials
                writeln!(f_mat, "newmtl light_{}", id_light).unwrap();
                writeln!(f_mat, "Ns 0.0").unwrap();
                writeln!(f_mat, "Ka 0.000000 0.000000 0.000000").unwrap();
                writeln!(f_mat, "Kd 0.0 0.0 0.0").unwrap();
                writeln!(f_mat, "Ke {} {} {}", rgb.r, rgb.g, rgb.b).unwrap();
                writeln!(f_mat, "Ni 0.000000").unwrap();
                writeln!(f_mat, "d 1.000000").unwrap();
                writeln!(f_mat, "illum 7").unwrap();
                f_mat.write_all(b"\n").unwrap();
            }
            _ => panic!("No support for this emission profile"),
        }
    };

    {
        // Write default material
        writeln!(mat_file, "newmtl export_default").unwrap();
        default_mat(&mut mat_file);
        mat_file.write_all(b"\n").unwrap();
    }

    // Need to write manually the obj file
    // --- Write all uname shapes
    let mut offset_point = 1;
    let mut offset_normal = 1;
    let mut offset_uv = 1;
    let mut nb_light = 0;
    for (i, shape) in scene_info.shapes.into_iter().enumerate() {
        let material_name = shape.material_name.clone();
        let shape_emission = shape.emission;
        match shape.data {
            pbrt_rs::Shape::TriMesh {
                indices,
                points,
                uv,
                normals,
            } => {
                // Load the relevent data and make the transformation
                let mat = shape.matrix;
                let uv = if let Some(uv) = uv { uv } else { vec![] };
                let normals = match normals {
                    Some(ref v) => v
                        .into_iter()
                        .map(|n| mat.transform_vector(n.clone()))
                        .collect(),
                    None => Vec::new(),
                };
                let points = points
                    .into_iter()
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
                file.write_all(b"\n").unwrap();
                if !uv.is_empty() {
                    number_channels += 1;
                    for t in &uv {
                        writeln!(file, "vt {} {}", t.x, t.y).unwrap();
                    }
                    file.write_all(b"\n").unwrap();
                }
                if !normals.is_empty() {
                    number_channels += 1;
                    for n in &normals {
                        writeln!(file, "vn {} {} {}", n.x, n.y, n.z).unwrap();
                    }
                    file.write_all(b"\n").unwrap();
                }

                // --- Write material
                match material_name {
                    None => {
                        if shape_emission.is_none() {
                            writeln!(file, "usemtl export_default").unwrap();
                        } else {
                            emission_mat(
                                nb_light,
                                format!("Unamed_{}", i),
                                &shape_emission,
                                &mut file,
                                &mut mat_file,
                            );
                            nb_light += 1;
                        }
                    }
                    Some(ref m) => {
                        if shape_emission.is_none() {
                            writeln!(file, "usemtl {}", m).unwrap();
                        } else {
                            warn!("Overwrite materials as it is a light");
                            emission_mat(
                                nb_light,
                                format!("Unamed_{}", i),
                                &shape_emission,
                                &mut file,
                                &mut mat_file,
                            );
                            nb_light += 1;
                        }
                    }
                };
                for index in indices {
                    let i1 = index.x;
                    let i2 = index.y;
                    let i3 = index.z;

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
                file.write_all(b"\n").unwrap();
                offset_point += points.len();
                offset_normal += normals.len();
                offset_uv += uv.len();
            }
            _ => panic!("All meshes need to be converted to trimesh!"),
        }
    } // End shapes

    // Export the materials
    let mut textures = vec![];
    info!("Exporting bsdfs...");
    for (name, bdsf) in scene_info.materials.iter() {
        info!(" - {}", name);
        writeln!(mat_file, "newmtl {}", name).unwrap();
        match bdsf {
            pbrt_rs::BSDF::Matte { kd, .. } => {
                writeln!(mat_file, "Ns 1.0").unwrap();
                writeln!(mat_file, "Ka 0.0 0.0 0.0").unwrap();
                writeln!(mat_file, "Tf 1.0 1.0 1.0").unwrap();
                writeln!(mat_file, "Ks 0.0 0.0 0.0").unwrap();
                writeln!(mat_file, "illum 4").unwrap();
                match kd {
                    pbrt_rs::parser::Spectrum::RGB(ref rgb) => {
                        let mut rgb = rgb.clone();
                        normalize_rgb(&mut rgb);
                        writeln!(mat_file, "Kd {} {} {}", rgb.r, rgb.g, rgb.b).unwrap()
                    }
                    pbrt_rs::parser::Spectrum::Texture(ref tex_name) => {
                        writeln!(mat_file, "Kd 0.0 0.0 0.0").unwrap();
                        let texture = &scene_info.textures[tex_name];
                        warn!(" - Texture file: {}", texture.filename);
                        writeln!(mat_file, "map_Kd {}", texture.filename).unwrap();
                        textures.push(texture.filename.clone());
                    }
                    _ => panic!("Unsupported texture for matte material"),
                }
            }
            pbrt_rs::BSDF::Glass { .. } => {
                writeln!(mat_file, "Ns 1000").unwrap();
                writeln!(mat_file, "Ka 0.0 0.0 0.0").unwrap();
                writeln!(mat_file, "Kd 0.0 0.0 0.0").unwrap();
                writeln!(mat_file, "Tf 0.1 0.1 0.1").unwrap();
                writeln!(mat_file, "Ks 0.5 0.5 0.5").unwrap();
                writeln!(mat_file, "Ni 1.31").unwrap(); // Glass
                writeln!(mat_file, "d 1.000000").unwrap();
                writeln!(mat_file, "illum 7").unwrap();
                // TODO: Read the properties
            }
            pbrt_rs::BSDF::Mirror { kr, .. } => {
                writeln!(mat_file, "Ns 100000.0").unwrap();
                writeln!(mat_file, "Ka 0.0 0.0 0.0").unwrap();
                writeln!(mat_file, "Kd 0.0 0.0 0.0").unwrap();
                writeln!(mat_file, "Tf 1.0 1.0 1.0").unwrap();
                writeln!(mat_file, "Ni 1.00").unwrap();
                writeln!(mat_file, "illum 3").unwrap();
                match kr {
                    pbrt_rs::parser::Spectrum::RGB(ref rgb) => {
                        let mut rgb = rgb.clone();
                        normalize_rgb(&mut rgb);
                        writeln!(mat_file, "Ks {} {} {}", rgb.r, rgb.g, rgb.b).unwrap()
                    }
                    _ => panic!("Unsupported texture for mirror material"),
                }
            }
            pbrt_rs::BSDF::Substrate { ks, kd, .. } => {
                writeln!(mat_file, "Ka 0.0 0.0 0.0").unwrap();
                writeln!(mat_file, "Tf 1.0 1.0 1.0").unwrap();
                writeln!(mat_file, "Ni 1.0").unwrap();
                writeln!(mat_file, "illum 4").unwrap();
                match ks {
                    pbrt_rs::parser::Spectrum::RGB(ref rgb) => {
                        let mut rgb = rgb.clone();
                        normalize_rgb(&mut rgb);
                        writeln!(mat_file, "Ks {} {} {}", rgb.r, rgb.g, rgb.b).unwrap()
                    }
                    pbrt_rs::parser::Spectrum::Texture(ref tex_name) => {
                        writeln!(mat_file, "Ks 0.0 0.0 0.0").unwrap();
                        let texture = &scene_info.textures[tex_name];
                        warn!(" - Texture file: {}", texture.filename);
                        writeln!(mat_file, "map_Ks {}", texture.filename).unwrap();
                        textures.push(texture.filename.clone());
                    }
                    _ => panic!("Unsupported texture for metal material"),
                }
                warn!("Rougness conversion is broken");
                writeln!(mat_file, "Ns {}", 0.1).unwrap();
                // match distribution.roughness {
                //     pbrt_rs::Param::Float(ref v) => {
                //         // TODO: Need a conversion formula for phong
                //         writeln!(mat_file, "Ns {}", 2.0 / v[0]).unwrap();
                //         info!("Found roughness: {}", 2.0 / v[0]);
                //     }
                //     _ => panic!("Unsupported texture for metal material"),
                // }
                match kd {
                    pbrt_rs::parser::Spectrum::RGB(ref rgb) => {
                        let mut rgb = rgb.clone();
                        normalize_rgb(&mut rgb);
                        writeln!(mat_file, "Kd {} {} {}", rgb.r, rgb.g, rgb.b).unwrap()
                    }
                    pbrt_rs::parser::Spectrum::Texture(ref tex_name) => {
                        writeln!(mat_file, "Kd 0.0 0.0 0.0").unwrap();
                        let texture = &scene_info.textures[tex_name];
                        warn!(" - Texture file: {}", texture.filename);
                        writeln!(mat_file, "map_Kd {}", texture.filename).unwrap();
                        textures.push(texture.filename.clone());
                    }
                    _ => panic!("Unsupported texture for metal material"),
                }
            }
            pbrt_rs::BSDF::Metal { k, .. } => {
                writeln!(mat_file, "Ka 0.0 0.0 0.0").unwrap();
                writeln!(mat_file, "Kd 0.0 0.0 0.0").unwrap();
                writeln!(mat_file, "Tf 1.0 1.0 1.0").unwrap();
                writeln!(mat_file, "Ni 1.00").unwrap();
                writeln!(mat_file, "illum 3").unwrap();
                match k {
                    pbrt_rs::parser::Spectrum::RGB(ref rgb) => {
                        let mut rgb = rgb.clone();
                        normalize_rgb(&mut rgb);
                        writeln!(mat_file, "Ks {} {} {}", rgb.r, rgb.g, rgb.b).unwrap()
                    }
                    pbrt_rs::parser::Spectrum::Texture(ref tex_name) => {
                        writeln!(mat_file, "Ks 0.0 0.0 0.0").unwrap();
                        let texture = &scene_info.textures[tex_name];
                        warn!(" - Texture file: {}", texture.filename);
                        writeln!(mat_file, "map_Ks {}", texture.filename).unwrap();
                        textures.push(texture.filename.clone());
                    }
                    _ => panic!("Unsupported texture for metal material"),
                }
                warn!("Rougness conversion is broken");
                writeln!(mat_file, "Ns {}", 0.1).unwrap();
                // match metal.roughness {
                //     pbrt_rs::Param::Float(ref v) => {
                //         // TODO: Need a conversion formula for phong
                //         writeln!(mat_file, "Ns {}", 2.0 / v[0]).unwrap();
                //         info!("Found roughness: {}", 2.0 / v[0]);
                //     }
                //     _ => panic!("Unsupported texture for metal material"),
                // }
            }
        }
        mat_file.write_all(b"\n").unwrap();
    } // End of materials

    info!("Number of textures detected: {}", textures.len());
    for tex in &textures {
        info!(" - {}", tex);
    }
}

/*
 This function will write camera info like direction
*/
fn print_camera_info(scene_info: &pbrt_rs::Scene) {
    for cam in &scene_info.cameras {
        info!("Camera information: ");
        match cam {
            pbrt_rs::Camera::Perspective {
                fov,
                world_to_camera,
            } => {
                info!(" - fov: {}", fov);
                info!(" - world_to_camera: {:?}", world_to_camera);

                // Compute view direction and position
                // to help setup other rendering system
                let mat = world_to_camera.inverse_transform().unwrap();
                let aspect_ratio = scene_info.image_size.x as f32 / scene_info.image_size.y as f32;
                let fov_rad = Rad(fov * aspect_ratio * std::f32::consts::PI / 180.0); //2.0 * f32::tan((fov / 2.0) * f32::consts::PI / 180.0));//(fov * f32::consts::PI / 180.0);
                let camera_to_sample =
                    Matrix4::from_nonuniform_scale(-0.5, -0.5 * aspect_ratio, 1.0)
                        * Matrix4::from_translation(Vector3::new(-1.0, -1.0 / aspect_ratio, 0.0))
                        * perspective(fov_rad, 1.0, 1e-2, 1000.0)
                        * Matrix4::from_nonuniform_scale(-1.0, 1.0, -1.0); // undo gluPerspective (z neg)
                let sample_to_camera = camera_to_sample.inverse_transform().unwrap();

                let near_p = sample_to_camera.transform_point(Point3::new(0.5, 0.5, 0.0));
                let d = near_p.to_vec().normalize();

                info!(
                    " - position: {:?}",
                    mat.transform_point(Point3::new(0.0, 0.0, 0.0))
                );
                info!(" - view dir: {:?}", mat.transform_vector(d));
            }
        }
    }
}

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
            .format_timestamp(None)
            .parse_filters("debug")
            .init();
    } else {
        env_logger::Builder::from_default_env()
            .format_timestamp(None)
            .parse_filters("info")
            .init();
    }

    // The parsing
    let mut scene_info = pbrt_rs::Scene::default();
    let mut state = pbrt_rs::State::default();
    let working_dir = std::path::Path::new(scene_path_str).parent().unwrap();
    pbrt_rs::read_pbrt_file(
        scene_path_str,
        Some(&working_dir),
        &mut scene_info,
        &mut state,
    );

    // Then do some transformation
    // if it is necessary
    for s in &mut scene_info.shapes {
        match &mut s.data {
            pbrt_rs::Shape::Ply { filename, .. } => {
                s.data = pbrt_rs::ply::read_ply(std::path::Path::new(filename)).to_trimesh();
            }
            _ => (),
        }
    }

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
        .map(|v| match &v.data {
            pbrt_rs::Shape::TriMesh { points, .. } => points.len(),
            _ => panic!("All mesh need to be converted or drop"),
        })
        .sum();
    let indices_sum: usize = scene_info
        .shapes
        .iter()
        .map(|v| match &v.data {
            pbrt_rs::Shape::TriMesh { indices, .. } => indices.len() / 3,
            _ => panic!("All mesh need to be converted or drop"),
        })
        .sum();
    info!("Total: ");
    info!(" - #triangles: {}", tri_sum);
    info!(" - #indices: {}", indices_sum);

    // Camera information
    info!("Image size: {:?}", scene_info.image_size);
    print_camera_info(&scene_info);

    if let Some(obj_path) = matches.value_of("obj") {
        info!("Export in OBJ: {}", obj_path);
        let obj_file_path = Path::new(obj_path);
        let mtl_file_path = Path::new(obj_path).with_extension("mtl");

        let mut file = File::create(obj_file_path).unwrap();
        file.write_all(b"# OBJ EXPORTED USING pbrt_rs\n").unwrap();
        writeln!(file, "mtllib {}", mtl_file_path.to_str().unwrap()).unwrap();

        let mut mat_file = File::create(mtl_file_path).unwrap();
        mat_file
            .write_all(b"# OBJ EXPORTED USING pbrt_rs\n")
            .unwrap();

        export_obj(scene_info, &mut file, &mut mat_file);
    }
}
