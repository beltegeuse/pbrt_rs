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
    info!("Image size: {:?}", scene_info.image_size);
    for cam in &scene_info.cameras {
        info!("Camera information: ");
        match cam {
            pbrt_rs::Camera::Perspective(ref c) => {
                info!(" - fov: {}", c.fov);
                info!(" - world_to_camera: {:?}", c.world_to_camera);

                // Compute view direction and position
                // to help setup other rendering system
                let mat = c.world_to_camera.inverse_transform().unwrap();
                let aspect_ratio = scene_info.image_size.x as f32 / scene_info.image_size.y as f32;
                let fov_rad = Rad(c.fov * aspect_ratio * std::f32::consts::PI / 180.0); //2.0 * f32::tan((fov / 2.0) * f32::consts::PI / 180.0));//(fov * f32::consts::PI / 180.0);
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

    if let Some(obj_path) = matches.value_of("obj") {
        info!("Export in OBJ: {}", obj_path);
        let obj_file_path = Path::new(obj_path);
        let mtl_file_path = Path::new(obj_path).with_extension("mtl");

        let mut file = File::create(obj_file_path).unwrap();
        file.write(b"# OBJ EXPORTED USING pbrt_rs\n").unwrap();
        writeln!(file, "mtllib {}", mtl_file_path.to_str().unwrap()).unwrap();

        let normalize_rgb = |r: &mut f32, g: &mut f32, b: &mut f32| {
            let max = r.max(b.max(*g));
            if max > 1.0 {
                *r /= max;
                *g /= max;
                *b /= max;
            }
        };

        let default_mat = |f: &mut File| {
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
                            shape: &pbrt_rs::ShapeInfo,
                            f_obj: &mut File,
                            f_mat: &mut File| {
            info!("Exporting emission:");
            info!(" - shape_name: {}", shape_name);

            match shape.emission {
                Some(pbrt_rs::Param::RGB(r, g, b)) => {
                    info!(" - emission: [{}, {}, {}]", r, g, b);
                    writeln!(f_obj, "usemtl light_{}", id_light).unwrap();
                    // Write the material file because the light is special materials
                    writeln!(f_mat, "newmtl light_{}", id_light).unwrap();
                    writeln!(f_mat, "Ns 0.0").unwrap();
                    writeln!(f_mat, "Ka 0.000000 0.000000 0.000000").unwrap();
                    writeln!(f_mat, "Kd 0.0 0.0 0.0").unwrap();
                    writeln!(f_mat, "Ke {} {} {}", r, g, b).unwrap();
                    writeln!(f_mat, "Ni 0.000000").unwrap();
                    writeln!(f_mat, "d 1.000000").unwrap();
                    writeln!(f_mat, "illum 7").unwrap();
                    f_mat.write(b"\n").unwrap();
                }
                _ => panic!("No support for this emission profile"),
            }
        };

        let mut file_material = File::create(mtl_file_path).unwrap();
        file_material
            .write(b"# OBJ EXPORTED USING pbrt_rs\n")
            .unwrap();
        {
            // Write default material
            writeln!(file_material, "newmtl export_default").unwrap();
            default_mat(&mut file_material);
            file_material.write(b"\n").unwrap();
        }

        // Need to write manually the obj file
        // --- Write all uname shapes
        let mut offset_point = 1;
        let mut offset_normal = 1;
        let mut offset_uv = 1;
        let mut nb_light = 0;
        for (i, shape) in scene_info.shapes.iter().enumerate() {
            let material_name = shape.material_name.clone();
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

                    // --- Write material
                    match material_name {
                        None => {
                            if shape.emission.is_none() {
                                writeln!(file, "usemtl {}", "export_default").unwrap();
                            } else {
                                emission_mat(
                                    nb_light,
                                    format!("Unamed_{}", i),
                                    &shape,
                                    &mut file,
                                    &mut file_material,
                                );
                                nb_light += 1;
                            }
                        }
                        Some(ref m) => {
                            if shape.emission.is_none() {
                                writeln!(file, "usemtl {}", m).unwrap();
                            } else {
                                warn!("Overwrite materials as it is a light");
                                emission_mat(
                                    nb_light,
                                    format!("Unamed_{}", i),
                                    &shape,
                                    &mut file,
                                    &mut file_material,
                                );
                                nb_light += 1;
                            }
                        }
                    };

                    // --- Indicies
                    if data.indices.len() % 3  != 0 {
                        error!("Number of vertices not multiples of 3: {}", data.indices.len());
                        continue;
                    }
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
        } // End shapes

        // Export the materials
        let mut textures = vec![];
        info!("Exporting bsdfs...");
        for (name, bdsf) in scene_info.materials.iter() {
            info!(" - {}", name);
            writeln!(file_material, "newmtl {}", name).unwrap();
            match bdsf {
                pbrt_rs::BSDF::Matte(ref matte) => {
                    writeln!(file_material, "Ns 1.0").unwrap();
                    writeln!(file_material, "Ka 0.0 0.0 0.0").unwrap();
                    writeln!(file_material, "Tf 1.0 1.0 1.0").unwrap();
                    writeln!(file_material, "Ks 0.0 0.0 0.0").unwrap();
                    writeln!(file_material, "illum 4").unwrap();
                    match matte.kd {
                        pbrt_rs::Param::RGB(mut r, mut g, mut b) => {
                            normalize_rgb(&mut r, &mut g, &mut b);
                            writeln!(file_material, "Kd {} {} {}", r, g, b).unwrap()
                        }
                        pbrt_rs::Param::Name(ref tex_name) => {
                            writeln!(file_material, "Kd 0.0 0.0 0.0").unwrap();
                            let texture = &scene_info.textures[tex_name];
                            warn!(" - Texture file: {}", texture.filename);
                            writeln!(file_material, "map_Kd {}", texture.filename).unwrap();
                            textures.push(texture.filename.clone());
                        }
                        _ => panic!("Unsupported texture for matte material"),
                    }
                }
                pbrt_rs::BSDF::Glass(ref _b) => {
                    writeln!(file_material, "Ns 1000").unwrap();
                    writeln!(file_material, "Ka 0.0 0.0 0.0").unwrap();
                    writeln!(file_material, "Kd 0.0 0.0 0.0").unwrap();
                    writeln!(file_material, "Tf 0.1 0.1 0.1").unwrap();
                    writeln!(file_material, "Ks 0.5 0.5 0.5").unwrap();
                    writeln!(file_material, "Ni 1.31").unwrap(); // Glass
                    writeln!(file_material, "d 1.000000").unwrap();
                    writeln!(file_material, "illum 7").unwrap();
                    // TODO: Read the properties
                }
                pbrt_rs::BSDF::Mirror(ref mirror) => {
                    writeln!(file_material, "Ns 100000.0").unwrap();
                    writeln!(file_material, "Ka 0.0 0.0 0.0").unwrap();
                    writeln!(file_material, "Kd 0.0 0.0 0.0").unwrap();
                    writeln!(file_material, "Tf 1.0 1.0 1.0").unwrap();
                    writeln!(file_material, "Ni 1.00").unwrap();
                    writeln!(file_material, "illum 3").unwrap();
                    match mirror.kr {
                        pbrt_rs::Param::RGB(mut r, mut g, mut b) => {
                            normalize_rgb(&mut r, &mut g, &mut b);
                            writeln!(file_material, "Ks {} {} {}", r, g, b).unwrap()
                        }
                        _ => panic!("Unsupported texture for mirror material"),
                    }
                }
                pbrt_rs::BSDF::Substrate(ref substrate) => {
                    writeln!(file_material, "Ka 0.0 0.0 0.0").unwrap();
                    writeln!(file_material, "Tf 1.0 1.0 1.0").unwrap();
                    writeln!(file_material, "Ni 1.0").unwrap();
                    writeln!(file_material, "illum 4").unwrap();
                    match substrate.ks {
                        pbrt_rs::Param::RGB(mut r, mut g, mut b) => {
                            normalize_rgb(&mut r, &mut g, &mut b);
                            writeln!(file_material, "Ks {} {} {}", r, g, b).unwrap()
                        }
                        pbrt_rs::Param::Name(ref tex_name) => {
                            writeln!(file_material, "Ks 0.0 0.0 0.0").unwrap();
                            let texture = &scene_info.textures[tex_name];
                            warn!(" - Texture file: {}", texture.filename);
                            writeln!(file_material, "map_Ks {}", texture.filename).unwrap();
                            textures.push(texture.filename.clone());
                        }
                        _ => panic!("Unsupported texture for metal material"),
                    }
                    match substrate.u_roughness {
                        pbrt_rs::Param::Float(ref v) => {
                            // TODO: Need a conversion formula for phong
                            writeln!(file_material, "Ns {}", 2.0 / v[0]).unwrap();
                            info!("Found roughness: {}", 2.0 / v[0]);
                        }
                        _ => panic!("Unsupported texture for metal material"),
                    }
                    match substrate.kd {
                        pbrt_rs::Param::RGB(mut r, mut g, mut b) => {
                            normalize_rgb(&mut r, &mut g, &mut b);
                            writeln!(file_material, "Kd {} {} {}", r, g, b).unwrap()
                        }
                        pbrt_rs::Param::Name(ref tex_name) => {
                            writeln!(file_material, "Kd 0.0 0.0 0.0").unwrap();
                            let texture = &scene_info.textures[tex_name];
                            warn!(" - Texture file: {}", texture.filename);
                            writeln!(file_material, "map_Kd {}", texture.filename).unwrap();
                            textures.push(texture.filename.clone());
                        }
                        _ => panic!("Unsupported texture for metal material"),
                    }
                }
                pbrt_rs::BSDF::Metal(ref metal) => {
                    writeln!(file_material, "Ka 0.0 0.0 0.0").unwrap();
                    writeln!(file_material, "Kd 0.0 0.0 0.0").unwrap();
                    writeln!(file_material, "Tf 1.0 1.0 1.0").unwrap();
                    writeln!(file_material, "Ni 1.00").unwrap();
                    writeln!(file_material, "illum 3").unwrap();
                    match metal.k {
                        pbrt_rs::Param::RGB(mut r, mut g, mut b) => {
                            normalize_rgb(&mut r, &mut g, &mut b);
                            writeln!(file_material, "Ks {} {} {}", r, g, b).unwrap()
                        }
                        pbrt_rs::Param::Name(ref tex_name) => {
                            writeln!(file_material, "Ks 0.0 0.0 0.0").unwrap();
                            let texture = &scene_info.textures[tex_name];
                            warn!(" - Texture file: {}", texture.filename);
                            writeln!(file_material, "map_Ks {}", texture.filename).unwrap();
                            textures.push(texture.filename.clone());
                        }
                        _ => panic!("Unsupported texture for metal material"),
                    }
                    match metal.roughness {
                        pbrt_rs::Param::Float(ref v) => {
                            // TODO: Need a conversion formula for phong
                            writeln!(file_material, "Ns {}", 2.0 / v[0]).unwrap();
                            info!("Found roughness: {}", 2.0 / v[0]);
                        }
                        _ => panic!("Unsupported texture for metal material"),
                    }
                }
                _ => panic!("Unsupported type"),
            }
            file_material.write(b"\n").unwrap();
        } // End of materials

        info!("Number of textures detected: {}", textures.len());
        for tex in &textures {
            info!(" - {}", tex);
        }
    }
}
