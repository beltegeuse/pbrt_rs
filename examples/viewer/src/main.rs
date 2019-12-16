#![allow(dead_code)]

extern crate byteorder;
extern crate cgmath;
extern crate clap;
extern crate env_logger;
extern crate pbrt_rs;
#[macro_use]
extern crate log;

use cgmath::*;
use clap::{App, Arg};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fs::File;
use std::path::Path;
use std::io::{BufRead, BufReader, BufWriter, Write};

pub struct Camera {
    pub img: Vector2<u32>,
    pub fov: f32, //< y
    // Internally
    camera_to_sample: Matrix4<f32>,
    sample_to_camera: Matrix4<f32>,
    to_world: Matrix4<f32>,
    to_local: Matrix4<f32>,
    // image rect
    image_rect_min: Point2<f32>,
    image_rect_max: Point2<f32>,
}

impl Camera {
    pub fn new(img: Vector2<u32>, fov: f32, mat: Matrix4<f32>) -> Camera {
        let to_world = mat;
        let to_local = to_world.inverse_transform().unwrap();

        // Compute camera informations
        // fov: y
        // TODO: Check this fov problem
        let aspect_ratio = img.x as f32 / img.y as f32;
        let fov_rad = Rad(fov * aspect_ratio * std::f32::consts::PI / 180.0);
        let camera_to_sample = Matrix4::from_nonuniform_scale(-0.5, -0.5 * aspect_ratio, 1.0)
            * Matrix4::from_translation(Vector3::new(-1.0, -1.0 / aspect_ratio, 0.0))
            * perspective(fov_rad, 1.0, 1e-2, 1000.0)
            * Matrix4::from_nonuniform_scale(-1.0, 1.0, -1.0); // undo gluPerspective (z neg)
        let sample_to_camera = camera_to_sample.inverse_transform().unwrap();

        // Compute the image plane inside the sample space.
        let p0 = sample_to_camera.transform_point(Point3::new(0.0, 0.0, 0.0));
        let p1 = sample_to_camera.transform_point(Point3::new(1.0, 1.0, 0.0));
        let image_rect_min = Point2::new(p0.x.min(p1.x), p0.y.min(p1.y)) / p0.z.min(p1.z);
        let image_rect_max = Point2::new(p0.x.max(p1.x), p0.y.max(p1.y)) / p0.z.max(p1.z);
        Camera {
            img,
            fov,
            camera_to_sample,
            sample_to_camera,
            to_world,
            to_local,
            image_rect_min,
            image_rect_max,
        }
    }

    pub fn size(&self) -> &Vector2<u32> {
        &self.img
    }

    pub fn scale_image(&mut self, s: f32) {
        self.img = Vector2::new(
            (s * self.img.x as f32) as u32,
            (s * self.img.y as f32) as u32,
        );
    }

    /// Compute the ray direction going through the pixel passed
    pub fn generate(&self, px: Point2<f32>) -> (Point3<f32>, Vector3<f32>) {
        let near_p = self.sample_to_camera.transform_point(Point3::new(
            px.x / (self.img.x as f32),
            px.y / (self.img.y as f32),
            0.0,
        ));
        let d = near_p.to_vec().normalize();

        (self.position(), self.to_world.transform_vector(d))
    }

    pub fn position(&self) -> Point3<f32> {
        self.to_world.transform_point(Point3::new(0.0, 0.0, 0.0))
    }

    pub fn print_info(&self) {
        let pix = Point2::new(self.img.x as f32 * 0.5 + 0.5, self.img.y as f32 * 0.5 + 0.5);
        let (pos, view_dir) = self.generate(pix);
        info!(" - Position: {:?}", pos);
        info!(" - View direction: {:?}", view_dir);
    }
}

#[derive(Debug)]
pub struct Intersection {
    pub t: f32, 
    pub p: Option<Point3<f32>>,
    pub n_geo: Option<Vector3<f32>>
}

impl Default for Intersection {
    fn default() -> Self { 
        return Self {
            t: std::f32::MAX,
            p: None,
            n_geo: None
        }
    } 
}

fn triangle_intersection(p_c: Point3<f32>, d_c: Vector3<f32>, 
    its: &mut Intersection,
    v0: Vector3<f32>, v1: Vector3<f32>, v2: Vector3<f32>) -> bool {
    let e1 = v1 - v0;
    let e2 = v2 - v0; 
    let n_geo = e1.cross(e2).normalize();
    let denom = d_c.dot(n_geo);
    if denom == 0.0 {
        return false;
    }
    // Distance for intersection
    let t = -(p_c - v0).dot(n_geo) / denom;
    if t < 0.0 {
        return false;
    }
    let p = p_c + t * d_c;
    let det = e1.cross(e2).magnitude();
    let u0 = e1.cross(p.to_vec() - v0);
    let v0 = (p.to_vec() - v0).cross(e2);
    if u0.dot(n_geo) < 0.0 || v0.dot(n_geo) < 0.0 {
        return false;
    }
    let v = u0.magnitude() / det;
    let u = v0.magnitude() / det;
    if u < 0.0 || v < 0.0 || u > 1.0 || v > 1.0 {
        return false;
    }
    if u + v <= 1.0 {
        if t < its.t {
            its.t = t;
            its.p = Some(p);
            its.n_geo = Some(n_geo);
            return true;
        }
    }
    false
}

fn intersection(scene: &pbrt_rs::Scene, p: Point3<f32>, d: Vector3<f32>) -> Intersection {
    let mut its = Intersection::default();
    
    for m in &scene.shapes {
        // Geometric information
        match m.data {
            pbrt_rs::Shape::TriMesh(ref data) => {
                // Do simple intesection
                let mat = m.matrix;
                let points: Vec<Vector3<f32>> = data.points
                .iter()
                .map(|n| mat.transform_point(n.clone()).to_vec())
                .collect();
                let indices = data.indices.clone();
                
                for i in indices {
                    triangle_intersection(p, d, &mut its, points[i.x], points[i.y], points[i.z]);
                }
            }
        }
    }
    
    return its;
} 

fn save_pfm(img_size: Vector2<u32>, data: Vec<f32>, imgout_path_str: &str) {
    let file = File::create(Path::new(imgout_path_str)).unwrap();
    let mut file = BufWriter::new(file);
    let header = format!("PF\n{} {}\n-1.0\n", img_size.x, img_size.y);
    file.write_all(header.as_bytes()).unwrap();
    for y in 0..img_size.y {
        for x in 0..img_size.x {
            let p = data[(y*img_size.x + x) as usize];
            file.write_f32::<LittleEndian>(p.abs()).unwrap();
            file.write_f32::<LittleEndian>(p.abs()).unwrap();
            file.write_f32::<LittleEndian>(p.abs()).unwrap();
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
    ).arg(
        Arg::with_name("ouput")
            .required(true)
            .takes_value(true)
            .index(2)
            .help("Output PFM"),
    )
    .get_matches();
    // Get params values
    let scene_path_str = matches
        .value_of("scene")
        .expect("no scene parameter provided");
    let output_str = matches
        .value_of("ouput")
        .expect("no ouput parameter provided");


	let mut scene_info = pbrt_rs::Scene::default();
    let mut state = pbrt_rs::State::default();
    let working_dir = std::path::Path::new(scene_path_str).parent().unwrap();
	pbrt_rs::read_pbrt_file(scene_path_str, &working_dir, &mut scene_info, &mut state);
	
	// For logging
	env_logger::Builder::from_default_env()
            .format_timestamp(None)
            .parse_filters("info")
            .init();

	// Print statistics
	info!("Scenes info: ");
	info!(" - BSDFS: {}", scene_info.materials.len());
	info!(" - Objects: ");
	info!("    * Unamed object: {}", scene_info.shapes.len());
	info!("    * Object: {}", scene_info.objects.len());
	info!("    * Object's instance: {}", scene_info.instances.len()); 

	// Load the camera information from PBRT
	let camera = {
		if let Some(camera) = scene_info.cameras.get(0) {
			match camera {
				pbrt_rs::Camera::Perspective(ref cam) => {
					let mat = cam.world_to_camera.inverse_transform().unwrap();
					info!("camera matrix: {:?}", mat);
					Camera::new(scene_info.image_size, cam.fov, mat)
				}
			}
		} else {
			panic!("The camera is not set!");
		}
	};

	// Render the image (depth image)
	let image_size = scene_info.image_size;
	let mut image_buffer = vec![0.0; (image_size.x * image_size.y) as usize]; 
	for iy in 0..image_size.y {
		for ix in 0..image_size.x {
            let (p, d) = camera.generate(Point2::new(ix as f32 + 0.5, iy as f32 + 0.5));
            
            // Compute the intersection
            let its = intersection(&scene_info, p, d);
            if let Some(pos) = its.p {
                let pix_id = (iy*image_size.x + ix) as usize;
                image_buffer[pix_id] = its.t;
            }
		}
    }
    
    let sum_dist = image_buffer.iter().sum::<f32>();
    let sum_dist = sum_dist / (image_size.x * image_size.y) as f32;
    println!("Sum average is: {}", sum_dist);

	// Save the image
    save_pfm(image_size, image_buffer, output_str);

}