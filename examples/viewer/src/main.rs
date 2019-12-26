#![allow(dead_code)]
// For the BVH tree
#![feature(iter_partition_in_place)]

extern crate byteorder;
extern crate cgmath;
extern crate clap;
extern crate env_logger;
extern crate pbr;
extern crate pbrt_rs;
#[macro_use]
extern crate log;

use byteorder::{LittleEndian, WriteBytesExt};
use cgmath::*;
use clap::{App, Arg};
use rayon::prelude::*;
use std::convert::TryInto;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Mutex;

// To avoid self intersection (in case)
const DISTANCE_BIAS: f32 = 0.001;

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

fn vec_min(v1: &Vector3<f32>, v2: &Vector3<f32>) -> Vector3<f32> {
    Vector3::new(v1.x.min(v2.x), v1.y.min(v2.y), v1.z.min(v2.z))
}

fn vec_max(v1: &Vector3<f32>, v2: &Vector3<f32>) -> Vector3<f32> {
    Vector3::new(v1.x.max(v2.x), v1.y.max(v2.y), v1.z.max(v2.z))
}

fn vec_div(v1: &Vector3<f32>, v2: &Vector3<f32>) -> Vector3<f32> {
    Vector3::new(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z)
}

fn vec_mult(v1: &Vector3<f32>, v2: &Vector3<f32>) -> Vector3<f32> {
    Vector3::new(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z)
}

fn vec_max_coords(v: Vector3<f32>) -> f32 {
    v.x.max(v.y.max(v.z))
}

fn vec_min_coords(v: Vector3<f32>) -> f32 {
    v.x.min(v.y.min(v.z))
}

#[derive(Debug)]
pub struct BoundingBox {
    pub p_min: Vector3<f32>,
    pub p_max: Vector3<f32>,
}

impl Default for BoundingBox {
    fn default() -> Self {
        Self {
            p_min: Vector3::new(std::f32::MAX, std::f32::MAX, std::f32::MAX),
            p_max: Vector3::new(std::f32::MIN, std::f32::MIN, std::f32::MIN),
        }
    }
}

impl BoundingBox {
    pub fn union_aabb(&self, b: &BoundingBox) -> BoundingBox {
        BoundingBox {
            p_min: vec_min(&self.p_min, &b.p_min),
            p_max: vec_max(&self.p_max, &b.p_max),
        }
    }

    pub fn union_vec(&self, v: &Vector3<f32>) -> BoundingBox {
        BoundingBox {
            p_min: vec_min(&self.p_min, v),
            p_max: vec_max(&self.p_max, v),
        }
    }

    pub fn size(&self) -> Vector3<f32> {
        self.p_max - self.p_min
    }

    pub fn center(&self) -> Vector3<f32> {
        self.size() * 0.5 + self.p_min
    }

    pub fn intersect(&self, p: &Point3<f32>, d: &Vector3<f32>, t_c: f32) -> Option<f32> {
        // TODO: direction inverse could be precomputed
        let t_0 = vec_div(&(self.p_min - p.to_vec()), &d);
        let t_1 = vec_div(&(self.p_max - p.to_vec()), &d);
        let t_min = vec_max_coords(vec_min(&t_0, &t_1));
        let t_max = vec_min_coords(vec_max(&t_0, &t_1));
        if t_min <= t_max {
            // TODO: Check if we need to add DISTANCE_BIAS
            if t_min >= t_c {
                None
            } else {
                Some(t_min)
            }
        } else {
            None
        }
    }
}

#[derive(Debug)]
struct Triangle {
    pub p0: Vector3<f32>,
    pub p1: Vector3<f32>,
    pub p2: Vector3<f32>,
}

impl Triangle {
    fn middle(&self) -> Vector3<f32> {
        (self.p0 + self.p1 + self.p2) / 3.0
    }

    fn intersection(&self, p_c: &Point3<f32>, d_c: &Vector3<f32>, its: &mut Intersection) -> bool {
        let e1 = self.p1 - self.p0;
        let e2 = self.p2 - self.p0;
        let n_geo = e1.cross(e2).normalize();
        let denom = d_c.dot(n_geo);
        if denom == 0.0 {
            return false;
        }
        // Distance for intersection
        let t = -(p_c - self.p0).dot(n_geo) / denom;
        if t < 0.0 {
            return false;
        }
        let p = p_c + t * d_c;
        let det = e1.cross(e2).magnitude();
        let u0 = e1.cross(p.to_vec() - self.p0);
        let v0 = (p.to_vec() - self.p0).cross(e2);
        if u0.dot(n_geo) < 0.0 || v0.dot(n_geo) < 0.0 {
            return false;
        }
        let v = u0.magnitude() / det;
        let u = v0.magnitude() / det;
        if u < 0.0 || v < 0.0 || u > 1.0 || v > 1.0 {
            return false;
        }
        if u + v <= 1.0 {
            // TODO: Review the condition because
            //      for now it only return true
            //      if the itersection is updated
            if t < its.t {
                its.t = t;
                its.p = Some(p);
                its.n_geo = Some(n_geo);
                return true;
            }
        }
        false
    }
}

#[derive(Debug)]
struct BVHNode {
    pub aabb: BoundingBox,
    pub first: usize,
    pub count: usize,
    pub left: Option<usize>,
    pub right: Option<usize>,
}

impl BVHNode {
    pub fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }
}

struct BHVAccel {
    pub triangles: Vec<Triangle>,
    pub nodes: Vec<BVHNode>,
    pub root: Option<usize>, // Root node
}

// Implementation from (C++): https://github.com/shiinamiyuki/minpt/blob/master/minpt.cpp
impl BHVAccel {
    // Internal build function
    // return the node ID (allocate node on the fly)
    fn build(&mut self, begin: usize, end: usize, depth: u32) -> Option<usize> {
        if end == begin {
            return None;
        }

        // TODO: Not very optimized ...
        let mut aabb = BoundingBox::default();
        for i in begin..end {
            aabb = aabb.union_vec(&self.triangles[i].p0);
            aabb = aabb.union_vec(&self.triangles[i].p1);
            aabb = aabb.union_vec(&self.triangles[i].p2);
        }

        // If the tree is too deep or not enough element
        // force to make a leaf
        // depth >= 20
        if end - begin <= 4 {
            // dbg!(aabb.size());
            self.nodes.push(BVHNode {
                aabb,
                first: begin,
                count: end - begin,
                left: None,
                right: None,
            });
            Some(self.nodes.len() - 1)
        } else {
            // For now cut on the biggest axis
            let aabb_size = aabb.size();
            let axis = if aabb_size.x > aabb_size.y {
                if aabb_size.x > aabb_size.z {
                    0
                } else {
                    2
                }
            } else {
                if aabb_size.y > aabb_size.z {
                    1
                } else {
                    2
                }
            };

            // Split based on largest axis (split inside the middle for now)
            // or split between triangles
            // TODO: Implements SAH
            // let split = (aabb.p_max[axis] + aabb.p_min[axis]) / 2.0;
            // let split_id = self.triangles[begin..end].iter_mut().partition_in_place(|t| t.middle()[axis] < split ) + begin;

            self.triangles[begin..end].sort_unstable_by(|t1, t2| {
                t1.middle()[axis].partial_cmp(&t2.middle()[axis]).unwrap()
            });
            let split_id = (begin + end) / 2;

            // TODO: Make better
            let left = self.build(begin, split_id, depth + 1);
            let right = self.build(split_id, end, depth + 1);
            self.nodes.push(BVHNode {
                aabb,
                first: 0, // TODO: Make the node invalid
                count: 0,
                left,
                right,
            });

            Some(self.nodes.len() - 1)
        }
    }

    pub fn create_from_scene(scene: &pbrt_rs::Scene) -> BHVAccel {
        // TODO: No object ID is associated for now...

        // Create the list of triangles from all the scene object
        // Note that for now, it is a simple BVH...
        let mut triangles = Vec::new();
        for m in &scene.shapes {
            // Geometric information
            match m.data {
                pbrt_rs::Shape::TriMesh(ref data) => {
                    // Do simple intesection
                    let mat = m.matrix;
                    let points: Vec<Vector3<f32>> = data
                        .points
                        .iter()
                        .map(|n| mat.transform_point(n.clone()).to_vec())
                        .collect();
                    let indices = data.indices.clone();

                    for i in indices {
                        triangles.push(Triangle {
                            p0: points[i.x],
                            p1: points[i.y],
                            p2: points[i.z],
                        });
                    }
                } // TODO: Need to do also for the instances of the same object
            }
        }

        let mut accel = BHVAccel {
            triangles,
            nodes: Vec::new(),
            root: None,
        };
        accel.root = accel.build(0, accel.triangles.len(), 0);
        info!("BVH stats: ");
        info!(" - Number of triangles: {}", accel.triangles.len());
        info!(" - Number of nodes: {}", accel.nodes.len());
        info!(
            " - AABB size root: {:?}",
            accel.nodes[accel.root.unwrap()].aabb.size()
        );
        // Check up the slides
        for n in &accel.nodes {
            if n.is_leaf() {
                let mut aabb = BoundingBox::default();
                for i in n.first..(n.first + n.count) {
                    aabb = aabb.union_vec(&accel.triangles[i].p0);
                    aabb = aabb.union_vec(&accel.triangles[i].p1);
                    aabb = aabb.union_vec(&accel.triangles[i].p2);
                }
                debug_assert_eq!(aabb.p_max, n.aabb.p_max);
                debug_assert_eq!(aabb.p_min, n.aabb.p_min);
            }
        }

        accel
    }

    fn intersection(&self, p: Point3<f32>, d: Vector3<f32>) -> Intersection {
        let mut its = Intersection::default();
        if self.root.is_none() {
            return its;
        }

        // Indices of nodes
        let mut stack: Vec<usize> = Vec::new();
        stack.reserve(100); // In case
        stack.push(self.root.unwrap());

        while let Some(curr_id) = stack.pop() {
            let n = &self.nodes[curr_id];
            let t_aabb = n.aabb.intersect(&p, &d, its.t);
            match (n.is_leaf(), t_aabb) {
                (_, None) => {
                    // Nothing to do as we miss the node
                }
                (true, Some(_t)) => {
                    for i in n.first..(n.first + n.count) {
                        self.triangles[i].intersection(&p, &d, &mut its);
                    }
                }
                (false, Some(_t)) => {
                    if let Some(left_id) = n.left {
                        stack.push(left_id);
                    }
                    if let Some(right_id) = n.right {
                        stack.push(right_id);
                    }
                }
            }
        }
        its
    }
}

#[derive(Debug)]
pub struct Intersection {
    pub t: f32,
    pub p: Option<Point3<f32>>,
    pub n_geo: Option<Vector3<f32>>,
}

impl Default for Intersection {
    fn default() -> Self {
        return Self {
            t: std::f32::MAX,
            p: None,
            n_geo: None,
        };
    }
}

fn intersection(scene: &pbrt_rs::Scene, p: Point3<f32>, d: Vector3<f32>) -> Intersection {
    let mut its = Intersection::default();

    for m in &scene.shapes {
        // Geometric information
        match m.data {
            pbrt_rs::Shape::TriMesh(ref data) => {
                // Do simple intesection
                let mat = m.matrix;
                let points: Vec<Vector3<f32>> = data
                    .points
                    .iter()
                    .map(|n| mat.transform_point(n.clone()).to_vec())
                    .collect();
                let indices = data.indices.clone();

                for i in indices {
                    let t = Triangle {
                        p0: points[i.x],
                        p1: points[i.y],
                        p2: points[i.z],
                    };
                    t.intersection(&p, &d, &mut its);
                }
            } // TODO: Need to do also for the instances of the same object
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
            // Flip vertically
            let id: usize = ((img_size.y - y - 1) * img_size.x + x).try_into().unwrap();
            let p = data[id];
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
        )
        .arg(
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

    // Construct the acceleration data structure
    info!("Build acceleration data structure ... ");
    let start = std::time::Instant::now();
    let accel = BHVAccel::create_from_scene(&scene_info);
    info!("Done {} ms", start.elapsed().as_millis());

    // Render the image (depth image)
    let image_size = scene_info.image_size;
    let image_buffer = Mutex::new(vec![0.0; (image_size.x * image_size.y) as usize]);
    let progress_bar = Mutex::new(pbr::ProgressBar::new(image_size.y as u64));

    let start = std::time::Instant::now();
    (0..image_size.y).into_par_iter().for_each(|iy| {
        let mut image_line = vec![0.0; image_size.x as usize];
        for ix in 0..image_size.x {
            let (p, d) = camera.generate(Point2::new(ix as f32 + 0.5, iy as f32 + 0.5));

            // Compute the intersection
            let its = accel.intersection(p, d);
            //let its = intersection(&scene_info, p, d);
            if let Some(_pos) = its.p {
                image_line[ix as usize] = its.t;
            }
        }

        {
            let beg = (iy * image_size.x) as usize;
            let end = beg + image_size.x as usize;
            image_buffer.lock().unwrap()[beg..end].copy_from_slice(&image_line);
            progress_bar.lock().unwrap().inc();
        }
    });
    info!("Done {} ms", start.elapsed().as_millis());

    let image_buffer = image_buffer.into_inner().unwrap();
    let sum_dist = image_buffer.iter().sum::<f32>();
    let sum_dist = sum_dist / (image_size.x * image_size.y) as f32;
    println!("Sum average is: {}", sum_dist);

    // Save the image
    save_pfm(image_size, image_buffer, output_str);
}
