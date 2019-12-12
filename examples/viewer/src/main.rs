#![allow(dead_code)]

extern crate cgmath;
extern crate clap;
extern crate env_logger;
extern crate pbrt_rs;
#[macro_use]
extern crate log;

use cgmath::*;

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

fn main() {
	let scene_path_str = "/home/muliana/projects/pbrt_rs/data/pbrt_rs_scenes/Spaceship/scene.pbrt";

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
	let image_buffer = vec![0.0; (image_size.x * image_size.y) as usize]; 
	for iy in 0..image_size.y {
		for ix in 0..image_size.x {
			let (p, d) = camera.generate(Point2::new(ix as f32 + 0.5, iy as f32 + 0.5));
			
			// TODO: Iterate on all the triangles
		}
	}

	// Save the image
	// TODO: Save the image as pfm

}