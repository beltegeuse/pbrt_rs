#![allow(dead_code)]
#![allow(clippy::cognitive_complexity)]

// For logging
#[macro_use]
extern crate log;
extern crate nom;
// Vector representation
extern crate cgmath;
// For loading ply files
extern crate ply_rs;

// parser
use cgmath::*;
use std::collections::HashMap;
use std::io::Read;
use std::rc::Rc;
use std::time::Instant;

// Mods
pub mod parser;
pub mod ply;

// Parser
use parser::*;

// Helper to remove the map elements
// if the elemet is not found, the default value is used
macro_rules! remove_default {
    ($map: expr, $name:expr, $default:expr) => {{
        if let Some(v) = $map.remove($name) {
            v
        } else {
            $default
        }
    }};
}

/// Camera representations
pub enum Camera {
    Perspective {
        fov: f32,
        world_to_camera: Matrix4<f32>,
    },
}
impl Camera {
    fn new(mut named_token: NamedToken, mat: Matrix4<f32>) -> Option<Self> {
        match &named_token.internal_type[..] {
            "perspective" => {
                let fov = named_token
                    .values
                    .remove("fov")
                    .expect("fov is not given")
                    .into_float()[0];
                Some(Camera::Perspective {
                    fov,
                    world_to_camera: mat,
                })
            }
            _ => {
                warn!(
                    "Camera case with {:?} is not cover",
                    named_token.internal_type
                );
                None
            }
        }
    }
}

//// Texture representation
#[derive(Debug)]
pub struct Texture {
    pub filename: String,
    pub trilinear: bool,
}

pub enum Roughness {
    Isotropic(BSDFFloat),
    Anisotropic { u: BSDFFloat, v: BSDFFloat },
}
pub struct Distribution {
    pub roughness: Roughness, // Depends of the material (metal: 0.01 iso, glass optional)
    pub remaproughness: bool, // True
}

// BSDF representation
pub enum BSDF {
    Matte {
        kd: Spectrum,             // 0.5
        sigma: Option<BSDFFloat>, // Pure lambertian if not provided
        bumpmap: Option<BSDFFloat>,
    },
    Metal {
        eta: Spectrum,              // Cu
        k: Spectrum,                // Cu
        distribution: Distribution, // 0.01 Iso
        bumpmap: Option<BSDFFloat>,
    },
    Substrate {
        kd: Spectrum,               // 0.5
        ks: Spectrum,               // 0.5
        distribution: Distribution, // 0.1
        bumpmap: Option<BSDFFloat>,
    },
    Glass {
        kr: Spectrum, // 1
        kt: Spectrum, // 1
        distribution: Option<Distribution>,
        eta: BSDFFloat, // 1.5
        bumpmap: Option<BSDFFloat>,
    },
    Mirror {
        kr: Spectrum, // 0.9
        bumpmap: Option<BSDFFloat>,
    },
    // TODO:
    // disney	DisneyMaterial
    // fourier	FourierMaterial
    // hair	HairMaterial
    // kdsubsurface	KdSubsurfaceMaterial
    // mix	MixMaterial
    // none	A special material that signifies that the surface it is associated with should be ignored for ray intersections. (This is useful for specifying regions of space associated with participating media.)
    // plastic	PlasticMaterial
    // subsurface	SubsurfaceMaterial
    // translucent	TranslucentMaterial
    // uber	UberMaterial
}
impl BSDF {
    fn new(mut named_token: NamedToken, unamed: bool) -> Option<Self> {
        // Get the BSDF type
        let bsdf_type = if unamed {
            named_token.internal_type
        } else {
            named_token
                .values
                .remove("type")
                .expect("bsdf type param is required")
                .into_string()
        };

        let bumpmap = match named_token.values.remove("bumpmap") {
            Some(v) => Some(v.into_bsdf_float()),
            None => None,
        };

        let parse_distribution =
            |map: &mut HashMap<String, Value>, default: Option<f32>| -> Option<Distribution> {
                let remaproughness =
                    remove_default!(map, "remaproughness", Value::Boolean(true)).into_bool();
                let alpha = match map.remove("roughness") {
                    Some(v) => Some(Roughness::Isotropic(v.into_bsdf_float())),
                    None => {
                        let u = map.remove("uroughness");
                        let v = map.remove("vroughness");
                        if u.is_some() && v.is_some() {
                            let u = u.unwrap().into_bsdf_float();
                            let v = v.unwrap().into_bsdf_float();
                            match (u, v) {
                                (BSDFFloat::Float(v_u), BSDFFloat::Float(v_v)) => {
                                    if v_u == v_v {
                                        Some(Roughness::Isotropic(BSDFFloat::Float(v_v)))
                                    } else {
                                        Some(Roughness::Anisotropic {
                                            u: BSDFFloat::Float(v_u),
                                            v: BSDFFloat::Float(v_v),
                                        })
                                    }
                                }
                                (u, v) => Some(Roughness::Anisotropic { u, v }),
                            }
                        } else if u.is_none() && v.is_none() {
                            None
                        } else {
                            panic!("{:?} {:?} roughness issue", u, v);
                        }
                    }
                };

                let alpha = if default.is_some() && alpha.is_none() {
                    Some(Roughness::Isotropic(BSDFFloat::Float(default.unwrap())))
                } else {
                    alpha
                };

                match alpha {
                    None => None,
                    Some(roughness) => Some(Distribution {
                        roughness,
                        remaproughness,
                    }),
                }
            };

        let bsdf = match &bsdf_type[..] {
            "matte" => {
                let kd = remove_default!(named_token.values, "Kd", Value::RGB(RGB::color(0.5)))
                    .into_spectrum();
                let sigma = match named_token.values.remove("sigma") {
                    None => None,
                    Some(v) => Some(v.into_bsdf_float()),
                };
                Some(BSDF::Matte { kd, sigma, bumpmap })
            }
            "metal" => {
                // TODO: Need to be able to export other material params
                let eta = remove_default!(
                    named_token.values,
                    "eta",
                    Value::RGB(RGB {
                        r: 0.199_990_69,
                        g: 0.922_084_6,
                        b: 1.099_875_9
                    })
                )
                .into_spectrum();
                let k = remove_default!(
                    named_token.values,
                    "k",
                    Value::RGB(RGB {
                        r: 3.904_635_4,
                        g: 2.447_633_3,
                        b: 2.137_652_6
                    })
                )
                .into_spectrum();

                let distribution = parse_distribution(&mut named_token.values, Some(0.01)).unwrap();
                Some(BSDF::Metal {
                    eta,
                    k,
                    distribution,
                    bumpmap,
                })
            }
            "substrate" => {
                let kd = remove_default!(named_token.values, "Kd", Value::RGB(RGB::color(0.5)))
                    .into_spectrum();
                let ks = remove_default!(named_token.values, "Ks", Value::RGB(RGB::color(0.5)))
                    .into_spectrum();
                let distribution = parse_distribution(&mut named_token.values, Some(0.1)).unwrap();
                Some(BSDF::Substrate {
                    kd,
                    ks,
                    distribution,
                    bumpmap,
                })
            }
            "glass" => {
                let kr = remove_default!(named_token.values, "Kr", Value::RGB(RGB::color(1.0)))
                    .into_spectrum();
                let kt = remove_default!(named_token.values, "Kt", Value::RGB(RGB::color(1.0)))
                    .into_spectrum();
                let eta = if let Some(eta) = named_token.values.remove("eta") {
                    eta.into_bsdf_float()
                } else {
                    remove_default!(named_token.values, "index", Value::Float(vec![1.5]))
                        .into_bsdf_float()
                };
                let distribution = parse_distribution(&mut named_token.values, None);

                Some(BSDF::Glass {
                    kr,
                    kt,
                    distribution,
                    eta,
                    bumpmap,
                })
            }
            "mirror" => {
                let kr = remove_default!(named_token.values, "Kr", Value::RGB(RGB::color(0.9)))
                    .into_spectrum();
                Some(BSDF::Mirror { kr, bumpmap })
            }
            _ => {
                warn!("BSDF case with {} is not cover", bsdf_type);
                None
            }
        };

        if bsdf.is_some() {
            if !named_token.values.is_empty() {
                panic!("Miss parameters: {:?}", named_token.values);
            }
        }

        bsdf
    }
}

/// Mesh representation
#[derive(Debug)]
pub enum Shape {
    TriMesh {
        indices: Vec<Vector3<usize>>,
        points: Vec<Point3<f32>>,
        normals: Option<Vec<Vector3<f32>>>,
        uv: Option<Vec<Vector2<f32>>>,
    },
    Ply {
        filename: String,
        alpha: Option<Texture>,
        shadowalpha: Option<Texture>,
    },
}
impl Shape {
    fn new(mut named_token: NamedToken, wk: &std::path::Path) -> Option<Self> {
        match &named_token.internal_type[..] {
            "trianglemesh" => {
                let points = named_token
                    .values
                    .remove("P")
                    .expect(&format!("P is required {:?}", named_token))
                    .into_vector3();
                let points = points.into_iter().map(|v| Point3::from_vec(v)).collect();
                let indices = named_token
                    .values
                    .remove("indices")
                    .expect(&format!("indice is required {:?}", named_token))
                    .into_integer();
                if indices.len() % 3 != 0 {
                    panic!("Support only 3 indices list {:?}", named_token);
                }
                let indices = indices
                    .chunks(3)
                    .map(|v| Vector3::new(v[0] as usize, v[1] as usize, v[2] as usize))
                    .collect();
                let normals = if let Some(v) = named_token.values.remove("N") {
                    Some(v.into_vector3())
                } else {
                    None
                };
                let uv = if let Some(v) = named_token.values.remove("uv") {
                    let v = v.into_float();
                    assert_eq!(v.len() % 2, 0);
                    let v = v.chunks(2).map(|v| Vector2::new(v[0], v[1])).collect();
                    Some(v)
                } else {
                    None
                };
                Some(Shape::TriMesh {
                    indices,
                    points,
                    normals,
                    uv,
                })
            }
            "plymesh" => {
                let filename = named_token
                    .values
                    .remove("filename")
                    .expect("filename is required")
                    .into_string();
                let filename = wk.join(filename).to_str().unwrap().to_owned();
                Some(Shape::Ply {
                    filename,
                    alpha: None,       // FIXME
                    shadowalpha: None, // FIXME
                })
            }
            _ => {
                warn!("Shape case with {} is not cover", named_token.internal_type);
                None
            }
        }
    }
}

/// Lights
#[derive(Debug)]
pub enum Light {
    Distant {
        luminance: Spectrum,
        from: Point3<f32>,
        to: Point3<f32>,
        scale: RGB,
    },
    Infinite {
        luminance: Spectrum,
        samples: u32,
        scale: RGB,
    },
    Point {
        intensity: Spectrum,
        from: Point3<f32>,
        scale: RGB,
    },
}
impl Light {
    fn new(mut named_token: NamedToken) -> Option<Self> {
        let scale = if let Some(scale) = named_token.values.remove("scale") {
            scale.into_rgb()
        } else {
            RGB::color(1.0)
        };

        match &named_token.internal_type[..] {
            "infinite" => {
                let samples =
                    remove_default!(named_token.values, "samples", Value::Integer(vec![1]))
                        .into_integer()[0] as u32;
                let luminance =
                    remove_default!(named_token.values, "L", Value::RGB(RGB::color(1.0)))
                        .into_spectrum();

                // In case the map name is provide, we will replace the luminance
                let luminance = if let Some(mapname) = named_token.values.remove("mapname") {
                    Spectrum::Mapname(mapname.into_string())
                } else {
                    luminance
                };

                Some(Light::Infinite {
                    luminance,
                    samples,
                    scale,
                })
            }
            "point" => {
                let intensity =
                    remove_default!(named_token.values, "I", Value::RGB(RGB::color(1.0)))
                        .into_spectrum();
                let from = remove_default!(
                    named_token.values,
                    "from",
                    Value::Vector3(vec![Vector3::new(0.0, 0.0, 0.0)])
                )
                .into_vector3()[0];
                let from = Point3::from_vec(from);
                Some(Light::Point {
                    intensity,
                    from,
                    scale,
                })
            }
            "distant" => {
                let luminance =
                    remove_default!(named_token.values, "L", Value::RGB(RGB::color(1.0)))
                        .into_spectrum();
                let from = remove_default!(
                    named_token.values,
                    "from",
                    Value::Vector3(vec![Vector3::new(0.0, 0.0, 0.0)])
                )
                .into_vector3()[0];
                let to = remove_default!(
                    named_token.values,
                    "to",
                    Value::Vector3(vec![Vector3::new(0.0, 0.0, 0.0)])
                )
                .into_vector3()[0];
                let from = Point3::from_vec(from);
                let to = Point3::from_vec(to);
                Some(Light::Distant {
                    luminance,
                    from,
                    to,
                    scale,
                })
            }
            _ => {
                warn!("Light case with {} is not cover", named_token.internal_type);
                None
            }
        }
    }
}
/// State of the parser
#[derive(Debug)]
pub struct State {
    named_material: Vec<Option<String>>,
    matrix: Vec<Matrix4<f32>>,
    emission: Vec<Option<Spectrum>>,
    object: Option<ObjectInfo>,
}
impl Default for State {
    fn default() -> Self {
        State {
            named_material: vec![None],
            matrix: vec![Matrix4::identity()],
            emission: vec![None],
            object: None,
        }
    }
}
impl State {
    // State
    fn save(&mut self) {
        let new_material = self.named_material.last().unwrap().clone();
        self.named_material.push(new_material);
        let new_matrix = *self.matrix.last().unwrap();
        self.matrix.push(new_matrix);
        let new_emission = self.emission.last().unwrap().clone();
        self.emission.push(new_emission);
    }
    fn restore(&mut self) {
        self.named_material.pop();
        self.matrix.pop();
        self.emission.pop();
    }

    // Matrix
    fn matrix(&self) -> Matrix4<f32> {
        *self.matrix.last().unwrap()
    }
    fn replace_matrix(&mut self, m: Matrix4<f32>) {
        let curr_mat = self.matrix.last_mut().unwrap();
        curr_mat.clone_from(&m);
    }
    // Named material
    fn named_material(&self) -> Option<String> {
        self.named_material.last().unwrap().clone()
    }
    fn set_named_matrial(&mut self, s: String) {
        let last_id = self.named_material.len() - 1;
        self.named_material[last_id] = Some(s);
    }
    // Emission
    fn emission(&self) -> Option<Spectrum> {
        self.emission.last().unwrap().clone()
    }
    fn set_emission(&mut self, e: Spectrum) {
        let last_id = self.emission.len() - 1;
        self.emission[last_id] = Some(e);
    }
    // Object
    fn new_object(&mut self, name: String) {
        self.object = Some(ObjectInfo {
            name,
            shapes: Vec::new(),
            matrix: self.matrix(),
        });
    }
    fn finish_object(&mut self) -> ObjectInfo {
        std::mem::replace(&mut self.object, None).unwrap()
    }
}

/// Scene representation
#[derive(Debug)]
pub struct ShapeInfo {
    pub data: Shape,
    pub material_name: Option<String>,
    pub matrix: Matrix4<f32>,
    pub emission: Option<Spectrum>,
}
impl ShapeInfo {
    fn new(shape: Shape, matrix: Matrix4<f32>) -> Self {
        Self {
            data: shape,
            material_name: None,
            matrix,
            emission: None,
        }
    }
}
#[derive(Debug)]
pub struct InstanceInfo {
    pub matrix: Matrix4<f32>,
    pub name: String,
}
#[derive(Debug)]
pub struct ObjectInfo {
    pub name: String,
    pub shapes: Vec<ShapeInfo>,
    pub matrix: Matrix4<f32>,
}

pub struct Scene {
    // General scene info
    pub cameras: Vec<Camera>,
    pub image_size: Vector2<u32>,
    // Materials
    pub number_unamed_materials: usize,
    pub materials: HashMap<String, BSDF>,
    pub textures: HashMap<String, Texture>,
    // 3D objects
    pub shapes: Vec<ShapeInfo>,                   //< unamed shapes
    pub objects: HashMap<String, Rc<ObjectInfo>>, //< shapes with objects
    pub instances: Vec<InstanceInfo>,             //< instances on the shapes
    pub lights: Vec<Light>,                       //< list of all light sources
    pub transforms: HashMap<String, Matrix4<f32>>,
}
impl Default for Scene {
    fn default() -> Self {
        Scene {
            cameras: Vec::default(),
            image_size: Vector2::new(512, 512),
            // materials
            number_unamed_materials: 0,
            materials: HashMap::default(),
            textures: HashMap::default(),
            // 3d object information
            shapes: Vec::default(),
            objects: HashMap::default(),
            instances: Vec::default(),
            lights: Vec::default(),
            transforms: HashMap::default(),
        }
    }
}

pub fn read_pbrt(
    scene_string: &str,
    working_dir: &std::path::Path,
    scene_info: &mut Scene,
    state: &mut State,
) {
    // Parse the scene
    let (scene_string, tokens) =
        parse::<nom::error::VerboseError<&str>>(scene_string).expect("Error during parsing");
    // Check that we parsed all!
    match scene_string {
        "" => (),
        _ => panic!("Parsing is not complete: {:?}", scene_string),
    }

    // pub enum Token {
    //     Texture {
    //         name: String,
    //         t: String,
    //         class: String,
    //         values: HashMap<String, Value>,
    //     },
    // }

    for t in tokens {
        match t {
            Token::Transform(values) => {
                let m00 = values[0];
                let m01 = values[1];
                let m02 = values[2];
                let m03 = values[3];
                let m10 = values[4];
                let m11 = values[5];
                let m12 = values[6];
                let m13 = values[7];
                let m20 = values[8];
                let m21 = values[9];
                let m22 = values[10];
                let m23 = values[11];
                let m30 = values[12];
                let m31 = values[13];
                let m32 = values[14];
                let m33 = values[15];
                #[rustfmt::skip]
                let matrix = Matrix4::new(
                    m00, m01, m02, m03,
                    m10, m11, m12, m13,
                    m20, m21, m22, m23,
                    m30, m31, m32, m33,
                );
                state.replace_matrix(matrix);
            },
            Token::ConcatTransform(values) => {
                let m00 = values[0];
                let m01 = values[1];
                let m02 = values[2];
                let m03 = values[3];
                let m10 = values[4];
                let m11 = values[5];
                let m12 = values[6];
                let m13 = values[7];
                let m20 = values[8];
                let m21 = values[9];
                let m22 = values[10];
                let m23 = values[11];
                let m30 = values[12];
                let m31 = values[13];
                let m32 = values[14];
                let m33 = values[15];

                #[rustfmt::skip]
                let matrix = state.matrix() * Matrix4::new(
                    m00, m01, m02, m03,
                    m10, m11, m12, m13,
                    m20, m21, m22, m23,
                    m30, m31, m32, m33,
                );
                state.replace_matrix(matrix);
            },
            Token::Scale(values) => {
                let matrix = state.matrix()
                    * Matrix4::from_diagonal(Vector4::new(
                        values[0], values[1], values[2], 1.0,
                    ));
                state.replace_matrix(matrix);
            },
            Token::LookAt {
                eye, look, up
            } => {
                let dir = (look - eye).normalize();
                let left = -dir.cross(up.normalize()).normalize();
                let new_up = dir.cross(left);

                #[rustfmt::skip]
                let matrix = state.matrix() *  Matrix4::new(
                    left.x, left.y, left.z, 0.0,
                    new_up.x, new_up.y, new_up.z, 0.0,
                    dir.x, dir.y, dir.z, 0.0,
                    eye.x, eye.y, eye.z, 1.0,
                ).inverse_transform().unwrap();

                state.replace_matrix(matrix);
            },
            Token::Translate(values) => {
                let matrix = state.matrix()
                        * Matrix4::from_translation(values);
                state.replace_matrix(matrix);
            },
            Token::Rotate {
                angle,
                axis
            } => {
                let matrix = state.matrix() * Matrix4::from_axis_angle(axis, Deg(angle));
                state.replace_matrix(matrix);
            },
            Token::Keyword(key) => {
                match key {
                    Keyword::AttributeBegin | Keyword::TransformBegin => {
                        state.save();
                    }
                    Keyword::AttributeEnd | Keyword::TransformEnd => {
                        state.restore();
                    }
                    Keyword::Identity => {
                        state.replace_matrix(Matrix4::from_diagonal(Vector4::new(
                        1.0, 1.0, 1.0, 1.0,
                    )));
                    }
                    Keyword::WorldBegin => {
                         // Reinit the transformation matrix
                        state.replace_matrix(Matrix4::identity());
                    }
                    Keyword::ObjectEnd => {
                        let object = state.finish_object();
                        scene_info
                        .objects
                        .insert(object.name.clone(), Rc::new(object));
                    }
                    Keyword::WorldEnd => {
                        // Nothing?
                    }
                    Keyword::ReverseOrientation => {
                        todo!();
                    }
                }
            },
            Token::ActiveTransform(_) => {
                todo!();
            },
            Token::MediumInterface { .. } => {
                todo!()
            },
            Token::Texture {
                name, class, mut values, .. // t not read
            } => {
                    // TODO: WK
                    // Check type as Well... {spectrum or float} -> Two lists

                    // TODO: A lot of parameters...
                    match &class[..] {
                        "imagemap" => {
                            let filename = working_dir.join(values.remove("filename").unwrap().into_string()).to_str().unwrap().to_owned();
                            scene_info.textures.insert(name, Texture {
                                filename,
                                trilinear: remove_default!(values, "trilinear", Value::Boolean(false)).into_bool(),
                            });
                        }
                        _ => warn!("texture type {} is ignored", class)
                    }
                },
            Token::NamedToken(mut named_token) => {
                // pub enum NamedTokenType {
                //     MakeNamedMedium,
                // }

                // pub struct NamedToken {
                //     pub internal_type: String,
                //     pub values: HashMap<String, Value>,
                //     pub object_type: NamedTokenType,
                // }

                match named_token.object_type {
                    NamedTokenType::Accelerator | NamedTokenType::Integrator | NamedTokenType::Sampler | NamedTokenType::PixelFilter | NamedTokenType::SurfaceIntegrator | NamedTokenType::VolumeIntegrator => {
                        // Nothing...
                    },
                    NamedTokenType::Camera => {
                        if let Some(c) = Camera::new(named_token, state.matrix()) {
                            scene_info.cameras.push(c);
                        }
                    },
                    NamedTokenType::MakeNamedMaterial => {
                        let name = named_token.internal_type.clone();
                        if let Some(bsdf) = BSDF::new(named_token, false) {
                            scene_info.materials.insert(name, bsdf);
                        }
                    }
                    NamedTokenType::NamedMaterial => {
                        assert!(named_token.values.is_empty());
                        state.set_named_matrial(named_token.internal_type);
                    }
                    NamedTokenType::Material => {
                        if let Some(bsdf) = BSDF::new(named_token, true) {
                            // Create a fake name...
                            let name = format!(
                                "unamed_material_{}",
                                scene_info.number_unamed_materials
                            );
                            scene_info.number_unamed_materials += 1;
                            scene_info.materials.insert(name.to_string(), bsdf);
                            state.set_named_matrial(name);
                        }
                    }
                    NamedTokenType::Shape => {
                        if let Some(shape) = Shape::new(named_token, working_dir) {
                            let mut shape = ShapeInfo::new(shape, state.matrix());
                            shape.material_name = state.named_material();
                            shape.emission = state.emission();
                            match &mut state.object {
                                Some(o) => {
                                    info!("Added inside an object: {}", o.name);
                                    o.shapes.push(shape)
                                }
                                None => {
                                    info!("Put inside scene_info");
                                    scene_info.shapes.push(shape)
                                }
                            };
                        }
                    }
                    NamedTokenType::Film => {
                        scene_info.image_size = Vector2::new(
                            named_token.values.remove("xresolution").unwrap().into_integer()[0] as u32,
                            named_token.values.remove("yresolution").unwrap().into_integer()[0] as u32,
                        );
                    }
                    NamedTokenType::AreaLightSource => {
                        match &named_token.internal_type[..] {
                            "diffuse" => {
                                if let Some(e) = named_token.values.remove("L") {
                                    state.set_emission(e.into_spectrum());
                                }
                            }
                            _ => warn!("Unsuppored area light: {}", named_token.internal_type),
                        }
                    }
                    NamedTokenType::LightSource => {
                        if let Some(light) = Light::new(named_token) {
                            scene_info.lights.push(light);
                        }
                    }
                    NamedTokenType::CoordSysTransform => {
                        state.replace_matrix(*scene_info.transforms.get(&named_token.internal_type).unwrap());
                    }
                    NamedTokenType::CoordSys => {
                        scene_info
                        .transforms
                        .insert(named_token.internal_type, state.matrix.last().unwrap().clone());
                    }
                    NamedTokenType::Include => {
                        info!("Include found: {}", named_token.internal_type);
                        let filename = working_dir.join(named_token.internal_type);
                        read_pbrt_file_wk(
                            filename.to_str().unwrap(),
                            scene_info,
                            state,
                            working_dir,
                        );
                    }
                    NamedTokenType::ObjectBegin => {
                        state.new_object(named_token.internal_type);
                    }
                    NamedTokenType::ObjectInstance => {
                        scene_info.instances.push(
                            InstanceInfo {
                                matrix: state.matrix(),
                                name: named_token.internal_type
                            }
                        )
                    }
                    _ => warn!("{:?} not implemented", named_token.object_type),
                }
            }
        }
    }
}

/**
 * Main method to read a PBRT file
 */
pub fn read_pbrt_file(path: &str, scene_info: &mut Scene, state: &mut State) {
    let now = Instant::now();
    info!("Loading: {}", path);
    let working_dir = std::path::Path::new(path).parent().unwrap();
    let mut file =
        std::fs::File::open(path).unwrap_or_else(|_| panic!("Impossible to open {}", path));
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    read_pbrt(&contents, working_dir, scene_info, state);
    info!("Time for parsing file: {:?}", Instant::now() - now);
}

fn read_pbrt_file_wk(
    path: &str,
    scene_info: &mut Scene,
    state: &mut State,
    working_dir: &std::path::Path,
) {
    let now = Instant::now();
    info!("Loading: {}", path);
    let mut file =
        std::fs::File::open(path).unwrap_or_else(|_| panic!("Impossible to open {}", path));
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    read_pbrt(&contents, working_dir, scene_info, state);
    info!("Time for parsing file: {:?}", Instant::now() - now);
}
