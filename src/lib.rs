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

/// BSDF representation
// pub struct MatteBSDF {
//     pub kd: Param,
//     pub sigma: Param,
//     pub bumpmap: Option<Param>,
// }
// pub struct MetalBSDF {
//     pub eta: Param,
//     pub k: Param,
//     pub roughness: Param,
//     pub u_roughness: Option<Param>,
//     pub v_roughness: Option<Param>,
//     pub bumpmap: Option<Param>,
//     pub remap_roughness: bool,
// }
// pub struct SubstrateBSDF {
//     pub kd: Param,
//     pub ks: Param,
//     pub u_roughness: Param,
//     pub v_roughness: Param,
//     pub bumpmap: Option<Param>,
//     pub remap_roughness: bool,
// }
// pub struct GlassBSDF {
//     pub kr: Param,
//     pub kt: Param,
//     pub u_roughness: Param,
//     pub v_roughness: Param,
//     pub index: Param,
//     pub bumpmap: Option<Param>,
//     pub remap_roughness: bool,
// }
// pub struct MirrorBSDF {
//     pub kr: Param,
//     pub bumpmap: Option<Param>,
// }
pub enum BSDF {
    // Matte(MatteBSDF),
// Metal(MetalBSDF),
// Substrate(SubstrateBSDF),
// Glass(GlassBSDF),
// Mirror(MirrorBSDF),
}
impl BSDF {
    // fn new(pairs: pest::iterators::Pair<Rule>, unamed: bool) -> Option<(String, Self)> {
    //     let (name, mut param) = parse_parameters(pairs);
    //     // TODO: Need to clone to avoid borrower checker
    //     let bsdf_type = if unamed {
    //         name.clone()
    //     } else {
    //         param
    //             .remove("type")
    //             .expect("bsdf type param is required")
    //             .into_name()
    //     };
    //     match bsdf_type.as_ref() {
    //         "matte" => {
    //             let kd = remove_default!(param, "Kd", Param::RGB(RGBValue::color(0.5)));
    //             let sigma = remove_default!(param, "sigma", Param::Float(vec![0.0]));
    //             let bumpmap = param.remove("bumpmap");
    //             if !param.is_empty() {
    //                 panic!("Miss parameters for Matte: {} => {:?}", name, param);
    //             }
    //             Some((name, BSDF::Matte(MatteBSDF { kd, sigma, bumpmap })))
    //         }
    //         "metal" => {
    //             // TODO: Need to be able to export other material params
    //             let eta = remove_default!(
    //                 param,
    //                 "eta",
    //                 Param::RGB(RGBValue {
    //                     r: 0.199_990_69,
    //                     g: 0.922_084_6,
    //                     b: 1.099_875_9
    //                 })
    //             );
    //             let k = remove_default!(
    //                 param,
    //                 "k",
    //                 Param::RGB(RGBValue {
    //                     r: 3.904_635_4,
    //                     g: 2.447_633_3,
    //                     b: 2.137_652_6
    //                 })
    //             );
    //             let roughness = remove_default!(param, "roughness", Param::Float(vec![0.1]));
    //             let u_roughness = param.remove("uroughness");
    //             let v_roughness = param.remove("vroughness");
    //             let bumpmap = param.remove("bumpmap");
    //             let remap_roughness =
    //                 remove_default!(param, "remaproughness", Param::Bool(true)).into_bool();
    //             if !param.is_empty() {
    //                 warn!("Miss parameters for Metal: {} => {:?}", name, param);
    //             }
    //             Some((
    //                 name,
    //                 BSDF::Metal(MetalBSDF {
    //                     eta,
    //                     k,
    //                     roughness,
    //                     u_roughness,
    //                     v_roughness,
    //                     bumpmap,
    //                     remap_roughness,
    //                 }),
    //             ))
    //         }
    //         "substrate" => {
    //             let kd = remove_default!(param, "Kd", Param::RGB(RGBValue::color(0.5)));
    //             let ks = remove_default!(param, "Ks", Param::RGB(RGBValue::color(0.5)));
    //             let u_roughness = remove_default!(param, "uroughness", Param::Float(vec![0.1]));
    //             let v_roughness = remove_default!(param, "vroughness", Param::Float(vec![0.1]));
    //             let bumpmap = param.remove("bumpmap");
    //             let remap_roughness =
    //                 remove_default!(param, "remaproughness", Param::Bool(true)).into_bool();
    //             if !param.is_empty() {
    //                 warn!("Miss parameters for Substrate: {} => {:?}", name, param);
    //             }
    //             Some((
    //                 name,
    //                 BSDF::Substrate(SubstrateBSDF {
    //                     kd,
    //                     ks,
    //                     u_roughness,
    //                     v_roughness,
    //                     bumpmap,
    //                     remap_roughness,
    //                 }),
    //             ))
    //         }
    //         "glass" => {
    //             let kr = remove_default!(param, "Kr", Param::RGB(RGBValue::color(1.0)));
    //             let kt = remove_default!(param, "Kt", Param::RGB(RGBValue::color(1.0)));
    //             let u_roughness = remove_default!(param, "uroughness", Param::Float(vec![0.0]));
    //             let v_roughness = remove_default!(param, "vroughness", Param::Float(vec![0.0]));
    //             let bumpmap = param.remove("bumpmap");
    //             let remap_roughness =
    //                 remove_default!(param, "remaproughness", Param::Bool(true)).into_bool();
    //             let index = if let Some(eta) = param.remove("eta") {
    //                 eta
    //             } else {
    //                 remove_default!(param, "index", Param::Float(vec![1.5]))
    //             };
    //             if !param.is_empty() {
    //                 warn!("Miss parameters for Glass: {} => {:?}", name, param);
    //             }
    //             Some((
    //                 name,
    //                 BSDF::Glass(GlassBSDF {
    //                     kr,
    //                     kt,
    //                     u_roughness,
    //                     v_roughness,
    //                     index,
    //                     bumpmap,
    //                     remap_roughness,
    //                 }),
    //             ))
    //         }
    //         "mirror" => {
    //             let kr = remove_default!(param, "Kr", Param::RGB(RGBValue::color(1.0)));
    //             let bumpmap = param.remove("bumpmap");
    //             Some((name, BSDF::Mirror(MirrorBSDF { kr, bumpmap })))
    //         }
    //         _ => {
    //             warn!("BSDF case with {} is not cover", bsdf_type);
    //             None
    //         }
    //     }
    // }
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
    fn new(mut named_token: NamedToken, wk: Option<&std::path::Path>) -> Option<Self> {
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
                    Some(v.into_vector2())
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
                let wk = match wk {
                    None => panic!("Plymesh is not supported without wk specified"),
                    Some(ref v) => v,
                };
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
    working_dir: Option<&std::path::Path>,
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
                    assert_eq!(class, "imagemap");
                    // Check type as Well... {spectrum or float} -> Two lists

                    // TODO: A lot of parameters...
                    scene_info.textures.insert(name, Texture {
                        filename: values.remove("filename").unwrap().into_string(),
                        trilinear: remove_default!(values, "trilinear", Value::Boolean(false)).into_bool(),
                    });
                },
            Token::NamedToken(mut named_token) => {
                // pub enum NamedTokenType {
                //     MakeNamedMedium,
                //     ObjectInstance,
                //     ObjectBegin,
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
                        todo!();
                    }
                    NamedTokenType::NamedMaterial => {
                        // if let Some((name, mat)) = BSDF::new(rule_pair, false) {
                        //     scene_info.materials.insert(name, mat);
                        // }
                        assert!(named_token.values.is_empty());
                        state.set_named_matrial(named_token.internal_type);
                    }
                    NamedTokenType::Material => {
                        todo!();
                        // Rule::material => {
                        //     if let Some((_, mat)) = BSDF::new(rule_pair, true) {
                        //         let name = format!(
                        //             "unamed_material_{}",
                        //             scene_info.number_unamed_materials
                        //         );
                        //         scene_info.number_unamed_materials += 1;
                        //         scene_info.materials.insert(name.to_string(), mat);
                        //         state.set_named_matrial(name);
                        //     }
                        // }
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
                        let wk = working_dir.as_ref().unwrap();
                        info!("Include found: {}", named_token.internal_type);
                        let filename = wk.join(named_token.internal_type);
                        read_pbrt_file(
                            filename.to_str().unwrap(),
                            working_dir,
                            scene_info,
                            state,
                        );
                    }
                    _ => todo!("{:?} not implemented", named_token.object_type),
                }
            }
        }
    }
}

/**
 * Main method to read a PBRT file
 */
pub fn read_pbrt_file(
    path: &str,
    working_dir: Option<&std::path::Path>,
    scene_info: &mut Scene,
    state: &mut State,
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
