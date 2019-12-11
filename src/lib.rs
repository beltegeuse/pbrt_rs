#![allow(dead_code)]
#![allow(clippy::cognitive_complexity)]

// For logging
#[macro_use]
extern crate log;
// For parsing
extern crate pest;
#[macro_use]
extern crate pest_derive;
// Vector representation
extern crate cgmath;
// For loading ply files
extern crate ply_rs;

// parser
use cgmath::*;
use pest::Parser;
use ply_rs::parser;
use ply_rs::ply;
use std::collections::HashMap;
use std::io::Read;
use std::rc::Rc;
use std::str::FromStr;
use std::time::Instant;

const _GRAMMAR: &str = include_str!("pbrt.pest");

#[derive(Parser)]
#[grammar = "pbrt.pest"]
struct PbrtParser;
fn pbrt_matrix(pairs: pest::iterators::Pairs<Rule>) -> Vec<f32> {
    let mut m: Vec<f32> = Vec::new();
    for rule_pair in pairs {
        // ignore brackets
        let not_opening: bool = rule_pair.as_str() != "[";
        let not_closing: bool = rule_pair.as_str() != "]";
        if not_opening && not_closing {
            let number = f32::from_str(rule_pair.clone().as_span().as_str()).unwrap();
            m.push(number);
        }
    }
    m
}
fn pbrt_parameter<T: FromStr>(pairs: &mut pest::iterators::Pairs<Rule>) -> (String, Vec<T>)
where
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    let mut values: Vec<T> = Vec::new();
    // single float or several floats using brackets
    let ident = pairs.next();
    let name = String::from(ident.unwrap().clone().as_span().as_str());
    let option = pairs.next();
    let lbrack = option.clone().unwrap();
    if lbrack.as_str() == "[" {
        // check for brackets
        let mut number = pairs.next();
        while number.is_some() {
            let pair = number.unwrap().clone();
            if pair.as_str() == "]" {
                // closing bracket found
                break;
            } else {
                let value = pair.as_span().as_str();
                // TODO: Only necessary for some of the names... might impact the performance of the parser
                let value = value.trim_matches('\"').to_string();
                let value = value
                    .parse::<T>()
                    .unwrap_or_else(|_| panic!("parsing error on parameter: {}", value));
                values.push(value);
            }
            number = pairs.next();
        }
    } else {
        // no brackets
        let mut number = option.clone();
        while number.is_some() {
            let pair = number.unwrap().clone();
            let value = pair
                .as_span()
                .as_str()
                .parse::<T>()
                .expect("parsing error on parameter");
            values.push(value);
            number = pairs.next();
        }
    }
    (name, values)
}

struct PlyFace {
    vertex_index: Vec<i32>,
}
impl ply::PropertyAccess for PlyFace {
    fn new() -> Self {
        PlyFace {
            vertex_index: Vec::new(),
        }
    }
    fn set_property(&mut self, key: String, property: ply::Property) {
        match (key.as_ref(), property) {
            ("vertex_indices", ply::Property::ListInt(vec)) => self.vertex_index = vec,
            ("vertex_indices", ply::Property::ListUInt(vec)) => {
                self.vertex_index = vec![0; vec.len()];
                for (i, v) in vec.iter().enumerate() {
                    self.vertex_index[i] = *v as i32;
                }
            }
            ("vertex_indices", ply::Property::ListUChar(vec)) => {
                self.vertex_index = vec![0; vec.len()];
                for (i, v) in vec.iter().enumerate() {
                    self.vertex_index[i] = i32::from(*v);
                }
            }
            (k, _) => panic!("Face: Unexpected key/value combination: key: {}", k),
        }
    }
}
struct PlyVertex {
    pos: Vector3<f32>,
    normal: Vector3<f32>,
    uv: Vector2<f32>,
    has_normal: bool,
    has_uv: bool,
}
impl ply::PropertyAccess for PlyVertex {
    fn new() -> Self {
        PlyVertex {
            pos: Vector3::new(0.0, 0.0, 0.0),
            normal: Vector3::new(0.0, 0.0, 0.0),
            uv: Vector2::new(0.0, 0.0),
            has_normal: false,
            has_uv: false,
        }
    }

    fn set_property(&mut self, key: String, property: ply::Property) {
        match (key.as_ref(), property) {
            ("x", ply::Property::Float(v)) => self.pos.x = v,
            ("y", ply::Property::Float(v)) => self.pos.y = v,
            ("z", ply::Property::Float(v)) => self.pos.z = v,
            ("nx", ply::Property::Float(v)) => {
                self.has_normal = true;
                self.normal.x = v
            }
            ("ny", ply::Property::Float(v)) => {
                self.has_normal = true;
                self.normal.y = v
            }
            ("nz", ply::Property::Float(v)) => {
                self.has_normal = true;
                self.normal.z = v
            }
            ("u", ply::Property::Float(v)) => {
                self.has_uv = true;
                self.uv.x = v
            }
            ("v", ply::Property::Float(v)) => {
                self.has_uv = true;
                self.uv.y = v
            }
            ("s", ply::Property::Float(v)) => {
                self.has_uv = true;
                self.uv.x = v
            }
            ("t", ply::Property::Float(v)) => {
                self.has_uv = true;
                self.uv.y = v
            }
            (k, _) => panic!("Face: Unexpected key/value combination: key: {}", k),
        }
    }
}

/// Intermediate representation
/// for parsing the parameters
#[derive(Debug, Clone)]
pub struct RGBValue {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}
impl RGBValue {
    pub fn color(v: f32) -> RGBValue {
        RGBValue { r: v, g: v, b: v }
    }
}
#[derive(Debug, Clone)]
pub enum Param {
    Integer(Vec<i32>),
    Float(Vec<f32>),
    Vector3(Vec<Vector3<f32>>),
    Vector2(Vec<Vector2<f32>>),
    Name(String),
    RGB(RGBValue),
    Bool(bool),
}
impl Param {
    fn into_float(self) -> Vec<f32> {
        match self {
            Param::Float(v) => v,
            _ => panic!("impossible to convert to float: {:?}", self),
        }
    }
    fn parse_float(pairs: &mut pest::iterators::Pairs<Rule>) -> (String, Self) {
        let (name, values) = pbrt_parameter(pairs);
        (name, Param::Float(values))
    }

    fn into_name(self) -> String {
        match self {
            Param::Name(v) => v,
            _ => panic!("impossible to convert to name: {:?}", self),
        }
    }
    fn parse_name(pairs: &mut pest::iterators::Pairs<Rule>) -> (String, Self) {
        let (name, values) = pbrt_parameter::<String>(pairs);
        let values = values[0].clone(); // TODO
        let values = values.trim_matches('\"').to_string();
        (name, Param::Name(values))
    }

    fn into_rgb(self) -> RGBValue {
        match self {
            Param::RGB(rgb) => rgb,
            _ => panic!("impossible to convert to rgb: {:?}", self),
        }
    }
    fn parse_rgb(pairs: &mut pest::iterators::Pairs<Rule>) -> (String, Self) {
        let (name, values) = pbrt_parameter::<f32>(pairs);
        (
            name,
            Param::RGB(RGBValue {
                r: values[0],
                g: values[1],
                b: values[2],
            }),
        )
    }

    fn into_integer(self) -> Vec<i32> {
        match self {
            Param::Integer(v) => v,
            _ => panic!("impossible to convert to integer: {:?}", self),
        }
    }
    fn parse_integer(pairs: &mut pest::iterators::Pairs<Rule>) -> (String, Self) {
        let (name, values) = pbrt_parameter::<i32>(pairs);
        (name, Param::Integer(values))
    }

    fn into_vector3(self) -> Vec<Vector3<f32>> {
        match self {
            Param::Vector3(v) => v,
            _ => panic!("impossible to convert to integer: {:?}", self),
        }
    }
    fn parse_vector3(pairs: &mut pest::iterators::Pairs<Rule>) -> (String, Self) {
        let (name, values) = pbrt_parameter::<f32>(pairs);
        if values.len() % 3 != 0 {
            panic!("Non 3 multiples for vector 3");
        }
        let values = values
            .chunks(3)
            .map(|v| Vector3::new(v[0], v[1], v[2]))
            .collect();
        (name, Param::Vector3(values))
    }

    fn into_vector2(self) -> Vec<Vector2<f32>> {
        match self {
            Param::Vector2(v) => v,
            _ => panic!("impossible to convert to integer: {:?}", self),
        }
    }
    fn parse_vector2(pairs: &mut pest::iterators::Pairs<Rule>) -> (String, Self) {
        let (name, values) = pbrt_parameter::<f32>(pairs);
        if values.len() % 2 != 0 {
            panic!("Non 2 multiples for point 2");
        }
        let values = values.chunks(2).map(|v| Vector2::new(v[0], v[1])).collect();
        (name, Param::Vector2(values))
    }

    fn into_bool(self) -> bool {
        match self {
            Param::Bool(v) => v,
            _ => panic!("impossible to convert to bool: {:?}", self),
        }
    }
    fn parse_bool(pairs: &mut pest::iterators::Pairs<Rule>) -> (String, Self) {
        let (name, values) = pbrt_parameter::<bool>(pairs);
        let values = values[0];
        (name, Param::Bool(values))
    }
}

fn parse_parameters(pairs: pest::iterators::Pair<Rule>) -> (String, HashMap<String, Param>) {
    let mut name = vec![];
    let mut param_map: HashMap<String, Param> = HashMap::default();
    for pair in pairs.into_inner() {
        match pair.as_rule() {
            Rule::empty_string => {}
            Rule::string => {
                let mut string_pairs = pair.into_inner();
                let ident = string_pairs.next();
                name.push(String::from_str(ident.unwrap().clone().as_span().as_str()).unwrap());
            }
            Rule::parameter => {
                for parameter_pair in pair.into_inner() {
                    match parameter_pair.as_rule() {
                        Rule::float_param => {
                            let (name, value) =
                                Param::parse_float(&mut parameter_pair.into_inner());
                            param_map.insert(name, value);
                        }
                        Rule::bool_param => {
                            let (name, value) = Param::parse_bool(&mut parameter_pair.into_inner());
                            param_map.insert(name, value);
                        }
                        Rule::string_param | Rule::texture_param => {
                            let (name, value) = Param::parse_name(&mut parameter_pair.into_inner());
                            param_map.insert(name, value);
                        }
                        Rule::rgb_param => {
                            let (name, value) = Param::parse_rgb(&mut parameter_pair.into_inner());
                            param_map.insert(name, value);
                        }
                        Rule::integer_param => {
                            let (name, value) =
                                Param::parse_integer(&mut parameter_pair.into_inner());
                            param_map.insert(name, value);
                        }
                        Rule::point_param | Rule::normal_param => {
                            let (name, value) =
                                Param::parse_vector3(&mut parameter_pair.into_inner());
                            param_map.insert(name, value);
                        }
                        Rule::point2_param => {
                            let (name, value) =
                                Param::parse_vector2(&mut parameter_pair.into_inner());
                            param_map.insert(name, value);
                        }
                        _ => warn!("Ignoring Parameter: {}", parameter_pair.as_str()),
                    }
                }
            }
            _ => warn!("Ignoring: {}", pair.as_str()),
        }
    }
    if !name.is_empty() {
        (name[0].clone(), param_map)
    } else {
        panic!("Parse parameter, name is not provided");
    }
}

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
pub struct CameraPerspective {
    pub fov: f32,
    pub world_to_camera: Matrix4<f32>,
}
pub enum Camera {
    Perspective(CameraPerspective),
}
impl Camera {
    fn new(pairs: pest::iterators::Pair<Rule>, mat: Matrix4<f32>) -> Option<Self> {
        let (name, mut param) = parse_parameters(pairs);
        match name.as_ref() {
            "perspective" => {
                let fov = param.remove("fov").expect("fov is not given").into_float()[0];
                Some(Camera::Perspective(CameraPerspective {
                    fov,
                    world_to_camera: mat,
                }))
            }
            _ => {
                warn!("Camera case with {} is not cover", name);
                None
            }
        }
    }
}

//// Texture representation
pub struct Texture {
    pub filename: String,
    pub trilinear: bool,
}
impl Texture {
    fn new(pairs: pest::iterators::Pair<Rule>, wk: &std::path::Path) -> Option<(String, Self)> {
        let (name, mut param) = parse_parameters(pairs);

        if let Some(filename) = param.remove("filename") {
            let trilinear = remove_default!(param, "trilinear", Param::Bool(true)).into_bool();
            Some((
                name,
                Texture {
                    filename: wk.join(filename.into_name()).to_str().unwrap().to_string(),
                    trilinear,
                },
            ))
        } else {
            warn!("Unsupported texture: {}", name);
            None
        }
    }
}

/// BSDF representation
pub struct MatteBSDF {
    pub kd: Param,
    pub sigma: Param,
    pub bumpmap: Option<Param>,
}
pub struct MetalBSDF {
    pub eta: Param,
    pub k: Param,
    pub roughness: Param,
    pub u_roughness: Option<Param>,
    pub v_roughness: Option<Param>,
    pub bumpmap: Option<Param>,
    pub remap_roughness: bool,
}
pub struct SubstrateBSDF {
    pub kd: Param,
    pub ks: Param,
    pub u_roughness: Param,
    pub v_roughness: Param,
    pub bumpmap: Option<Param>,
    pub remap_roughness: bool,
}
pub struct GlassBSDF {
    pub kr: Param,
    pub kt: Param,
    pub u_roughness: Param,
    pub v_roughness: Param,
    pub index: Param,
    pub bumpmap: Option<Param>,
    pub remap_roughness: bool,
}
pub struct MirrorBSDF {
    pub kr: Param,
    pub bumpmap: Option<Param>,
}
pub enum BSDF {
    Matte(MatteBSDF),
    Metal(MetalBSDF),
    Substrate(SubstrateBSDF),
    Glass(GlassBSDF),
    Mirror(MirrorBSDF),
}
impl BSDF {
    fn new(pairs: pest::iterators::Pair<Rule>, unamed: bool) -> Option<(String, Self)> {
        let (name, mut param) = parse_parameters(pairs);
        // TODO: Need to clone to avoid borrower checker
        let bsdf_type = if unamed {
            name.clone()
        } else {
            param
                .remove("type")
                .expect("bsdf type param is required")
                .into_name()
        };
        match bsdf_type.as_ref() {
            "matte" => {
                let kd = remove_default!(param, "Kd", Param::RGB(RGBValue::color(0.5)));
                let sigma = remove_default!(param, "sigma", Param::Float(vec![0.0]));
                let bumpmap = param.remove("bumpmap");
                if !param.is_empty() {
                    panic!("Miss parameters for Matte: {} => {:?}", name, param);
                }
                Some((name, BSDF::Matte(MatteBSDF { kd, sigma, bumpmap })))
            }
            "metal" => {
                // TODO: Need to be able to export other material params
                let eta = remove_default!(
                    param,
                    "eta",
                    Param::RGB(RGBValue {
                        r: 0.199_990_69,
                        g: 0.922_084_6,
                        b: 1.099_875_9
                    })
                );
                let k = remove_default!(
                    param,
                    "k",
                    Param::RGB(RGBValue {
                        r: 3.904_635_4,
                        g: 2.447_633_3,
                        b: 2.137_652_6
                    })
                );
                let roughness = remove_default!(param, "roughness", Param::Float(vec![0.1]));
                let u_roughness = param.remove("uroughness");
                let v_roughness = param.remove("vroughness");
                let bumpmap = param.remove("bumpmap");
                let remap_roughness =
                    remove_default!(param, "remaproughness", Param::Bool(true)).into_bool();
                if !param.is_empty() {
                    warn!("Miss parameters for Metal: {} => {:?}", name, param);
                }
                Some((
                    name,
                    BSDF::Metal(MetalBSDF {
                        eta,
                        k,
                        roughness,
                        u_roughness,
                        v_roughness,
                        bumpmap,
                        remap_roughness,
                    }),
                ))
            }
            "substrate" => {
                let kd = remove_default!(param, "Kd", Param::RGB(RGBValue::color(0.5)));
                let ks = remove_default!(param, "Ks", Param::RGB(RGBValue::color(0.5)));
                let u_roughness = remove_default!(param, "uroughness", Param::Float(vec![0.1]));
                let v_roughness = remove_default!(param, "vroughness", Param::Float(vec![0.1]));
                let bumpmap = param.remove("bumpmap");
                let remap_roughness =
                    remove_default!(param, "remaproughness", Param::Bool(true)).into_bool();
                if !param.is_empty() {
                    warn!("Miss parameters for Substrate: {} => {:?}", name, param);
                }
                Some((
                    name,
                    BSDF::Substrate(SubstrateBSDF {
                        kd,
                        ks,
                        u_roughness,
                        v_roughness,
                        bumpmap,
                        remap_roughness,
                    }),
                ))
            }
            "glass" => {
                let kr = remove_default!(param, "Kr", Param::RGB(RGBValue::color(1.0)));
                let kt = remove_default!(param, "Kt", Param::RGB(RGBValue::color(1.0)));
                let u_roughness = remove_default!(param, "uroughness", Param::Float(vec![0.1]));
                let v_roughness = remove_default!(param, "vroughness", Param::Float(vec![0.1]));
                let bumpmap = param.remove("bumpmap");
                let remap_roughness =
                    remove_default!(param, "remaproughness", Param::Bool(true)).into_bool();
                let index = if let Some(eta) = param.remove("eta") {
                    eta
                } else {
                    remove_default!(param, "index", Param::Float(vec![1.5]))
                };
                if !param.is_empty() {
                    warn!("Miss parameters for Glass: {} => {:?}", name, param);
                }
                Some((
                    name,
                    BSDF::Glass(GlassBSDF {
                        kr,
                        kt,
                        u_roughness,
                        v_roughness,
                        index,
                        bumpmap,
                        remap_roughness,
                    }),
                ))
            }
            "mirror" => {
                let kr = remove_default!(param, "Kr", Param::RGB(RGBValue::color(1.0)));
                let bumpmap = param.remove("bumpmap");
                Some((name, BSDF::Mirror(MirrorBSDF { kr, bumpmap })))
            }
            _ => {
                warn!("BSDF case with {} is not cover", bsdf_type);
                None
            }
        }
    }
}

/// Mesh representation
#[derive(Debug)]
pub struct TriMeshShape {
    pub indices: Vec<Vector3<usize>>,
    pub points: Vec<Point3<f32>>,
    pub normals: Option<Vec<Vector3<f32>>>,
    pub uv: Option<Vec<Vector2<f32>>>,
}
#[derive(Debug)]
pub enum Shape {
    TriMesh(TriMeshShape),
}
impl Shape {
    fn new(pairs: pest::iterators::Pair<Rule>, wk: &std::path::Path) -> Option<(String, Self)> {
        let (name, mut param) = parse_parameters(pairs);
        match name.as_ref() {
            "trianglemesh" => {
                let points = param.remove("P").expect("P is required").into_vector3();
                let points = points.into_iter().map(|v| Point3::from_vec(v)).collect();
                let indices = param
                    .remove("indices")
                    .expect("indice is required")
                    .into_integer();
                if indices.len() % 3 != 0 {
                    panic!("Support only 3 indices list");
                }
                let indices = indices
                    .chunks(3)
                    .map(|v| Vector3::new(v[0] as usize, v[1] as usize, v[2] as usize))
                    .collect();
                let normals = if let Some(v) = param.remove("N") {
                    Some(v.into_vector3())
                } else {
                    None
                };
                let uv = if let Some(v) = param.remove("uv") {
                    Some(
                        v.into_float()
                            .chunks(2)
                            .map(|v| Vector2::new(v[0], v[1]))
                            .collect(),
                    )
                } else {
                    None
                };
                Some((
                    name,
                    Shape::TriMesh(TriMeshShape {
                        indices,
                        points,
                        normals,
                        uv,
                    }),
                ))
            }
            "plymesh" => {
                let filename = param
                    .remove("filename")
                    .expect("filename is required")
                    .into_name();
                info!("Reading {} ...", filename);
                let filename = wk.join(filename);
                let f = match std::fs::File::open(filename.clone()) {
                    Ok(f) => f,
                    Err(e) => {
                        panic!("Error in opening: {:?} [wk: {:?}] => {:?}", filename, wk, e);
                    }
                };
                let mut f = std::io::BufReader::new(f);
                // create parsers
                let vertex_parser = parser::Parser::<PlyVertex>::new();
                let face_parser = parser::Parser::<PlyFace>::new();
                // read the header
                let header = vertex_parser.read_header(&mut f).unwrap();
                // TODO: Safely unwrap
                let mut vertex_list = Vec::new();
                let mut face_list = Vec::new();
                for (_ignore_key, element) in &header.elements {
                    // we could also just parse them in sequence, but the file format might change
                    match element.name.as_ref() {
                        "vertex" => {
                            vertex_list = vertex_parser
                                .read_payload_for_element(&mut f, &element, &header)
                                .unwrap();
                        }
                        "face" => {
                            face_list = face_parser
                                .read_payload_for_element(&mut f, &element, &header)
                                .unwrap();
                        }
                        _ => panic!("Unexpeced element!"),
                    }
                }
                info!(" - #vertex: {}", vertex_list.len());
                info!(" - #face: {}", face_list.len());
                let mut indices = Vec::new();
                for f in face_list {
                    if f.vertex_index.len() == 3 {
                        indices.push(Vector3::new(
                            f.vertex_index[0] as usize,
                            f.vertex_index[1] as usize,
                            f.vertex_index[2] as usize,
                        ));
                    } else if f.vertex_index.len() == 4 {
                        // Quad is detected
                        let quad_indices = f
                            .vertex_index
                            .into_iter()
                            .map(|v| v as usize)
                            .collect::<Vec<usize>>();
                        indices.push(Vector3::new(quad_indices[0], quad_indices[1], quad_indices[2]));
                        indices.push(Vector3::new(quad_indices[2], quad_indices[3], quad_indices[0]));
                    } else {
                    }
                }
                let normals = if vertex_list[0].has_normal {
                    Some(vertex_list.iter().map(|v| v.normal).collect())
                } else {
                    None
                };
                let uv = if vertex_list[0].has_uv {
                    Some(vertex_list.iter().map(|v| v.uv).collect())
                } else {
                    None
                };
                let vertex_list = vertex_list
                    .into_iter()
                    .map(|v| Point3::from_vec(v.pos))
                    .collect();

                Some((
                    name,
                    Shape::TriMesh(TriMeshShape {
                        indices,
                        points: vertex_list,
                        normals,
                        uv,
                    }),
                ))
            }
            _ => {
                warn!("Shape case with {} is not cover", name);
                None
            }
        }
    }
}

/// Lights
#[derive(Debug)]
pub struct DistantLight {
    pub luminance: Param,
    pub from: Point3<f32>,
    pub to: Point3<f32>,
    pub scale: RGBValue,
}
#[derive(Debug)]
pub struct InfiniteLight {
    pub luminance: Param, // Can be RGB or map
    pub samples: u32,
    pub scale: RGBValue,
}
#[derive(Debug)]
pub struct PointLight {
    pub intensity: RGBValue,
    pub from: Point3<f32>,
    pub scale: RGBValue,
}
#[derive(Debug)]
pub enum Light {
    Distant(DistantLight),
    Infinite(InfiniteLight),
    Point(PointLight),
}
impl Light {
    fn new(pairs: pest::iterators::Pair<Rule>) -> Option<Self> {
        let (name, mut param) = parse_parameters(pairs);
        info!("Reading light source: {}", name);
        let scale = if let Some(scale) = param.remove("scale") {
            let s = scale.into_rgb();
            info!(" - Scale: {:?}", s);
            s
        } else {
            RGBValue::color(1.0)
        };

        match name.as_ref() {
            "infinite" => {
                let samples = if let Some(samples) = param.remove("samples") {
                    samples.into_integer()[0] as u32
                } else {
                    1
                };
                let luminance = if let Some(luminance) = param.remove("L") {
                    luminance
                } else {
                    Param::RGB(RGBValue::color(1.0))
                };
                let luminance = if let Some(mapname) = param.remove("mapname") {
                    mapname
                } else {
                    luminance
                };
                Some(Light::Infinite(InfiniteLight {
                    luminance,
                    samples,
                    scale,
                }))
            }
            _ => {
                warn!("Light case with {} is not cover", name);
                None
            }
        }
    }
}
/// State of the parser
#[derive(Clone, Debug)]
pub struct State {
    named_material: Vec<Option<String>>,
    matrix: Vec<Matrix4<f32>>,
    emission: Vec<Option<Param>>,
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
    fn emission(&self) -> Option<Param> {
        self.emission.last().unwrap().clone()
    }
    fn set_emission(&mut self, e: Param) {
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
    pub emission: Option<Param>,
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
pub struct InstanceInfo {
    pub matrix: Matrix4<f32>,
    pub object: Rc<ObjectInfo>,
}
#[derive(Clone, Debug)]
pub struct ObjectInfo {
    pub name: String,
    pub shapes: Vec<Rc<ShapeInfo>>,
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
    pub shapes: Vec<Rc<ShapeInfo>>,               //< unamed shapes
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

pub fn read_pbrt_file(
    path: &str,
    working_dir: &std::path::Path,
    scene_info: &mut Scene,
    state: &mut State,
) {
    let now = Instant::now();
    info!("Loading: {}", path);
    let file = std::fs::File::open(path).unwrap_or_else(|_| panic!("Impossible to open {}", path));
    let mut reader = std::io::BufReader::new(file);
    let mut str_buf: String = String::default();
    let _num_bytes = reader.read_to_string(&mut str_buf);
    info!("Time for reading file: {:?}", Instant::now() - now);

    let now = Instant::now();
    let pairs =
        PbrtParser::parse(Rule::pbrt, &str_buf).unwrap_or_else(|e| panic!("Parsing error: {}", e));
    for pair in pairs {
        let span = pair.clone().as_span();
        debug!("Rule:    {:?}", pair.as_rule());
        debug!("Span:    {:?}", span);
        debug!("Text:    {}", span.as_str());
        for inner_pair in pair.into_inner() {
            debug!("Inner Rule:    {:?}", inner_pair.as_rule());
            debug!("Inner Text:    {}", inner_pair.as_str());
            match inner_pair.as_rule() {
                Rule::transform => {
                    // FIMXE: Does the rule replace the transformation?
                    let values = pbrt_matrix(inner_pair.into_inner());
                    if values.len() != 16 {
                        panic!("Transform need to have 16 floats: {:?}", values);
                    }
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
                }
                Rule::concat_transform => {
                    let values = pbrt_matrix(inner_pair.into_inner());
                    if values.len() != 16 {
                        panic!("Transform need to have 16 floats: {:?}", values);
                    }
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
                }
                Rule::scale => {
                    let values = pbrt_matrix(inner_pair.into_inner());
                    if values.len() != 3 {
                        panic!("Scale need to have 3 floats: {:?}", values);
                    }
                    let matrix = state.matrix()
                        * Matrix4::from_diagonal(Vector4::new(
                            values[0], values[1], values[2], 1.0,
                        ));
                    state.replace_matrix(matrix);
                }
                Rule::look_at => {
                    let values = pbrt_matrix(inner_pair.into_inner());
                    if values.len() != 9 {
                        panic!("LookAt need to have 9 floats: {:?}", values);
                    }
                    let eye = Point3::new(values[0], values[1], values[2]);
                    let target = Point3::new(values[3], values[4], values[5]);
                    let up = Vector3::new(values[6], values[7], values[8]);

                    let dir = (target - eye).normalize();
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
                    info!("After lookat: {:?}", state.matrix());
                }
                Rule::translate => {
                    let values = pbrt_matrix(inner_pair.into_inner());
                    if values.len() != 3 {
                        panic!("Translate need to have 3 floats: {:?}", values);
                    }
                    let matrix = state.matrix()
                        * Matrix4::from_translation(Vector3::new(values[0], values[1], values[2]));
                    state.replace_matrix(matrix);
                }
                Rule::rotate => {
                    let values = pbrt_matrix(inner_pair.into_inner());
                    if values.len() != 4 {
                        panic!("LookAt need to have 4 floats: {:?}", values);
                    }
                    let angle = values[0];
                    let axis = Vector3::new(values[1], values[2], values[3]).normalize();
                    let matrix = state.matrix() * Matrix4::from_axis_angle(axis, Deg(angle));
                    state.replace_matrix(matrix);
                }
                Rule::named_statement => {
                    for rule_pair in inner_pair.into_inner() {
                        match rule_pair.as_rule() {
                            Rule::integrator | Rule::sampler | Rule::pixel_filter => {
                                // Ignore these parameters
                            }
                            Rule::camera => {
                                if let Some(c) = Camera::new(rule_pair, state.matrix()) {
                                    scene_info.cameras.push(c);
                                }
                            }
                            Rule::texture => {
                                if let Some((name, mat)) = Texture::new(rule_pair, working_dir) {
                                    scene_info.textures.insert(name, mat);
                                }
                            }
                            Rule::make_named_material => {
                                if let Some((name, mat)) = BSDF::new(rule_pair, false) {
                                    scene_info.materials.insert(name, mat);
                                }
                            }
                            Rule::named_material => {
                                let (name, _) = parse_parameters(rule_pair);
                                state.set_named_matrial(name);
                            }
                            Rule::material => {
                                if let Some((_, mat)) = BSDF::new(rule_pair, true) {
                                    let name = format!(
                                        "unamed_material_{}",
                                        scene_info.number_unamed_materials
                                    );
                                    scene_info.number_unamed_materials += 1;
                                    scene_info.materials.insert(name.to_string(), mat);
                                }
                            }
                            Rule::shape => {
                                if let Some((_name, shape)) = Shape::new(rule_pair, &working_dir) {
                                    let mut shape = ShapeInfo::new(shape, state.matrix());
                                    shape.material_name = state.named_material();
                                    shape.emission = state.emission();
                                    let shape = Rc::new(shape);
                                    match state.object {
                                        Some(ref mut o) => {
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
                            Rule::film => {
                                let (_name, mut param) = parse_parameters(rule_pair);
                                scene_info.image_size = Vector2::new(
                                    param.remove("xresolution").unwrap().into_integer()[0] as u32,
                                    param.remove("yresolution").unwrap().into_integer()[0] as u32,
                                );
                            }
                            Rule::area_light_source => {
                                let (typename, mut light) = parse_parameters(rule_pair);
                                match typename.as_ref() {
                                    "diffuse" => {
                                        if let Some(e) = light.remove("L") {
                                            // TODO: If other name, not exported
                                            state.set_emission(e);
                                        }
                                    }
                                    _ => warn!("Unsuppored area light: {}", typename),
                                }
                            }
                            Rule::light_source => {
                                if let Some(light) = Light::new(rule_pair) {
                                    scene_info.lights.push(light);
                                }
                            }
                            Rule::coord_sys_transform => {
                                let (name, _) = parse_parameters(rule_pair);
                                state.replace_matrix(*scene_info.transforms.get(&name).unwrap());
                            }
                            Rule::coord_sys => {
                                let (name, _) = parse_parameters(rule_pair);
                                scene_info
                                    .transforms
                                    .insert(name, state.matrix.last().unwrap().clone());
                            }
                            Rule::include => {
                                let (name, _) = parse_parameters(rule_pair);
                                info!("Include found: {}", name);
                                let filename = working_dir.join(name);
                                read_pbrt_file(
                                    filename.to_str().unwrap(),
                                    working_dir,
                                    scene_info,
                                    state,
                                );
                            }
                            _ => warn!("Ignoring named statement: {:?}", rule_pair.as_rule()),
                        }
                    }
                }
                Rule::keyword => {
                    for rule_pair in inner_pair.into_inner() {
                        match rule_pair.as_rule() {
                            Rule::attribute_begin | Rule::transform_begin => {
                                state.save();
                            }
                            Rule::object_begin => {
                                // In san miguel, attribute begin and object begin are wrong...
                                let (name, _) = parse_parameters(rule_pair);
                                state.new_object(name);
                            }
                            Rule::object_end => {
                                let object = state.finish_object();
                                scene_info
                                    .objects
                                    .insert(object.name.clone(), Rc::new(object));
                            }
                            Rule::object_instance => {
                                let (name, _) = parse_parameters(rule_pair);
                                let object = match scene_info.objects.get(&name) {
                                    Some(ref o) => Rc::clone(o),
                                    None => {
                                        panic!("Impossible to found the object named: {}", name)
                                    }
                                };
                                scene_info.instances.push(InstanceInfo {
                                    matrix: state.matrix(),
                                    object,
                                })
                            }
                            Rule::attribute_end | Rule::transform_end => {
                                state.restore();
                            }
                            Rule::identity => {
                                state.replace_matrix(Matrix4::from_diagonal(Vector4::new(
                                    1.0, 1.0, 1.0, 1.0,
                                )));
                            }
                            Rule::world_begin => {
                                // Reinit the transformation matrix
                                state.replace_matrix(Matrix4::identity());
                            }
                            _ => warn!("Ignoring keyword: {:?}", rule_pair.as_rule()),
                        }
                    }
                }
                _ => warn!("Ignoring: {:?}", span.as_str()),
            }
        }
    }
    info!("Time for parsing file: {:?}", Instant::now() - now);
}
