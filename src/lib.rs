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
        let not_opening: bool = rule_pair.as_str() != String::from("[");
        let not_closing: bool = rule_pair.as_str() != String::from("]");
        if not_opening && not_closing {
            let number = f32::from_str(rule_pair.clone().into_span().as_str()).unwrap();
            m.push(number);
        }
    }
    m
}
fn pbrt_parameter<T: FromStr>(pairs: &mut pest::iterators::Pairs<Rule>) -> (String, Vec<T>)
where
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    let mut floats: Vec<T> = Vec::new();
    // single float or several floats using brackets
    let ident = pairs.next();
    let name = String::from(ident.unwrap().clone().into_span().as_str());
    let option = pairs.next();
    let lbrack = option.clone().unwrap();
    if lbrack.as_str() == String::from("[") {
        // check for brackets
        let mut number = pairs.next();
        while number.is_some() {
            let pair = number.unwrap().clone();
            if pair.as_str() == String::from("]") {
                // closing bracket found
                break;
            } else {
                let float = pair
                    .into_span()
                    .as_str()
                    .parse::<T>()
                    .expect("parsing error on parameter");
                floats.push(float);
            }
            number = pairs.next();
        }
    } else {
        // no brackets
        let mut number = option.clone();
        while number.is_some() {
            let pair = number.unwrap().clone();
            let float = pair
                .into_span()
                .as_str()
                .parse::<T>()
                .expect("parsing error on parameter");
            floats.push(float);
            number = pairs.next();
        }
    }
    (name, floats)
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
            (k, _) => panic!("Face: Unexpected key/value combination: key: {}", k),
        }
    }
}

/// Intermediate representation
/// for parsing the parameters
#[derive(Debug, Clone)]
pub enum Param {
    Integer(Vec<i32>),
    Float(Vec<f32>),
    Vector3(Vec<Vector3<f32>>),
    Name(String),
    RGB(f32, f32, f32),
}
impl Param {
    fn to_float(self) -> Vec<f32> {
        match self {
            Param::Float(v) => v,
            _ => panic!("impossible to convert to float: {:?}", self),
        }
    }
    fn parse_float(pairs: &mut pest::iterators::Pairs<Rule>) -> (String, Self) {
        let (name, values) = pbrt_parameter(pairs);
        (name, Param::Float(values))
    }

    fn to_name(self) -> String {
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

    fn to_rgb(self) -> (f32, f32, f32) {
        match self {
            Param::RGB(r, g, b) => (r, g, b),
            _ => panic!("impossible to convert to rgb: {:?}", self),
        }
    }
    fn parse_rgb(pairs: &mut pest::iterators::Pairs<Rule>) -> (String, Self) {
        let (name, values) = pbrt_parameter::<f32>(pairs);
        (name, Param::RGB(values[0], values[1], values[2]))
    }

    fn to_integer(self) -> Vec<i32> {
        match self {
            Param::Integer(v) => v,
            _ => panic!("impossible to convert to integer: {:?}", self),
        }
    }
    fn parse_integer(pairs: &mut pest::iterators::Pairs<Rule>) -> (String, Self) {
        let (name, values) = pbrt_parameter::<i32>(pairs);
        (name, Param::Integer(values))
    }

    fn to_vector3(self) -> Vec<Vector3<f32>> {
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
}

fn parse_parameters(pairs: pest::iterators::Pair<Rule>) -> (String, HashMap<String, Param>) {
    let mut name = None;
    let mut param_map: HashMap<String, Param> = HashMap::default();
    for pair in pairs.into_inner() {
        match pair.as_rule() {
            Rule::empty_string => {}
            Rule::string => {
                let mut string_pairs = pair.into_inner();
                let ident = string_pairs.next();
                name = Some(String::from_str(ident.unwrap().clone().into_span().as_str()).unwrap());
            }
            Rule::parameter => {
                for parameter_pair in pair.into_inner() {
                    match parameter_pair.as_rule() {
                        Rule::float_param => {
                            let (name, value) =
                                Param::parse_float(&mut parameter_pair.into_inner());
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
                        _ => warn!("Ignoring Parameter: {}", parameter_pair.as_str()),
                    }
                }
            }
            _ => warn!("Ignoring: {}", pair.as_str()),
        }
    }
    if let Some(name) = name {
        (name, param_map)
    } else {
        panic!("Parse parameter, name is not provided");
    }
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
                let fov = param.remove("fov").expect("fov is not given").to_float()[0];
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

/// BSDF representation
pub struct DiffuseBSDF {
    pub kd: Param,
}
pub enum BSDF {
    Diffuse(DiffuseBSDF),
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
                .to_name()
        };
        match bsdf_type.as_ref() {
            "matte" => {
                let kd = param
                    .remove("Kd")
                    .expect("Kd parameter need to be provided");
                Some((name, BSDF::Diffuse(DiffuseBSDF { kd })))
            }
            _ => {
                warn!("BSDF case with {} is not cover", bsdf_type);
                None
            }
        }
    }
}

/// Mesh representation
pub struct TriMeshShape {
    pub indices: Vec<u32>,
    pub points: Vec<Vector3<f32>>,
    pub normals: Option<Vec<Vector3<f32>>>,
    pub uv: Option<Vec<Vector2<f32>>>,
}
pub enum Shape {
    TriMesh(TriMeshShape),
}
impl Shape {
    fn new(pairs: pest::iterators::Pair<Rule>, wk: &std::path::Path) -> Option<(String, Self)> {
        let (name, mut param) = parse_parameters(pairs);
        match name.as_ref() {
            "trianglemesh" => {
                let points = param.remove("P").expect("P is required").to_vector3();
                let indices = param
                    .remove("indices")
                    .expect("indice is required")
                    .to_integer()
                    .into_iter()
                    .map(|v| v as u32)
                    .collect();
                let normals = if let Some(v) = param.remove("N") {
                    Some(v.to_vector3())
                } else {
                    None
                };
                let uv = if let Some(v) = param.remove("uv") {
                    Some(
                        v.to_float()
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
                    .to_name();
                let filename = wk.join(filename);
                //info!("Reading ply: {:?}", filename);
                let mut f = std::fs::File::open(filename).unwrap();
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
                        _ => panic!("Enexpeced element!"),
                    }
                }
                //info!(" - #vertex: {}", vertex_list.len());
                //info!(" - #face: {}", face_list.len());
                let mut indices = Vec::new();
                for f in face_list {
                    indices.extend(f.vertex_index.into_iter().map(|v| v as u32));
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
                let vertex_list = vertex_list.into_iter().map(|v| v.pos).collect();

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

/// State of the parser
#[derive(Clone, Debug)]
pub struct State {
    pub named_material: Option<String>,
    pub matrix: Matrix4<f32>,
    pub emission: Option<Param>,
}
impl Default for State {
    fn default() -> Self {
        State {
            named_material: None,
            matrix: Matrix4::identity(),
            emission: None,
        }
    }
}

/// Scene representation
pub struct ShapeInfo {
    pub data: Shape,
    pub material_name: Option<String>,
    pub emission: Option<Param>,
}
impl ShapeInfo {
    fn new(shape: Shape) -> Self {
        Self {
            data: shape,
            material_name: None,
            emission: None,
        }
    }
}
pub struct Scene {
    pub cameras: Vec<Camera>,
    pub materials: HashMap<String, BSDF>,
    pub shapes: Vec<ShapeInfo>,
    pub number_unamed_materials: usize,
    pub image_size: Vector2<u32>,
}
impl Default for Scene {
    fn default() -> Self {
        Scene {
            cameras: Vec::default(),
            materials: HashMap::default(),
            shapes: Vec::default(),
            number_unamed_materials: 0,
            image_size: Vector2::new(512, 512),
        }
    }
}

pub fn read_pbrt_file(path: &str, scene_info: &mut Scene, state: State) {
    let now = Instant::now();
    info!("Loading: {}", path);
    let working_dir = std::path::Path::new(path.clone()).parent().unwrap();
    let file = std::fs::File::open(path.clone()).expect(&format!("Impossible to open {}", path));
    let mut reader = std::io::BufReader::new(file);
    let mut str_buf: String = String::default();
    let _num_bytes = reader.read_to_string(&mut str_buf);
    info!("Time for reading file: {:?}", Instant::now() - now);

    let now = Instant::now();
    let pairs =
        PbrtParser::parse(Rule::pbrt, &str_buf).unwrap_or_else(|e| panic!("Parsing error: {}", e));
    let mut state = vec![state];
    for pair in pairs {
        let span = pair.clone().into_span();
        debug!("Rule:    {:?}", pair.as_rule());
        debug!("Span:    {:?}", span);
        debug!("Text:    {}", span.as_str());
        for inner_pair in pair.into_inner() {
            match inner_pair.as_rule() {
                Rule::transform => {
                    let values = pbrt_matrix(inner_pair.into_inner());
                    if values.len() != 16 {
                        panic!("Transform need to have 16 floats: {:?}", values);
                    }
                    let matrix = state.last().unwrap().matrix * Matrix4::new(
                        values[0], values[1], values[2], values[3], values[4], values[5],
                        values[6], values[7], values[8], values[9], values[10], values[11],
                        values[12], values[13], values[14], values[15],
                    );
                    state.last_mut().unwrap().matrix = matrix;
                }
                Rule::scale => {
                    let values = pbrt_matrix(inner_pair.into_inner());
                    if values.len() != 3 {
                        panic!("Scale need to have 3 floats: {:?}", values);
                    }
                    let matrix = state.last().unwrap().matrix * Matrix4::from_diagonal(
                        Vector4::new(values[0], values[1], values[2], 1.0),
                    );
                    state.last_mut().unwrap().matrix = matrix;
                }
                Rule::look_at => {
                    let values = pbrt_matrix(inner_pair.into_inner());
                    if values.len() != 9 {
                        panic!("LookAt need to have 9 floats: {:?}", values);
                    }
                    let eye = Point3::new(values[0], values[1], values[2]);
                    let target = Point3::new(values[3], values[4], values[5]);
                    let up = Vector3::new(values[6], values[7], values[8]);
                    let matrix = state.last().unwrap().matrix * Matrix4::look_at(eye, target, up);
                    state.last_mut().unwrap().matrix = matrix;
                }
                Rule::named_statement => {
                    for rule_pair in inner_pair.into_inner() {
                        match rule_pair.as_rule() {
                            Rule::camera => {
                                if let Some(c) =
                                    Camera::new(rule_pair, state.last().unwrap().matrix)
                                {
                                    scene_info.cameras.push(c);
                                }
                            }
                            Rule::make_named_material => {
                                if let Some((name, mat)) = BSDF::new(rule_pair, false) {
                                    scene_info.materials.insert(name, mat);
                                }
                            }
                            Rule::named_material => {
                                let (name, _) = parse_parameters(rule_pair);
                                state.last_mut().unwrap().named_material = Some(name);
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
                                    state.last().unwrap().named_material.clone();
                                    let mut shape = ShapeInfo::new(shape);
                                    shape.material_name =
                                        state.last().unwrap().named_material.clone();
                                    shape.emission = state.last().unwrap().emission.clone();
                                    scene_info.shapes.push(shape);
                                }
                            }
                            Rule::film => {
                                let (_name, mut param) = parse_parameters(rule_pair);
                                scene_info.image_size = Vector2::new(
                                    param.remove("xresolution").unwrap().to_integer()[0] as u32,
                                    param.remove("yresolution").unwrap().to_integer()[0] as u32,
                                );
                            }
                            Rule::area_light_source => {
                                let (typename, mut light) = parse_parameters(rule_pair);
                                match typename.as_ref() {
                                    "diffuse" => {
                                        state.last_mut().unwrap().emission =
                                            Some(light.remove("L").unwrap());
                                    }
                                    _ => warn!("Unsuppored area light: {}", typename),
                                }
                            }
                            Rule::include => {
                                let (name, _) = parse_parameters(rule_pair);
                                let filename = working_dir.join(name);
                                read_pbrt_file(
                                    filename.to_str().unwrap(),
                                    scene_info,
                                    state.last().unwrap().clone(),
                                );
                                unimplemented!();
                            }
                            _ => warn!("Ignoring named statement: {:?}", rule_pair.as_rule()),
                        }
                    }
                }
                Rule::keyword => {
                    for rule_pair in inner_pair.into_inner() {
                        match rule_pair.as_rule() {
                            Rule::attribute_begin | Rule::transform_begin => {
                                let new_state = state.last().unwrap().clone();
                                state.push(new_state);
                            }
                            Rule::attribute_end | Rule::transform_end => {
                                state.pop();
                            }
                            Rule::world_begin => {
                                // Reinit the transformation matrix
                                state.last_mut().unwrap().matrix = Matrix4::identity();
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
