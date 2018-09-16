extern crate clap;
extern crate env_logger;
#[macro_use]
extern crate log;
extern crate pest;
#[macro_use]
extern crate pest_derive;
extern crate cgmath;
extern crate ply_rs;

// parser
use cgmath::{Vector2, Vector3};
use clap::{App, Arg};
use pest::Parser;
use ply_rs as ply;
use std::collections::HashMap;
use std::io::Read;
use std::str::FromStr;
use std::time::Instant;

const _GRAMMAR: &str = include_str!("pbrt.pest");

#[derive(Parser)]
#[grammar = "pbrt.pest"]
struct PbrtParser;

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

/// Intermediate representation
/// for parsing the parameters
#[derive(Debug)]
enum Param {
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
    fn parse_float(pairs: &mut pest::iterators::Pairs<Rule>) -> (String, Box<Self>) {
        let (name, values) = pbrt_parameter(pairs);
        (name, Box::new(Param::Float(values)))
    }

    fn to_name(self) -> String {
        match self {
            Param::Name(v) => v,
            _ => panic!("impossible to convert to name: {:?}", self),
        }
    }
    fn parse_name(pairs: &mut pest::iterators::Pairs<Rule>) -> (String, Box<Self>) {
        let (name, values) = pbrt_parameter::<String>(pairs);
        let values = values[0].clone(); // TODO
        let values = values.trim_matches('\"').to_string();
        (name, Box::new(Param::Name(values)))
    }

    fn to_rgb(self) -> (f32, f32, f32) {
        match self {
            Param::RGB(r, g, b) => (r, g, b),
            _ => panic!("impossible to convert to rgb: {:?}", self),
        }
    }
    fn parse_rgb(pairs: &mut pest::iterators::Pairs<Rule>) -> (String, Box<Self>) {
        let (name, values) = pbrt_parameter::<f32>(pairs);
        (name, Box::new(Param::RGB(values[0], values[1], values[2])))
    }

    fn to_integer(self) -> Vec<i32> {
        match self {
            Param::Integer(v) => v,
            _ => panic!("impossible to convert to integer: {:?}", self),
        }
    }
    fn parse_integer(pairs: &mut pest::iterators::Pairs<Rule>) -> (String, Box<Self>) {
        let (name, values) = pbrt_parameter::<i32>(pairs);
        (name, Box::new(Param::Integer(values)))
    }

    fn to_vector3(self) -> Vec<Vector3<f32>> {
        match self {
            Param::Vector3(v) => v,
            _ => panic!("impossible to convert to integer: {:?}", self),
        }
    }
    fn parse_vector3(pairs: &mut pest::iterators::Pairs<Rule>) -> (String, Box<Self>) {
        let (name, values) = pbrt_parameter::<f32>(pairs);
        if values.len() % 3 != 0 {
            panic!("Non 3 multiples for vector 3");
        }
        let values = values
            .chunks(3)
            .map(|v| Vector3::new(v[0], v[1], v[2]))
            .collect();
        (name, Box::new(Param::Vector3(values)))
    }
}

fn parse_parameters(pairs: pest::iterators::Pair<Rule>) -> (String, HashMap<String, Box<Param>>) {
    let mut name = None;
    let mut param_map: HashMap<String, Box<Param>> = HashMap::default();
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
                        Rule::string_param => {
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
struct CameraPerspective {
    pub fov: f32,
}
enum Camera {
    Perspective(CameraPerspective),
}
impl Camera {
    fn new(pairs: pest::iterators::Pair<Rule>) -> Option<Box<Self>> {
        let (name, mut param) = parse_parameters(pairs);
        match name.as_ref() {
            "perspective" => {
                let fov = param.remove("fov").expect("fov is not given").to_float()[0];
                Some(Box::new(Camera::Perspective(CameraPerspective { fov })))
            }
            _ => {
                warn!("Camera case with {} is not cover", name);
                None
            }
        }
    }
}

/// BSDF representation
struct DiffuseBSDF {
    pub kd: Box<Param>,
}
enum BSDF {
    Diffuse(DiffuseBSDF),
}
impl BSDF {
    fn new(pairs: pest::iterators::Pair<Rule>) -> Option<(String, Box<Self>)> {
        let (name, mut param) = parse_parameters(pairs);
        // TODO: Need to clone to avoid borrower checker
        let bsdf_type = param
            .remove("type")
            .expect("bsdf type param is required")
            .to_name();
        match bsdf_type.as_ref() {
            "matte" => {
                let kd = param
                    .remove("Kd")
                    .expect("Kd parameter need to be provided");
                Some((name, Box::new(BSDF::Diffuse(DiffuseBSDF { kd }))))
            }
            _ => {
                warn!("BSDF case with {} is not cover", bsdf_type);
                None
            }
        }
    }
}

/// Mesh representation
struct TriMeshShape {
    pub indices: Vec<u32>,
    pub points: Vec<Vector3<f32>>,
    pub normals: Option<Vec<Vector3<f32>>>,
    pub uv: Option<Vec<Vector2<f32>>>,
}
enum Shape {
    TriMesh(TriMeshShape),
}
impl Shape {
    fn new(
        pairs: pest::iterators::Pair<Rule>,
        wk: &std::path::Path,
    ) -> Option<(String, Box<Self>)> {
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
                    Box::new(Shape::TriMesh(TriMeshShape {
                        indices,
                        points,
                        normals,
                        uv,
                    })),
                ))
            }
            "plymesh" => {
                let filename = param
                    .remove("filename")
                    .expect("filename is required")
                    .to_name();
                let filename = wk.join(filename);
                let mut f = std::fs::File::open(filename).unwrap();
                // create a parser
                let p = ply::parser::Parser::<ply::ply::DefaultElement>::new();
                // use the parser: read the entire file
                let ply = p.read_ply(&mut f);
                // make sure it did work
                assert!(ply.is_ok());
                let ply = ply.unwrap();
                info!("Ply header: {:#?}", ply.header);
                None
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
struct State {
    pub named_material: Option<String>,
}
impl Default for State {
    fn default() -> Self {
        State {
            named_material: None,
        }
    }
}

/// Scene representation
struct Scene {
    pub cameras: Vec<Box<Camera>>,
    pub materials: HashMap<String, Box<BSDF>>,
    pub shapes: Vec<(Option<String>, Box<Shape>)>,
}
impl Default for Scene {
    fn default() -> Self {
        Scene {
            cameras: Vec::default(),
            materials: HashMap::default(),
            shapes: Vec::default(),
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
        ).arg(Arg::with_name("debug").short("d").help("debug output"))
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

    // Read the file
    let now = Instant::now();
    info!("Loading: {}", scene_path_str);
    let working_dir = std::path::Path::new(scene_path_str.clone())
        .parent()
        .unwrap();
    let file = std::fs::File::open(scene_path_str.clone())
        .expect(&format!("Impossible to open {}", scene_path_str));
    let mut reader = std::io::BufReader::new(file);
    let mut str_buf: String = String::default();
    let _num_bytes = reader.read_to_string(&mut str_buf);
    info!("Time for reading file: {:?}", Instant::now() - now);

    let now = Instant::now();
    let pairs =
        PbrtParser::parse(Rule::pbrt, &str_buf).unwrap_or_else(|e| panic!("Parsing error: {}", e));

    // The parsing loop
    let mut scene_info = Scene::default();
    let mut state = vec![State::default()];
    for pair in pairs {
        let span = pair.clone().into_span();
        debug!("Rule:    {:?}", pair.as_rule());
        debug!("Span:    {:?}", span);
        debug!("Text:    {}", span.as_str());
        for inner_pair in pair.into_inner() {
            match inner_pair.as_rule() {
                Rule::named_statement => {
                    for rule_pair in inner_pair.into_inner() {
                        match rule_pair.as_rule() {
                            Rule::camera => {
                                if let Some(c) = Camera::new(rule_pair) {
                                    scene_info.cameras.push(c);
                                }
                            }
                            Rule::make_named_material => {
                                if let Some((name, mat)) = BSDF::new(rule_pair) {
                                    scene_info.materials.insert(name, mat);
                                }
                            }
                            Rule::named_material => {
                                let (name, _) = parse_parameters(rule_pair);
                                state.last_mut().unwrap().named_material = Some(name);
                            }
                            Rule::shape => {
                                if let Some((_name, shape)) = Shape::new(rule_pair, &working_dir) {
                                    let name_bsdf = state.last().unwrap().named_material.clone();
                                    scene_info.shapes.push((name_bsdf, shape));
                                }
                            }
                            _ => warn!("Ignoring named statement: {:?}", rule_pair.as_rule()),
                        }
                    }
                }
                _ => warn!("Ignoring: {:?}", span.as_str()),
            }
        }
    }
    info!("Time for parsing file: {:?}", Instant::now() - now);

    // Print statistics
    info!("Scenes info: ");
    info!(" - BSDFS: {}", scene_info.materials.len());
    info!(" - Shapes: {}", scene_info.shapes.len());
}
