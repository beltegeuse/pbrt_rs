use cgmath::*;
use ply_rs::parser;
use ply_rs::ply;

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

pub struct PlyLoaded {
    pub indices: Vec<Vector3<usize>>,
    pub points: Vec<Point3<f32>>,
    pub normals: Option<Vec<Vector3<f32>>>,
    pub uv: Option<Vec<Vector2<f32>>>,
}
impl PlyLoaded {
    pub fn to_trimesh(self) -> crate::Shape {
        crate::Shape::TriMesh {
            indices: self.indices,
            points: self.points,
            normals: self.normals,
            uv: self.uv,
        }
    }
}

pub fn read_ply(filename: &std::path::Path, verbose: bool) -> PlyLoaded {
    if verbose {
        info!("Reading {:?} ...", filename);
    }
    let f = match std::fs::File::open(filename) {
        Ok(f) => f,
        Err(e) => {
            panic!("Error in opening: {:?} => {:?}", filename, e);
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
    if verbose {
        info!(" - #vertex: {}", vertex_list.len());
        info!(" - #face: {}", face_list.len());
    }
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
            indices.push(Vector3::new(
                quad_indices[0],
                quad_indices[1],
                quad_indices[2],
            ));
            indices.push(Vector3::new(
                quad_indices[2],
                quad_indices[3],
                quad_indices[0],
            ));
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

    PlyLoaded {
        indices,
        points: vertex_list,
        normals,
        uv,
    }
}
