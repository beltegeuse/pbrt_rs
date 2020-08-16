use cgmath::*;
use nom::{
    bytes::complete::{tag, take_while},
    character::complete::char,
    error::ParseError,
    number::complete::float,
    sequence::{delimited, preceded},
    IResult,
};
use std::collections::HashMap;

/// parser combinators are constructed from the bottom up:
/// first we write parsers for the smallest elements (here a space character),
/// then we'll combine them in larger parsers
fn sp<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&'a str, &'a str, E> {
    let chars = " \t\r\n";

    // nom combinators like `take_while` return a function. That function is the
    // parser,to which we can pass the input
    take_while(move |c| chars.contains(c))(i)
}

fn parse_string<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&'a str, &'a str, E> {
    nom::bytes::complete::take_until("\"")(i)
}

fn parse_string_sp<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&'a str, &'a str, E> {
    let chars = " \""; // space or "
    take_while(move |c| chars.contains(c))(i)
}
// FIXME: Values need to be vec
//  For many types
#[derive(Debug, Clone)]
pub enum Value {
    Integer(Vec<i32>),
    Float(Vec<f32>),
    Vector3(Vec<Vector3<f32>>),
    String(String),
    Boolean(bool),
}

pub fn parse_value<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&'a str, (String, Value), E> {
    let (i, v) = preceded(sp, delimited(
        char('"'),
        nom::multi::many_m_n(2, 2, preceded(sp, parse_string_sp)),
        char('"'),
    ))(i)?;

    let v = v.into_iter().map(|v| v.to_owned()).collect::<Vec<_>>();
    assert_eq!(v.len(), 2); // Not necessary
    let n = v[1].clone();

    let (i, v) = match &v[0][..] {
        "integer" => {
            let (i, v) = preceded(
                sp,
                delimited(
                    preceded(char('['), sp),
                    nom::multi::many0(preceded(sp, nom::character::complete::digit1)),
                    preceded(sp, char(']')),
                ),
            )(i)?;
            let v = v.into_iter().map(|v| v.parse::<i32>().unwrap()).collect();
            (i, Value::Integer(v))
        }
        "bool" => {
            let (i, v) = preceded(
                sp,
                delimited(
                    preceded(char('['), sp),
                    delimited(char('"'), nom::character::complete::alpha1, char('"')),
                    preceded(sp, char(']')),
                ),
            )(i)?;

            let v = match v {
                "false" => false,
                "true" => true,
                _ => panic!("Wrong bool type: {}", v),
            };

            (i, Value::Boolean(v))
        }
        "point" | "normal" => {
            let (i, v) = preceded(
                sp,
                delimited(
                    preceded(char('['), sp),
                    nom::multi::many0(preceded(sp, float)),
                    preceded(sp, char(']')),
                ),
            )(i)?;
            assert_eq!(v.len() % 3, 0);
            let v = v
                .chunks_exact(3)
                .map(|v| Vector3::new(v[0], v[1], v[2]))
                .collect();
            (i, Value::Vector3(v))
        }
        "float" | "rgb" => {
            let (i, v) = preceded(
                sp,
                delimited(
                    preceded(char('['), sp),
                    nom::multi::many0(preceded(sp, float)),
                    preceded(sp, char(']')),
                ),
            )(i)?;
            (i, Value::Float(v))
        }
        "string" | "texture" => {
            let (i, v) = preceded(
                sp,
                delimited(
                    preceded(char('['), sp),
                    delimited(char('"'), parse_string, char('"')),
                    preceded(sp, char(']')),
                ),
            )(i)?;
            (i, Value::String(v.to_owned()))
        }
        _ => panic!("{:?} not valid type", v[0]),
    };

    Ok((i, (n, v)))
}

#[derive(Debug, Clone)]
pub enum Keyword {
    AttributeBegin,
    AttributeEnd,
    Identity,
    ObjectBegin,
    ObjectEnd,
    ObjectInstance,
    ReverseOrientation,
    TransformBegin,
    TransformEnd,
    WorldBegin,
    WorldEnd,
}
pub fn parse_keyword<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&'a str, Keyword, E> {
    nom::branch::alt((
        nom::combinator::map(tag("AttributeBegin"), |_| Keyword::AttributeBegin),
        nom::combinator::map(tag("AttributeEnd"), |_| Keyword::AttributeEnd),
        nom::combinator::map(tag("Identity"), |_| Keyword::Identity),
        nom::combinator::map(tag("ObjectBegin"), |_| Keyword::ObjectBegin),
        nom::combinator::map(tag("ObjectInstance"), |_| Keyword::ObjectInstance),
        nom::combinator::map(tag("ReverseOrientation"), |_| Keyword::ReverseOrientation),
        nom::combinator::map(tag("TransformBegin"), |_| Keyword::TransformBegin),
        nom::combinator::map(tag("TransformEnd"), |_| Keyword::TransformEnd),
        nom::combinator::map(tag("WorldBegin"), |_| Keyword::WorldBegin),
        nom::combinator::map(tag("WorldEnd"), |_| Keyword::WorldEnd),
    ))(i)
}

#[derive(Debug, Clone)]
pub enum NamedTokenType {
    Accelerator,
    AreaLightSource,
    Camera,
    CoordSys,
    CoordSysTransform,
    Film,
    Integrator,
    LightSource,
    MakeNamedMaterial,
    MakeNamedMedium,
    Material,
    NamedMaterial,
    Include,
    PixelFilter,
    Sampler,
    Shape,
}
pub fn parse_named_token_type<'a, E: ParseError<&'a str>>(
    i: &'a str,
) -> IResult<&'a str, NamedTokenType, E> {
    nom::branch::alt((
        nom::combinator::map(tag("Accelerator"), |_| NamedTokenType::Accelerator),
        nom::combinator::map(tag("AreaLightSource"), |_| NamedTokenType::AreaLightSource),
        nom::combinator::map(tag("Camera"), |_| NamedTokenType::Camera),
        nom::combinator::map(tag("CoordSys"), |_| NamedTokenType::CoordSys),
        nom::combinator::map(tag("CoordSysTransform"), |_| {
            NamedTokenType::CoordSysTransform
        }),
        nom::combinator::map(tag("Film"), |_| NamedTokenType::Film),
        nom::combinator::map(tag("Integrator"), |_| NamedTokenType::Integrator),
        nom::combinator::map(tag("LightSource"), |_| NamedTokenType::LightSource),
        nom::combinator::map(tag("MakeNamedMaterial"), |_| {
            NamedTokenType::MakeNamedMaterial
        }),
        nom::combinator::map(tag("MakeNamedMedium"), |_| NamedTokenType::MakeNamedMedium),
        nom::combinator::map(tag("Material"), |_| NamedTokenType::Material),
        nom::combinator::map(tag("NamedMaterial"), |_| NamedTokenType::NamedMaterial),
        nom::combinator::map(tag("Include"), |_| NamedTokenType::Include),
        nom::combinator::map(tag("PixelFilter"), |_| NamedTokenType::PixelFilter),
        nom::combinator::map(tag("Sampler"), |_| NamedTokenType::Sampler),
        nom::combinator::map(tag("Shape"), |_| NamedTokenType::Shape),
    ))(i)
}

#[derive(Debug, Clone)]
pub struct NamedToken {
    pub internal_type: String,
    pub values: HashMap<String, Value>,
    pub object_type: NamedTokenType,
}

pub fn parse_named_token<'a, E: ParseError<&'a str>>(
    i: &'a str,
) -> IResult<&'a str, NamedToken, E> {
    let (i, object_type) = parse_named_token_type(i)?;

    let (i, internal_type) = nom::combinator::cut(preceded(
        sp,
        delimited(
            preceded(char('"'), sp),
            parse_string,
            preceded(sp, char('"')),
        ),
    ))(i)?;

    let (i, values) = nom::combinator::cut(nom::multi::fold_many0(
        preceded(sp, parse_value),
        HashMap::new(),
        |mut acc: HashMap<String, Value>, item: (String, Value)| {
            acc.insert(item.0, item.1);
            acc
        },
    ))(i)?;

    Ok((
        i,
        NamedToken {
            internal_type: internal_type.to_owned(),
            values,
            object_type,
        },
    ))
}
pub fn parse_named_token_many<'a, E: ParseError<&'a str>>(
    i: &'a str,
) -> IResult<&'a str, Vec<NamedToken>, E> {
    nom::multi::fold_many1(
        preceded(sp, parse_named_token),
        Vec::new(),
        |mut acc: Vec<NamedToken>, item: NamedToken| {
            acc.push(item);
            acc
        },
    )(i)
}

#[derive(Debug, Clone)]
pub enum Token {
    Transform(Vec<f32>),
    Texture {
        infos: Vec<String>,
        values: HashMap<String, Value>,
    },
    NamedToken(NamedToken),
    Keyword(Keyword),
}

// ContextError<&'a str>
pub fn transform<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&'a str, Token, E> {
    let (i, v) = preceded(
        preceded(tag("Transform"), sp),
        delimited(
            preceded(char('['), sp),
            nom::multi::many0(preceded(sp, float)),
            preceded(sp, char(']')),
        ),
    )(i)?;

    assert_eq!(v.len(), 16);

    Ok((i, Token::Transform(v)))
}

pub fn texture<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&'a str, Token, E> {
    let (i, _) = preceded(tag("Texture"), sp)(i)?;

    // Contains all the info
    let infos = std::cell::RefCell::new(vec![]);
    let values = std::cell::RefCell::new(HashMap::new());

    let (i, _) = nom::multi::many0_count(nom::branch::alt((
        nom::combinator::map(preceded(sp, parse_value), |item: (String, Value)| {
            let mut values = values.borrow_mut();
            values.insert(item.0, item.1);
        }),
        nom::combinator::map(
            preceded(sp, delimited(char('"'), parse_string, char('"'))),
            |item: &str| {
                infos.borrow_mut().push(item.to_owned());
            },
        ),
    )))(i)?;

    Ok((
        i,
        Token::Texture {
            infos: infos.into_inner(),
            values: values.into_inner(),
        },
    ))
}

pub fn parse_token<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&'a str, Token, E> {
    nom::branch::alt((
        nom::combinator::map(parse_named_token, |v| Token::NamedToken(v)),
        nom::combinator::map(parse_keyword, |v| Token::Keyword(v)),
        transform,
        texture,
    ))(i)
}

pub fn parse<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&'a str, Vec<Token>, E> {
    let (i, v) = nom::multi::fold_many1(
        preceded(sp, parse_token),
        Vec::new(),
        |mut acc: Vec<Token>, item: Token| {
            acc.push(item);
            acc
        },
    )(i)?;

    // Skip last char
    // TODO: Make sure that it is Ok to do that
    let (i, _) = sp(i)?;

    Ok((i, v))
}

#[cfg(test)]
mod tests {
    use std::io::Read;
    fn check_parsing<
        'a,
        T: std::fmt::Debug,
        E: nom::error::ParseError<&'a str> + std::fmt::Debug,
    >(
        res: nom::IResult<&'a str, T, E>,
    ) {
        let (i, v) = res.expect("Error during parsing");
        println!("==============================");
        println!("Parsed: {:?}", v);
        println!("==============================");
        match i {
            "" => (),
            _ => panic!("Parsing is not complete: {:?}", i),
        }
    }

    #[test]
    fn transform() {
        let s = "Transform [ 0.993341 -0.0130485 -0.114467 -0 -0 0.993565 -0.11326 -0 -0.115208 -0.112506 -0.98695 -0 1.33651 -11.1523 51.6855 1]";
        check_parsing(crate::parser::transform::<nom::error::VerboseError<&str>>(
            s,
        ));
    }

    #[test]
    fn value_integer() {
        let s = "\"integer maxdepth\" [ 65 ]";
        check_parsing(crate::parser::parse_value::<nom::error::VerboseError<&str>>(s));
    }
    #[test]
    fn value_float() {
        let s = "\"float ywidth\" [ 1.000000 ]";
        check_parsing(crate::parser::parse_value::<nom::error::VerboseError<&str>>(s));
    }

    #[test]
    fn value_string() {
        let s = "\"string filename\" [ \"bathroom2.png\" ]";
        check_parsing(crate::parser::parse_value::<nom::error::VerboseError<&str>>(s));
    }

    #[test]
    fn statement() {
        let s = "Film \"image\" \"integer xresolution\" [ 1280 ] \"integer yresolution\" [ 720 ] \"string filename\" [ \"bathroom2.png\" ]";
        check_parsing(crate::parser::parse_named_token::<
            nom::error::VerboseError<&str>,
        >(s));
    }

    #[test]
    fn statements() {
        let s = r#"Integrator "path" "integer maxdepth" [ 65 ] 
        Sampler "sobol" "integer pixelsamples" [ 64 ] 
        PixelFilter "triangle" "float xwidth" [ 1.000000 ] "float ywidth" [ 1.000000 ] 
        Film "image" "integer xresolution" [ 1024 ] "integer yresolution" [ 1024 ] "string filename" [ "cornell-box.png" ] 
        Camera "perspective" "float fov" [ 19.500000 ]"#;

        check_parsing(crate::parser::parse_named_token_many::<
            nom::error::VerboseError<&str>,
        >(s));
    }

    #[test]
    fn shape() {
        let s = r#"Shape "trianglemesh" "integer indices" [ 0 1 2 0 2 3 ] "point P" [ -1 1.74846e-007 -1 -1 1.74846e-007 1 1 -1.74846e-007 1 1 -1.74846e-007 -1 ] "normal N" [ 4.37114e-008 1 1.91069e-015 4.37114e-008 1 1.91069e-015 4.37114e-008 1 1.91069e-015 4.37114e-008 1 1.91069e-015 ] "float uv" [ 0 0 1 0 1 1 0 1 ]"#;
        check_parsing(crate::parser::parse_named_token_many::<
            nom::error::VerboseError<&str>,
        >(s));
    }

    #[test]
    fn statements2() {
        let s = r#"	MakeNamedMaterial "Light" "string type" [ "matte" ] "rgb Kd" [ 0.000000 0.000000 0.000000 ] 
        NamedMaterial "Floor" 
        Shape "trianglemesh" "integer indices" [ 0 1 2 0 2 3 ] "point P" [ -1 1.74846e-007 -1 -1 1.74846e-007 1 1 -1.74846e-007 1 1 -1.74846e-007 -1 ] "normal N" [ 4.37114e-008 1 1.91069e-015 4.37114e-008 1 1.91069e-015 4.37114e-008 1 1.91069e-015 4.37114e-008 1 1.91069e-015 ] "float uv" [ 0 0 1 0 1 1 0 1 ]"#;
        check_parsing(crate::parser::parse::<nom::error::VerboseError<&str>>(s));
    }

    #[test]
    fn cbox() {
        let mut f = std::fs::File::open("./data/pbrt/cornell-box/scene.pbrt").unwrap();
        let mut contents = String::new();
        f.read_to_string(&mut contents).unwrap();
        check_parsing(crate::parser::parse::<nom::error::VerboseError<&str>>(
            &contents,
        ));
    }

    #[test]
    fn spaceship() {
        let mut f = std::fs::File::open("./data/pbrt/spaceship/scene.pbrt").unwrap();
        let mut contents = String::new();
        f.read_to_string(&mut contents).unwrap();
        check_parsing(crate::parser::parse::<nom::error::VerboseError<&str>>(
            &contents,
        ));
    }

    #[test]
    fn classroom() {
        let mut f = std::fs::File::open("./data/pbrt/classroom/scene.pbrt").unwrap();
        let mut contents = String::new();
        f.read_to_string(&mut contents).unwrap();
        check_parsing(crate::parser::parse::<nom::error::VerboseError<&str>>(
            &contents,
        ));
    }

    #[test]
    fn texture() {
        let s = r#"Texture "Texture01" "spectrum" "imagemap" "string filename" [ "textures/wood1.tga" ] "bool trilinear" [ "true" ]"#;
        check_parsing(crate::parser::texture::<nom::error::VerboseError<&str>>(&s));
    }

    #[test]
    fn benedikt_scenes() {
        let scenes = vec![
            "./veach-bidir/scene.pbrt",
            "./living-room/scene.pbrt",
            "./cornell-box/scene.pbrt",
            "./classroom/scene.pbrt",
            "./kitchen/scene.pbrt",
            "./bathroom2/scene.pbrt",
            "./volumetric-caustic/scene.pbrt",
            "./staircase/scene.pbrt",
            "./veach-ajar/scene.pbrt",
            "./spaceship/scene.pbrt",
            "./veach-mis/scene.pbrt",
            "./water-caustic/scene.pbrt",
        ];
        let wk = std::path::Path::new("./data/pbrt");
        for s in scenes {
            let filename = wk.join(std::path::Path::new(s)).to_str().unwrap().to_owned();
            println!("READ: {:?}", filename);
            let mut f = std::fs::File::open(filename).unwrap();
            let mut contents = String::new();
            f.read_to_string(&mut contents).unwrap();
            check_parsing(crate::parser::parse::<nom::error::VerboseError<&str>>(
                &contents,
            ));
        }
    }
}
