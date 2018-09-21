# pbrt_rs

Rust crate to parse [PBRT v3](https://github.com/mmp/pbrt-v3) scenes files. A big part of the code was extracted from the excellent [PBRT's Rust version](https://github.com/wahn/rs_pbrt).
This parser is tested and integrated inside [rustlight](https://github.com/beltegeuse/rustlight) to check its correctness and usability.

The current version of the parser is very unstable and is not supporting all PBRT scenes features. More features will be added in the future. When the crate is more stable, it will be published on crates.io with proper documentation.

![Kitchen scene with GPT and weighted reconstruction](http://beltegeuse.s3-website-ap-northeast-1.amazonaws.com/rustlight/pbrt_rs.png)

[Kitchen scene](https://benedikt-bitterli.me/resources/) rendered inside rustlight with GPT and weighted reconstruction (256 spp).

## TODO / Problems
- A problem in transformation computation makes some scene like veach-door broken.
