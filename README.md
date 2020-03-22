# pbrt_rs

Rust crate to parse [PBRT v3](https://github.com/mmp/pbrt-v3) scenes files. A big part of the code was extracted from the excellent [PBRT's Rust version](https://github.com/wahn/rs_pbrt).
This parser is tested and integrated inside [rustlight](https://github.com/beltegeuse/rustlight) to check its correctness and usability.

![Kitchen scene with GPT and weighted reconstruction](http://beltegeuse.s3-website-ap-northeast-1.amazonaws.com/rustlight/pbrt_rs.png)

[Kitchen scene](https://benedikt-bitterli.me/resources/) rendered inside rustlight with GPT and weighted reconstruction (256 spp).

## Running examples

This crate provided a useful example to test this library.

### Obj Exporter

Basic export of a PBRT scene into OBJ format. This is one example of how to export a PBRT scene:

```
cargo run --example=obj_exporter --release -- -o veach-door.obj data/pbrt/veach-ajar/scene.pbrt
```

This command generates .obj and .mtl files. The .mtl needs tweaking so you get the same material look inside your rendering system.

### Viewer

Only show the depth from the sensor perspective. For now, only the perspective camera is supported. Use simple BVH and multi-threaded rendering. This is one example of how to render a PBRT scene:

```
cargo run --example=viewer --release -- data/pbrt/veach-ajar/scene.pbrt out.pfm
```

## Known issues
- Does not support all PBRT primitives (shapes, bsdf, sensors)
- Pest parsing failed in some cases (under investigation)

I am waiting to resolve these issues above before publishing this crate officially. 