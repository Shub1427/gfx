#![allow(dead_code)] //TODO: remove

use hal::{ColorSlot};
use hal::pso;
use {gl, GlContainer};
use smallvec::SmallVec;

pub(crate) fn bind_polygon_mode(gl: &GlContainer, mode: pso::PolygonMode, bias: Option<pso::State<pso::DepthBias>>) {
    use hal::pso::PolygonMode::*;

    let (gl_draw, gl_offset) = match mode {
        Point => (gl::POINT, gl::POLYGON_OFFSET_POINT),
        Line(width) => {
            unsafe {
                #[cfg(not(target_arch = "wasm32"))]
                gl.LineWidth(width);
                #[cfg(target_arch = "wasm32")]
                unimplemented!();
            };
            (gl::LINE, gl::POLYGON_OFFSET_LINE)
        },
        Fill => (gl::FILL, gl::POLYGON_OFFSET_FILL),
    };

    unsafe {
        #[cfg(not(target_arch = "wasm32"))]
        gl.PolygonMode(gl::FRONT_AND_BACK, gl_draw);
        #[cfg(target_arch = "wasm32")]
        unimplemented!();
    };

    match bias {
        Some(pso::State::Static(bias)) => unsafe {
            #[cfg(not(target_arch = "wasm32"))]
            gl.Enable(gl_offset);
            #[cfg(not(target_arch = "wasm32"))]
            gl.PolygonOffset(bias.slope_factor as _, bias.const_factor as _);
            #[cfg(target_arch = "wasm32")]
            unimplemented!();
        }
        _ => unsafe {
            #[cfg(not(target_arch = "wasm32"))]
            gl.Disable(gl_offset);
            #[cfg(target_arch = "wasm32")]
            unimplemented!();
        }
    }
}

pub(crate) fn bind_rasterizer(gl: &GlContainer, r: &pso::Rasterizer, is_embedded: bool) {
    use hal::pso::FrontFace::*;

    unsafe {
        #[cfg(not(target_arch = "wasm32"))]
        gl.FrontFace(match r.front_face {
            Clockwise => gl::CW,
            CounterClockwise => gl::CCW,
        });
        #[cfg(target_arch = "wasm32")]
        unimplemented!();
    };

    if !r.cull_face.is_empty() {
        unsafe {
            #[cfg(not(target_arch = "wasm32"))]
            gl.Enable(gl::CULL_FACE);
            #[cfg(not(target_arch = "wasm32"))]
            gl.CullFace(match r.cull_face {
                pso::Face::FRONT => gl::FRONT,
                pso::Face::BACK => gl::BACK,
                _ => gl::FRONT_AND_BACK,
            });
            #[cfg(target_arch = "wasm32")]
            unimplemented!();
        }
    } else {
        unsafe {
            #[cfg(not(target_arch = "wasm32"))]
            gl.Disable(gl::CULL_FACE);
            #[cfg(target_arch = "wasm32")]
            unimplemented!();
        }
    }

    if !is_embedded {
        bind_polygon_mode(gl, r.polygon_mode, r.depth_bias);
        match false { //TODO
            true => unsafe {
                #[cfg(not(target_arch = "wasm32"))]
                gl.Enable(gl::MULTISAMPLE);
                #[cfg(target_arch = "wasm32")]
                unimplemented!();
            }
            false => unsafe {
                #[cfg(not(target_arch = "wasm32"))]
                gl.Disable(gl::MULTISAMPLE);
                #[cfg(target_arch = "wasm32")]
                unimplemented!();
            }
        }
    }
}

pub(crate) fn bind_draw_color_buffers(gl: &GlContainer, num: usize) {
    #[cfg(target_arch = "wasm32")]
    unimplemented!();
    let attachments: SmallVec<[gl::types::GLenum; 16]> =
        (0..num).map(|x| gl::COLOR_ATTACHMENT0 + x as u32).collect();
    #[cfg(not(target_arch = "wasm32"))]
    unsafe { gl.DrawBuffers(num as gl::types::GLint, attachments.as_ptr()) };
}

pub fn map_comparison(cmp: pso::Comparison) -> gl::types::GLenum {
    use hal::pso::Comparison::*;
    match cmp {
        Never        => gl::NEVER,
        Less         => gl::LESS,
        LessEqual    => gl::LEQUAL,
        Equal        => gl::EQUAL,
        GreaterEqual => gl::GEQUAL,
        Greater      => gl::GREATER,
        NotEqual     => gl::NOTEQUAL,
        Always       => gl::ALWAYS,
    }
}

pub(crate) fn bind_depth(gl: &GlContainer, depth: &pso::DepthTest) {
    match *depth {
        pso::DepthTest::On { fun, write } => unsafe {
            #[cfg(not(target_arch = "wasm32"))]
            gl.Enable(gl::DEPTH_TEST);
            #[cfg(not(target_arch = "wasm32"))]
            gl.DepthFunc(map_comparison(fun));
            #[cfg(not(target_arch = "wasm32"))]
            gl.DepthMask(write as _);
            #[cfg(target_arch = "wasm32")]
            unimplemented!();
        },
        pso::DepthTest::Off => unsafe {
            #[cfg(not(target_arch = "wasm32"))]
            gl.Disable(gl::DEPTH_TEST);
            #[cfg(target_arch = "wasm32")]
            unimplemented!();
        },
    }
}

fn map_operation(op: pso::StencilOp) -> gl::types::GLenum {
    use hal::pso::StencilOp::*;
    match op {
        Keep          => gl::KEEP,
        Zero          => gl::ZERO,
        Replace       => gl::REPLACE,
        IncrementClamp=> gl::INCR,
        IncrementWrap => gl::INCR_WRAP,
        DecrementClamp=> gl::DECR,
        DecrementWrap => gl::DECR_WRAP,
        Invert        => gl::INVERT,
    }
}

pub(crate) fn bind_stencil(
    gl: &GlContainer,
    stencil: &pso::StencilTest,
    (ref_front, ref_back): (pso::StencilValue, pso::StencilValue),
    cull: Option<pso::Face>,
) {
    fn bind_side(gl: &GlContainer, face: gl::types::GLenum, side: &pso::StencilFace, ref_value: pso::StencilValue) {
        unsafe {
            let mr = match side.mask_read {
                pso::State::Static(v) => v,
                pso::State::Dynamic => !0,
            };
            let mw = match side.mask_write {
                pso::State::Static(v) => v,
                pso::State::Dynamic => !0,
            };
            #[cfg(not(target_arch = "wasm32"))]
            gl.StencilFuncSeparate(face, map_comparison(side.fun), ref_value as _, mr);
            #[cfg(not(target_arch = "wasm32"))]
            gl.StencilMaskSeparate(face, mw);
            #[cfg(not(target_arch = "wasm32"))]
            gl.StencilOpSeparate(face, map_operation(side.op_fail), map_operation(side.op_depth_fail), map_operation(side.op_pass));
            #[cfg(target_arch = "wasm32")]
            unimplemented!();
        }
    }
    match *stencil {
        pso::StencilTest::On { ref front, ref back } => {
            unsafe {
                #[cfg(not(target_arch = "wasm32"))]
                gl.Enable(gl::STENCIL_TEST);
                #[cfg(target_arch = "wasm32")]
                unimplemented!();
            };
            if let Some(cf) = cull {
                if !cf.contains(pso::Face::FRONT) {
                    bind_side(gl, gl::FRONT, front, ref_front);
                }
                if !cf.contains(pso::Face::BACK) {
                    bind_side(gl, gl::BACK, back, ref_back);
                }
            }
        }
        pso::StencilTest::Off => unsafe {
            #[cfg(not(target_arch = "wasm32"))]
            gl.Disable(gl::STENCIL_TEST);
            #[cfg(target_arch = "wasm32")]
            unimplemented!();
        },
    }
}

fn map_factor(factor: pso::Factor) -> gl::types::GLenum {
    use hal::pso::Factor::*;
    match factor {
        Zero => gl::ZERO,
        One => gl::ONE,
        SrcColor => gl::SRC_COLOR,
        OneMinusSrcColor => gl::ONE_MINUS_SRC_COLOR,
        DstColor => gl::DST_COLOR,
        OneMinusDstColor => gl::ONE_MINUS_DST_COLOR,
        SrcAlpha => gl::SRC_ALPHA,
        OneMinusSrcAlpha => gl::ONE_MINUS_SRC_ALPHA,
        DstAlpha => gl::DST_ALPHA,
        OneMinusDstAlpha => gl::ONE_MINUS_DST_ALPHA,
        ConstColor => gl::CONSTANT_COLOR,
        OneMinusConstColor => gl::ONE_MINUS_CONSTANT_COLOR,
        ConstAlpha => gl::CONSTANT_ALPHA,
        OneMinusConstAlpha => gl::ONE_MINUS_CONSTANT_ALPHA,
        SrcAlphaSaturate => gl::SRC_ALPHA_SATURATE,
        Src1Color => gl::SRC1_COLOR,
        OneMinusSrc1Color => gl::ONE_MINUS_SRC1_COLOR,
        Src1Alpha => gl::SRC1_ALPHA,
        OneMinusSrc1Alpha => gl::ONE_MINUS_SRC1_ALPHA,
    }
}

fn map_blend_op(operation: pso::BlendOp) -> (gl::types::GLenum, gl::types::GLenum, gl::types::GLenum) {
    match operation {
        pso::BlendOp::Add { src, dst }    => (gl::FUNC_ADD,              map_factor(src), map_factor(dst)),
        pso::BlendOp::Sub { src, dst }    => (gl::FUNC_SUBTRACT,         map_factor(src), map_factor(dst)),
        pso::BlendOp::RevSub { src, dst } => (gl::FUNC_REVERSE_SUBTRACT, map_factor(src), map_factor(dst)),
        pso::BlendOp::Min => (gl::MIN, gl::ZERO, gl::ZERO),
        pso::BlendOp::Max => (gl::MAX, gl::ZERO, gl::ZERO),
    }
}

pub(crate) fn bind_blend(gl: &GlContainer, desc: &pso::ColorBlendDesc) {
    use hal::pso::ColorMask as Cm;

    match desc.1 {
        pso::BlendState::On { color, alpha } => unsafe {
            let (color_eq, color_src, color_dst) = map_blend_op(color);
            let (alpha_eq, alpha_src, alpha_dst) = map_blend_op(alpha);
            #[cfg(not(target_arch = "wasm32"))]
            gl.Enable(gl::BLEND);
            #[cfg(not(target_arch = "wasm32"))]
            gl.BlendEquationSeparate(color_eq, alpha_eq);
            #[cfg(not(target_arch = "wasm32"))]
            gl.BlendFuncSeparate(color_src, color_dst, alpha_src, alpha_dst);
            #[cfg(target_arch = "wasm32")]
            unimplemented!();
        },
        pso::BlendState::Off => unsafe {
            #[cfg(not(target_arch = "wasm32"))]
            gl.Disable(gl::BLEND);
            #[cfg(target_arch = "wasm32")]
            unimplemented!();
        },
    };

    unsafe {
        #[cfg(not(target_arch = "wasm32"))]
        gl.ColorMask(
            desc.0.contains(Cm::RED) as _,
            desc.0.contains(Cm::GREEN) as _,
            desc.0.contains(Cm::BLUE) as _,
            desc.0.contains(Cm::ALPHA) as _,
        );
        #[cfg(target_arch = "wasm32")]
        unimplemented!();
    }
}

pub(crate) fn bind_blend_slot(gl: &GlContainer, slot: ColorSlot, desc: &pso::ColorBlendDesc) {
    use hal::pso::ColorMask as Cm;

    match desc.1 {
        pso::BlendState::On { color, alpha } => unsafe {
            let (color_eq, color_src, color_dst) = map_blend_op(color);
            let (alpha_eq, alpha_src, alpha_dst) = map_blend_op(alpha);
            //Note: using ARB functions as they are more compatible
            #[cfg(not(target_arch = "wasm32"))]
            gl.Enablei(gl::BLEND, slot as _);
            #[cfg(not(target_arch = "wasm32"))]
            gl.BlendEquationSeparateiARB(slot as _, color_eq, alpha_eq);
            #[cfg(not(target_arch = "wasm32"))]
            gl.BlendFuncSeparateiARB(slot as _, color_src, color_dst, alpha_src, alpha_dst);
            //#[cfg(target_arch = "wasm32")]
            //unimplemented!();
        },
        pso::BlendState::Off => unsafe {
            #[cfg(not(target_arch = "wasm32"))]
            gl.Disablei(gl::BLEND, slot as _);
            //#[cfg(target_arch = "wasm32")]
            //unimplemented!();
        },
    };

    unsafe {
        #[cfg(not(target_arch = "wasm32"))]
        gl.ColorMaski(slot as _,
            desc.0.contains(Cm::RED) as _,
            desc.0.contains(Cm::GREEN) as _,
            desc.0.contains(Cm::BLUE) as _,
            desc.0.contains(Cm::ALPHA) as _,
        );
        //#[cfg(target_arch = "wasm32")]
        //unimplemented!();
    }
}

pub(crate) fn unlock_color_mask(gl: &GlContainer) {
    unsafe {
        #[cfg(not(target_arch = "wasm32"))]
        gl.ColorMask(gl::TRUE, gl::TRUE, gl::TRUE, gl::TRUE);
        #[cfg(target_arch = "wasm32")]
        gl.color_mask(true, true, true, true);
    }
}

pub(crate) fn set_blend_color(gl: &GlContainer, color: pso::ColorValue) {
    unsafe {
        #[cfg(not(target_arch = "wasm32"))]
        gl.BlendColor(color[0], color[1], color[2], color[3]);
        #[cfg(target_arch = "wasm32")]
        gl.blend_color(color[0], color[1], color[2], color[3]);
    }
}
