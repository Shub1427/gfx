#include <metal_stdlib>
using namespace metal;

typedef struct {
    float4 src_coords [[attribute(0)]];
    float4 dst_coords [[attribute(1)]];
} TextureBlitAttributes;

typedef struct {
    float4 position [[position]];
    float4 uv;
    uint layer [[render_target_array_index]];
} VertexData;

vertex VertexData vs_blit(TextureBlitAttributes in [[stage_in]]) {
    float4 pos = { 0.0, 0.0, 0.0f, 1.0f };
    pos.xy = in.dst_coords.xy * 2.0 - 1.0;
    return VertexData { pos, in.src_coords, uint(in.dst_coords.z) };
}

fragment float4 ps_blit(
    VertexData in [[stage_in]],
    texture2d<float> tex2D [[ texture(0) ]],
    sampler sampler2D [[ sampler(0) ]]
) {
  return tex2D.sample(sampler2D, in.uv.xy, level(in.uv.w));
}

fragment float4 ps_blit_array(
    VertexData in [[stage_in]],
    texture2d_array<float> tex2DArray [[ texture(0) ]],
    sampler sampler2D [[ sampler(0) ]]
) {
  return tex2DArray.sample(sampler2D, in.uv.xy, uint(in.uv.z), level(in.uv.w));
}

kernel void cs_fill_buffer(
    device uint *buffer [[ buffer(0) ]],
    constant uint &value [[ buffer(1) ]],
    uint index [[ thread_position_in_grid ]]
) {
    buffer[index] = value;
}
