#include "../config.cuh"

#include <optix.h>

#include "../params/per_ray_data.cuh"
#include "../device/shader_common.cuh"
#include "../device/random_number_generator.cuh"
#include "../../include/launch_params.h"
#include <stdio.h>

extern "C" __global__ void __anyhit__radiance()
{
    const TriangleMeshSBTData &sbtData =*(const TriangleMeshSBTData*) optixGetSbtDataPointer();
    PRD &prd = *getPRD<PRD>();
    
    float opacity = 1.0f;     
    if(sbtData.diffuseTexture.hasTexture){
        // 基本的な交差点の情報を取得
        const int primID = optixGetPrimitiveIndex();
        const uint3 index = sbtData.index[primID];
        const float u = optixGetTriangleBarycentrics().x;
        const float v = optixGetTriangleBarycentrics().y;

        // Diffuse テクスチャ座標を取得
        const float2 &UVDiffuse1 = sbtData.diffuseTexcoord[index.x];
        const float2 &UVDiffuse2 = sbtData.diffuseTexcoord[index.y];
        const float2 &UVDiffuse3 = sbtData.diffuseTexcoord[index.z];

        const float2 diffuseTextureCoordinate = (1.0f - u - v) * UVDiffuse1
            + u * UVDiffuse2
            + v * UVDiffuse3;

        opacity = tex2D<float4>(sbtData.diffuseTexture.texture, diffuseTextureCoordinate.x, 1.0f - diffuseTextureCoordinate.y).w;
    }

    if(opacity < 1.0f && opacity <= prd.random()){
        optixIgnoreIntersection();
        return;
    }

    // const int instID = optixGetInstanceId();
    // // レイの進行方向を取得
    // const float3 rayDirection = normalize(optixGetWorldRayDirection());
    // // レイとプリミティブの交差点を取得
    // const float rayLength = optixGetRayTmax();
    // const float3 intersectedPoint = optixGetWorldRayOrigin() + rayLength * rayDirection;
    // if(instID == 3){
    //     if(fabsf(intersectedPoint.x - 0.5f) > 0.25f || fabsf(intersectedPoint.z - 0.5f) > 0.25f || intersectedPoint.y < 0.2f){
    //         optixIgnoreIntersection();
    //         return;
    //     }
    // }
}
