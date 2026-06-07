#include "../config.cuh"

#include <optix.h>

#include "../params/per_ray_data.cuh"
#include "../device/shader_common.cuh"
#include "../device/random_number_generator.cuh"
#include "../../include/launch_params.h"
#include <stdio.h>

extern "C" __global__ void __raygen__renderFrame()
{
    // // ピクセル位置の取得
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const int accumID = optixLaunchParams.frame.accumID;
    const int frameID = optixLaunchParams.frame.frameID;
    const auto &camera = optixLaunchParams.camera;

    const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;


    // PRD: per-ray-data
    PRD prd;
    // 乱数をピクセル位置とフレーム数をシード値として初期化
    prd.random.init(fbIndex, optixLaunchParams.frame.frameID);
    
    uint32_t u0, u1;
    packPointer( &prd, u0, u1);

    float3 pixelColor = make_float3(0.f); 
    float3 pixelNormal = make_float3(0.f);
    float3 pixelAlbedo = make_float3(0.f);

    for(int sampleID = 0; sampleID < optixLaunchParams.frame.numPixelSamples; sampleID ++){

        // prd.emission = make_float3(0.f);
        prd.contribution = make_float3(0.f);
        prd.albedo   = make_float3(1.f);
        prd.continueTrace = true;
        prd.bounce = 0;
        prd.pdf.bxdf = 1.0f;
        prd.pdf.light = 1.0f;

        // スクリーン空間上のサンプル点をサブピクセル精度でサンプリング
        const float2 screen = make_float2(
            ((float)(optixLaunchParams.frame.size.x - ix - 1) + prd.random()) / (float)optixLaunchParams.frame.size.x, 
            ((float)(optixLaunchParams.frame.size.y - iy - 1) + prd.random()) / (float)optixLaunchParams.frame.size.y 
        );


        // Attention! : レンズ系の direct callable 関数は，プログラム全体でオフセット：0
        const LensRay ray = 
            optixDirectCall<LensRay, const float2, const float, const float>(optixLaunchParams.camera.cameraMode, screen, prd.random(), prd.random());

        prd.position = ray.org;
        prd.wi = ray.dir;

        

        while(prd.bounce < optixLaunchParams.frame.maxBounce)
        {
            optixTrace( optixLaunchParams.traversable,
                prd.position,
                prd.wi,
                1e-4f,    // tmin
                1e20f,  // tmax
                0.0f,   // rayTime
                OptixVisibilityMask( 255 ),
                // OPTIX_RAY_FLAG_NONE,
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                RADIANCE_RAY_TYPE,
                RAY_TYPE_COUNT,
                RADIANCE_RAY_TYPE,
                u0, u1);

            if(!prd.continueTrace){
                break;
            }
            prd.bounce ++;
        }

        pixelColor = pixelColor + (prd.contribution * ray.weight - pixelColor) / (sampleID + 1.0f);
        pixelNormal = pixelNormal + (prd.primaryNormal - pixelNormal) / (sampleID + 1.0f);
        pixelAlbedo = pixelAlbedo + (prd.primaryAlbedo - pixelAlbedo) / (sampleID + 1.0f);

    }
    pixelColor = clamp(pixelColor, 0.0f, 1000.0f);
    if(!isfinite(pixelColor.x) || !isfinite(pixelColor.y) || !isfinite(pixelColor.z)){
        pixelColor = make_float3(0.0f);
    }

    float4 rgba = make_float4(pixelColor, 1.0f);

    
    if (optixLaunchParams.frame.accumID > 0){
        float4 rgba_now = optixLaunchParams.frame.colorBuffer[fbIndex];
        rgba = rgba_now + (rgba - rgba_now) / (optixLaunchParams.frame.accumID + 1.0f); 
    }
    
    optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
    optixLaunchParams.frame.albedoBuffer[fbIndex] = make_float4(pixelAlbedo, 1.0f);
    optixLaunchParams.frame.normalBuffer[fbIndex] = make_float4(pixelNormal, 1.0f);
}
