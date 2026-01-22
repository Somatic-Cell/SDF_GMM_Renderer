#include "../config.cuh"

#include <optix.h>

#include "../params/per_ray_data.cuh"
#include "../device/shader_common.cuh"
#include "../device/random_number_generator.cuh"
#include "../../include/launch_params.h"

extern "C" __device__ float3 __direct_callable__bsdf_glass_sample(const float3 wo, const IntersectedData& material, PRD* prd)
{
    prd->lastHitMaterialType = MATERIAL_TYPE_GLASS;

    float cosThetaI = wo.y;
    float eta = 1.0f / material.ior;
    float denom = 1.0 + eta;
    float numer = 1.0 - eta;
    
    float r = numer / denom;

    float3 N = make_float3(0.f, 1.0f, 0.f);
    
    if(cosThetaI < 0.0f){
        N = -N;
        cosThetaI = -cosThetaI;
    }

    if(
        fabsf(prd->position.z - 1.0f) < 0.02f 
    )
    {
        prd->pdf.bxdf = 1.0f;
        return normalize(-1.0f * wo);
    }

    float F0 = r * r;
    // float F0 = 0.2f;
    float fresnel = schlick(wo, N, F0);

    // if(fabsf(prd->position.z) < 0.05f 
    //     || fabsf(prd->position.x - 1.0f) < 0.04f 
    //     || fabsf(prd->position.x) < 0.04f 
    //     || fabsf(prd->position.y) < 0.05f)
    // {
    //     if(prd->random() < fresnel){
    //         prd->pdf.bxdf = fresnel;
    //         return normalize(- 1.0f * wo + N * 2.f * dot(N, wo));
    //     } else {
    //         prd->pdf.bxdf = 1.0f - fresnel;
    //         return normalize(-1.0f * wo);   
    //     }
    // }



    float sin2ThetaT = (1.0f - cosThetaI * cosThetaI) * eta * eta;

    if(sin2ThetaT > 1.0f)
    {
        prd->pdf.bxdf = 1.0f;
        return normalize(- 1.0f * wo + N * 2.f * dot(N, wo));
    }
    else if(prd->random() < fresnel){
        prd->pdf.bxdf = fresnel;
        return normalize(- 1.0f * wo + N * 2.f * dot(N, wo));
    } else {
        float cosThetaT = sqrtf(fmaxf(1.0f - sin2ThetaT, 1e-7f));
        // prd->albedo *= (eta * eta);
        prd->albedo *= material.baseColor;
        prd->pdf.bxdf = 1.0f - fresnel;
        return normalize(eta * (-1.0f * wo) + (eta * cosThetaI - cosThetaT) * N);   
    }

}

extern "C" __device__ float3 __direct_callable__bsdf_glass_eval(const float3 wi, const float3 wo, const IntersectedData& material, PRD* prd)
{
    prd->pdf.bxdf = 0.0f;
    prd->lastHitMaterialType = MATERIAL_TYPE_GLASS;
    return make_float3(0.0f);
}