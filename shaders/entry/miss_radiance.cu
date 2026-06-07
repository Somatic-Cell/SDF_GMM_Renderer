#include "../config.cuh"

#include <optix.h>

#include "../params/per_ray_data.cuh"
#include "../device/shader_common.cuh"
#include "../device/random_number_generator.cuh"
#include "../../include/launch_params.h"
#include <stdio.h>


// radiance ray が交差しない = 環境マップにあたる
extern "C" __global__ void __miss__radiance()
{
    PRD &prd = *getPRD<PRD>(); 
    
    float u, v;
    orthogonalToUVCoord(prd.wi, &u, &v);
    float3 emission =  make_float3(tex2D<float4>(optixLaunchParams.envMap, u, v));
    
    
    if(prd.bounce == 0){
        prd.contribution += prd.albedo * emission;
        prd.continueTrace = false;
        return;
    }

    const int wp = optixLaunchParams.envMapInfo.patchSize.x;
    const int hp = optixLaunchParams.envMapInfo.patchSize.y;

    int col = min(max(int(u * (float)wp), 0), wp -1);
    int row = min(max(int(v * (float)hp), 0), hp -1);

    const int patchIndex = row * wp + col;

    float th0, th1, ph0, ph1;
    thetaPhiFromPatch(row, col, wp, hp, &th0, &th1, &ph0, &ph1);

    const float deltaPhi = ph1 - ph0;
    const float deltaCosT = cosf(th0) - cosf(th1);
    const float dOmega = deltaPhi *deltaCosT;

    const float totalWeight = optixLaunchParams.envMapInfo.totalWeight;
    float pdfEnvDir = 0.0f;

    if(totalWeight > 0.0f && dOmega > 0.0f){
        const float pPatch = optixLaunchParams.envMapInfo.patchWeight[patchIndex] / totalWeight;
        pdfEnvDir = pPatch * (1.0f / dOmega);
    }

    const float pSelected = 1.0f / float(optixLaunchParams.light.numLights);
    const float pdfLight = pdfEnvDir * pSelected;

    const float weight = 
    (prd.lastHitMaterialType == MATERIAL_TYPE_GLASS) 
    ? 1.0f 
    : balanceHeuristicWeight(1, fmaxf(prd.pdf.bxdf, 1e-7f), 1, fmaxf(pdfLight, 1e-7f));
    
    emission *= weight; 
    prd.contribution += emission * prd.albedo;
    prd.continueTrace = false;
}