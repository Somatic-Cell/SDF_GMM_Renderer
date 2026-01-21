#include "../config.cuh"

#include <optix.h>

#include "../params/per_ray_data.cuh"
#include "../device/shader_common.cuh"
#include "../device/random_number_generator.cuh"
#include "../../include/launch_params.h"
#include <stdio.h>

extern "C" __device__ float3 __direct_callable__brdf_principled_sample(const float3 wo, const IntersectedData& material, PRD* prd)
{
    float3 wi;
    float3 wm;
    const float roughness = material.roughness;
    const float alpha = fmaxf(roughness * roughness, 1e-3f);
    const float metallic = material.metallic;
    const float3 baseColor = material.baseColor;
    const float weightDiffuse = fmaxf(1.0f - metallic, 0.f);
    const float weightSpecular = 1.0f;

    // MIS
    const float weightSum = weightDiffuse + weightSpecular;
    const float normalizedWeightDiffuse  = weightDiffuse / weightSum;
    const float normalizedWeightSpecular = weightSpecular / weightSum;

    float pdf;
    if(normalizedWeightDiffuse > prd->random()){
        // diffuse
        wi = randomCosineHemisphere(prd->random(), prd->random());
    } else {
        // specular
        wm = visibleNormalSampling(alpha, wo, prd->random(), prd->random());
        wi = normalize(-1.0f * wo + 2.0f * dot(wo, wm) * wm);
    }

    const float diffusePdf = getLambertPdf(wo, wi);
    const float specularPdf = getGGXPdf(alpha, wo, wi);

    pdf = normalizedWeightDiffuse * diffusePdf + normalizedWeightSpecular * specularPdf;
    // printf("Diffuse PDF %f, Specular PDF: %f \n", diffusePdf, specularPdf);
    

    prd->lastHitMaterialType = MATERIAL_TYPE_PRINCIPLED_BRDF;
    prd->pdf.bxdf = pdf;
    return wi;

}

extern "C" __device__ float3 __direct_callable__brdf_principled_eval(const float3 wi, const float3 wo, const IntersectedData& material, PRD* prd)
{

    // NaN を発生させない
    if(wi.y < 1e-7f){
        return make_float3(0.0f);
    }
    
    const float roughness = material.roughness;
    const float alpha = fmaxf(roughness * roughness, 1e-3f);
    const float metallic = material.metallic;
    const float3 baseColor = material.baseColor;
    const float weightDiffuse = fmaxf(1.0f - metallic, 0.f);
    const float weightSpecular = 1.0f;

    // MIS
    const float weightSum = weightDiffuse + weightSpecular;
    const float normalizedWeightDiffuse  = weightDiffuse / weightSum;
    const float normalizedWeightSpecular = weightSpecular / weightSum;

    const float diffusePdf = getLambertPdf(wo, wi);
    const float specularPdf = getGGXPdf(alpha, wo, wi);

    float pdf = normalizedWeightDiffuse * diffusePdf + normalizedWeightSpecular * specularPdf;

    prd->pdf.bxdf = pdf;

    const float3 F0 = lerp(make_float3(0.04f), baseColor, metallic);

    const float3 diffuseColor = evalLambertBRDF(baseColor, wo, wi);
    const float3 specularColor = evalSpecularBRDF(alpha, F0, wo, wi);

    prd->lastHitMaterialType = MATERIAL_TYPE_PRINCIPLED_BRDF;
    float3 color = diffuseColor * weightDiffuse + specularColor;
    return color;
}