#include "../config.cuh"

#include <optix.h>

#include "../params/per_ray_data.cuh"
#include "../device/shader_common.cuh"
#include "../device/random_number_generator.cuh"
#include "../../include/launch_params.h"
#include <stdio.h>

extern "C" __global__ void __closesthit__radiance()
{

    const TriangleMeshSBTData &sbtData =*(const TriangleMeshSBTData*) optixGetSbtDataPointer();
    PRD &prd = *getPRD<PRD>();

    // 基本的な交差点の情報を取得
    const int primID = optixGetPrimitiveIndex();
    const uint3 index = sbtData.index[primID];
    const float2 uv = optixGetTriangleBarycentrics();
    const float u = uv.x;
    const float v = uv.y;

    // レイの進行方向を取得
    const float3 rayDirection = normalize(optixGetWorldRayDirection());
    // レイとプリミティブの交差点を取得
    const float rayLength = optixGetRayTmax();
    const float3 intersectedPoint = optixGetWorldRayOrigin() + rayLength * rayDirection;

    // 形状処理用. 面法線を取得
    prd.instanceID = sbtData.instanceID;
    const float3 &V1 = optixTransformPointFromObjectToWorldSpace(sbtData.vertex[index.x]);
    const float3 &V2 = optixTransformPointFromObjectToWorldSpace(sbtData.vertex[index.y]);
    const float3 &V3 = optixTransformPointFromObjectToWorldSpace(sbtData.vertex[index.z]);
    
    float3 Ng = normalize(cross(V2-V1, V3-V1));
    const float triangleArea = 0.5f * fmaxf(length(cross(V2 - V1, V3 - V1)), 1e-7f);



    // Diffuse テクスチャ座標を取得
    const float2 &UVDiffuse1 = sbtData.diffuseTexcoord[index.x];
    const float2 &UVDiffuse2 = sbtData.diffuseTexcoord[index.y];
    const float2 &UVDiffuse3 = sbtData.diffuseTexcoord[index.z];

    const float2 diffuseTextureCoordinate = (1.0f - u - v) * UVDiffuse1
        + u * UVDiffuse2
        + v * UVDiffuse3;

    // Normal テクスチャ座標を取得
    const float2 &UVNormal1 = sbtData.normalTexcoord[index.x];
    const float2 &UVNormal2 = sbtData.normalTexcoord[index.y];
    const float2 &UVNormal3 = sbtData.normalTexcoord[index.z];
    
    const float2 normalTextureCoordinate = (1.0f - u - v) * UVNormal1
        + u * UVNormal2
        + v * UVNormal3;

    // Emissive テクスチャ座標を取得
    const float2 &UVEmissive1 = sbtData.emissiveTexcoord[index.x];
    const float2 &UVEmissive2 = sbtData.emissiveTexcoord[index.y];
    const float2 &UVEmissive3 = sbtData.emissiveTexcoord[index.z];
    const float2 emissiveTextureCoordinate = (1.0f - u - v) * UVEmissive1
        + u * UVEmissive2
        + v * UVEmissive3;

    // シェーディング用．頂点法線があれば頂点法線を，なければ面法線を使用
    
    // mymath::matrix3x3 matrixO2WNormal = optixLaunchParams.frame.normalMatrixBuffer[optixGetInstanceId()];
    float3 Ns = Ng;
    if(sbtData.hasNormal){
        const float3 &N1 = sbtData.normal[index.x];
        const float3 &N2 = sbtData.normal[index.y];
        const float3 &N3 = sbtData.normal[index.z];
        Ns =    (1.0f - u - v) * N1
                +            u * N2
                +            v * N3;
        Ns = optixTransformNormalFromObjectToWorldSpace(Ns);
    }
    Ns = normalize(Ns);
    
    float3 nml = make_float3(0.0f, 0.0f, 1.0f);
    if(sbtData.normalTexture.hasTexture) {
        float4 fromTexture = tex2D<float4>(sbtData.normalTexture.texture, normalTextureCoordinate.x, 1.0f - normalTextureCoordinate.y);
        nml = (make_float3(fromTexture) - make_float3(0.5f)) * 2.0f ;
        nml.y = -nml.y;
        nml.z = sqrtf(fmaxf(1.0f - nml.y * nml.y - nml.x * nml.x, 0.0001f));
    }

    // 接ベクトル空間の計算
    float4 tan = (fabsf(Ns.x) > 0.999f) ? make_float4(0.0f, 1.0f, 0.0f, 1.0f) : make_float4(1.0f, 0.0f, 0.0f, 1.0f);
    if(sbtData.hasTangent){
        const float4 &T1 = sbtData.tangent[index.x];
        const float4 &T2 = sbtData.tangent[index.y];
        const float4 &T3 = sbtData.tangent[index.z];
        tan =  (1.0f - u - v) * T1
            +            u * T2
            +            v * T3;
        tan = make_float4(optixTransformNormalFromObjectToWorldSpace(make_float3(tan)), tan.w);
    }

    
    const float3 T = normalize(make_float3(tan));
    const float3 B = normalize(cross(Ns, T) * tan.w);
    
    Ns = normalize(T * nml.x + B * nml.y + Ns * nml.z);
    float3 biNormal = normalize(cross(T, Ns)); 
    float3 tangent = normalize(cross(Ns, biNormal));
    
    // Albedo の計算
    float3 albedo = sbtData.color;
    const int instID = optixGetInstanceId();
    if(instID == 3){ // water
        albedo = make_float3(1.0f);
    } 
    if(instID == 1){ // wall
        // albedo = make_float3(15.0f, 48.0f, 24.0f) / 255.f; // 緑
        albedo = make_float3(97.0f, 240.0f, 250.0f) / 255.f; // 水色 (dam などのシーン)
        // albedo = make_float3(1.0f); 
    }
    float density[6] = {0.99f, 1.00f, 1.05f, 1.15f, 1.25f, 1.5f};
    if(instID > 3){
        // albedo = make_float3(1.0f, 0.0f, 0.0f); // 多数のパーティクルがあるシーンでランダムに色を割りふる
        albedo = instanceIdToRGB(instID * 3); // 多数のパーティクルがあるシーンでランダムに色を割りふる
        // albedo = make_float3(0.4549019f, 0.20784f, 0.9490196f); // 紫
    }
    
    if(sbtData.diffuseTexture.hasTexture){
        float4 fromTexture = tex2D<float4>(
            sbtData.diffuseTexture.texture, diffuseTextureCoordinate.x, 1.0f - diffuseTextureCoordinate.y);
        albedo = make_float3(fromTexture);
    }

    float4 arm = make_float4(0.0f, 0.01f, 0.0f, 0.0f);
    
    if(sbtData.rmTexture.hasTexture){
        arm = tex2D<float4>(sbtData.rmTexture.texture, diffuseTextureCoordinate.x, 1.0f - diffuseTextureCoordinate.y);
    }

    IntersectedData matData;
    matData.ior = sbtData.ior; // glass
    matData.metallic = arm.z;
    matData.roughness = arm.y;
    matData.baseColor = albedo;

    if(dot(rayDirection, Ng) > 0.f && sbtData.materialType != MATERIAL_TYPE_LIGHT) {
        Ng = -Ng;
        Ns = -Ns;
        tangent = -tangent;
        biNormal = normalize(cross(Ns, tangent));
        // biNormal = -biNormal;
        matData.ior = 1.0f / matData.ior; // glass
    }

    prd.position = intersectedPoint; // 後で置き場を考える

    // メッシュの光源と交差した場合の処理
    if(sbtData.materialType == MATERIAL_TYPE_LIGHT)
    {
        const float cosTheta = fabsf(dot(-1.0f * rayDirection, Ng));
        if(cosTheta > 1e-7f)
        {
            float3 emission = sbtData.emissive;
            // if(sbtData.emissiveTexture.hasTexture){
            //     float4 fromTexture = tex2D<float4>(
            //         sbtData.emissiveTexture.texture, emissiveTextureCoordinate.x, 1.0f - emissiveTextureCoordinate.y);
            //     emission = make_float3(fromTexture);
            // }
            emission *= optixLaunchParams.light.lightIntensityFactor;


            if(prd.bounce != 0)
            {
                const float r = rayLength;
                const float pSelect = 1.0f / float(optixLaunchParams.light.numLights);
                float pArea = 1.0f / (triangleArea);
                // const float geometricTerm = cosTheta / (rayLength * rayLength);
                const float pdfLight =  pSelect * pArea * (r * r) / cosTheta;
                
                const float weight = balanceHeuristicWeight(1, fmaxf(prd.pdf.bxdf, 1e-7f), 1, fmaxf(pdfLight, 1e-7f));
                prd.contribution += emission * prd.albedo * weight;
            } else {
                prd.contribution += emission * prd.albedo;
            }
        }
            

        if(prd.bounce == 0){
            prd.primaryNormal = Ns;
            prd.primaryAlbedo = albedo;
        }
        prd.continueTrace = false;
        return;
    }


    const float3 woLocal = worldToLocal(-1.0f * rayDirection, tangent, Ns, biNormal);
    // ガラスのメッシュと交差した場合の処理（レイトレーシングを続行）
    if(sbtData.materialType == MATERIAL_TYPE_GLASS)
    {
        int callableFunctionOffset = NUM_LENS_TYPE + MATERIAL_TYPE_GLASS * 2; // 登録した順番が lens の後で，glass * 2 が 対応するマテリアルの sample 関数
        float3 wiLocal = optixDirectCall<float3, const float3, const IntersectedData& , PRD*>(callableFunctionOffset, woLocal, matData, &prd);
        prd.wi = localToWorld(wiLocal, tangent, Ns, biNormal);
        prd.continueTrace = true;
        
        prd.position += (wiLocal.y > 0.0f) ? Ng * 1e-3f : -1.0f * Ng * 1e-3f;

        if(prd.bounce == 0){
            prd.primaryNormal = Ns;
            prd.primaryAlbedo = albedo;
        }
        return;
    }


    const int numLights = optixLaunchParams.light.numLights; // 後で置き場を考える
    // MIS によるシェーディング
    // ライトサンプリング
    if (numLights > 0){

        //MIS
        const int indexLight = (numLights > 1) ? clamp(static_cast<int>(floor(prd.random() * (float)numLights)), 0, numLights - 1) : 0;
        LightDefinition light = optixLaunchParams.light.lightDefinition[indexLight];

        const int callLightType = NUM_LENS_TYPE + NUM_BXDF + light.lightType;
        LightSample lightSample = optixDirectCall<LightSample, LightDefinition, PRD*>(callLightType, light, &prd);
        const float3 wiLocal = worldToLocal(lightSample.direction, tangent, Ns, biNormal);

        if(lightSample.pdf > 0.f && wiLocal.y > 0.0f)
        {
            ShadowPRD shadowPrd;
            uint32_t u0, u1;
            packPointer(&shadowPrd, u0, u1);
            

            // BSDF の計算
            int callableFunctionOffset = NUM_LENS_TYPE + sbtData.materialType * 2 + 1; // 登録した順番が lens の後で，materialType * 2 + 1 が 対応するマテリアルの eval 関数 
            float3 bxdfValue = optixDirectCall<float3, const float3, const float3, const IntersectedData& , PRD*>(callableFunctionOffset, wiLocal, woLocal, matData, &prd);
            float bxdfPdf = prd.pdf.bxdf;
            const float3 wi = localToWorld(wiLocal, tangent, Ns, biNormal); 


            // // 光源へ接続して可視性を判断
            optixTrace( 
                optixLaunchParams.traversable,
                intersectedPoint, // 出射位置
                wi,
                1e-3f,
                lightSample.distance - 1e-3f,
                0.0f,
                OptixVisibilityMask( 255 ),
                OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
                    | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                SHADOW_RAY_TYPE,    // SBT offset
                RAY_TYPE_COUNT,     // SBT stride
                SHADOW_RAY_TYPE,    // miss SBT Index
                u0, u1
            );

            if(shadowPrd.visible){
                float weight = balanceHeuristicWeight(1, lightSample.pdf, 1, bxdfPdf);
                prd.contribution += prd.albedo * lightSample.emission * bxdfValue * wiLocal.y * weight / fmaxf(lightSample.pdf, 1e-7f);
            }
        }
        prd.continueTrace = true;
    }

    
    // 次の方向の決定
    // BSDF のサンプリング とアルベドの変更
    int callableFunctionOffset = NUM_LENS_TYPE + sbtData.materialType * 2;
    const float3 sampledWiLocal = optixDirectCall<float3, const float3, const IntersectedData&, PRD*>(callableFunctionOffset, woLocal, matData, &prd);
    const float3 sampledWi = localToWorld(sampledWiLocal, tangent, Ns, biNormal);
    
    callableFunctionOffset = NUM_LENS_TYPE + sbtData.materialType * 2 + 1;
    const float3 bxdfValue = optixDirectCall<float3, const float3, const float3, const IntersectedData& , PRD*>(callableFunctionOffset, sampledWiLocal, woLocal, matData, &prd);
    
    // if(sampledWiLocal.y < 1e-7f){
    //     prd.continueTrace = false;
    //     return;
    // }
    
    prd.position += 1e-3f * Ng;
    prd.albedo *= bxdfValue * sampledWiLocal.y / fmaxf(prd.pdf.bxdf, 1e-7f);
    prd.wi = sampledWi; 
    prd.continueTrace = true;
    
    if(prd.bounce == 0){
        prd.primaryNormal = Ns;
        prd.primaryAlbedo = albedo;
    }
}