# SDFGMMRenderer v.0.1

某研究結果をレンダリングするためのレンダラ．\
NVIDIA CUDA / OptiX を使用．

- [依存関係](#依存関係)
- [フォルダやファイルの説明](#フォルダやファイルの説明)
- [準備](#準備)
    1. [ダウンロード](#1-ダウンロード)
    1. [ビルド](#2-ビルド)
    1. [その他](#3-その他)
- [実行](#実行)
- [レンダラの機能](#レンダラの機能)

## 依存関係

### 要事前インストール
#### GPU レンダリング用 SDK
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) ( >= v.12.0)
- [NVIDIA OptiX](https://developer.nvidia.com/designworks/optix/download) (デフォルトでは 9.1.0)

### その他使用する外部ライブラリ　（個別にダウンロードする必要なし．ライセンス管理のために列挙）
- [Open Asset Import Library (assimp)](https://github.com/assimp/assimp?tab=readme-ov-file#open-asset-import-library-assimp) ：アセットの読み込み
- [DirectXTex texture processing library](https://github.com/microsoft/DirectXTex)：``` .DDS ``` ファイルの読み込み
- [fpng](https://github.com/richgel999/fpng)：爆速で .png ファイルを生成する
- [GLFW](https://github.com/glfw/glfw)：ウィンドウの描画
- [Dear ImGui](https://github.com/ocornut/imgui)：GUI の描画
- [stb](https://github.com/nothings/stb)：テクスチャの読み込み，出力画像の保存
- [JSON for Modern C++](https://github.com/nlohmann/json?tab=readme-ov-file)：シーンファイルの読み書き

## フォルダやファイルの説明

このプロジェクトは，以下のように構成されている：
```
.
├── envmap      # 環境マップを入れる (.hdr 形式)
├── ext         # 外部ライブラリの置き場
├── include     # ヘッダファイルの置き場
├── kernels     # OptiX 外で使用する CUDA ファイルの置き場
├── model       # 使用したいメッシュの置き場 (使用できる拡張子：".fbx", ".obj", ".gltf", ".glb", ".ply")，シミュレーション結果の置き場
├── output      # レンダリングした結果を保存した際の出力先
├── scene       # シーンファイル（後述）の置き場
├── shaders     # OptiX に関係する CUDA ファイルの置き場
    ├── callable    # Direct callable 関数が実装されたプログラムの置き場
    ├── device      # シェーダ全体で使用する便利ツールの置き場       
    ├── entry       # レイトラバーサルのためのシェーダの置き場
    └── params      # PRD など
├── src         # ソースファイルの置き場
├── utils       # 便利ツールの置き場
├── viewer      # OpenGL を使った，レンダリング過程のビューワ
├── build.bat   # コマンドプロンプトなどから簡単にビルドするためのバッチファイル
└── execute.bat # コマンドプロンプトなどから簡単に実行するためのバッチファイル 
```

## 準備 (Windows 環境)

### 1. ダウンロード

コマンドプロンプトなどを開き，作業したいディレクトリ上で，本リポジトリをローカル環境にクローンする：

```
$ git clone --recursive https://github.com/Somatic-Cell/PhotonicRT.git 
```
【注意】 サブモジュールを含めてクローンするために，```--recursive``` オプションを指定． \
これを指定すれば，[使用する外部ライブラリ](#使用する外部ライブラリ) を個別にダウンロードする必要がない．


### 2. ビルド

#### プロジェクトのビルド方法

[CMake](https://cmake.org/download/) を用いる．

<details><summary>CMake のインストール方法</summary>

[リンク](https://cmake.org/download/) 先から Windows x64 版をインストールすること．\
インストールできているかどうかを確認したい場合は
```
$ cmake
```
と実行すればわかる．

また，パスが通っているかどうかを確認したければ

```
$ where cmake
```
とすれば確認できる．

</details>

以下の手順に従ってビルドする（Release ビルドの場合）：
```
$ mkdir build
$ cd build
$ cmake .. 
$ cmake --build . --config Release --verbose
$ cd ..
```
クリーンビルドをしたい場合は，上記を実行する前に，以下を実行して build ディレクトリを削除する：
```
$ rmdir /S build
```
これらのコマンドを毎回打ち込むのは面倒くさい．\
上記一式のコマンドは，``` build.bat ``` ファイルにまとめて記述してあるので，プロジェクトのルートディレクトリで，単に
```
$ build.bat
```
と実行すれば全て自動で行ってくれる．

#### バージョン指定
<details><summary> GPU のアーキテクチャ 指定</summary>

``` CMakeLists.txt ``` では，[CUDA GPU compute capability](https://developer.nvidia.com/cuda-gpus) を ``` CMAKE_CUDA_ARCHITECTURES ``` で指定する．
実行する環境に合わせて，適宜変更すること．

Compute capability の例

| 世代 | 機種名 | Compute capability |
| --- | --- | :---: |
| Ampere | GeForce RTX 30系統 | 86 |
| Ada | GeForce RTX 40系統 | 89 |
| Blackwell | GeForce RTX 50系統 | 120 |

</details>


<details><summary> (Release / Debug) ビルドのバージョン指定</summary>

```Debug``` / ```Release``` ビルドを切り替えたい場合は，単に上記のビルドで
```
$ cmake --build . --config Release --verbose
```
を
```
$ cmake --build . --config Debug --verbose
```
に置き換えればよい．
実行時間を計測する際は，必ず ```Release``` を指定すること．

</details>

<details><summary>OptiX のバージョン変更</summary>

OptiX のバージョンを変更したい場合は，プロジェクトのルートディレクトリにある 
 ```CMakeLists.txt``` を編集する．
デフォルトでは OptiX 9.1.0 で動作するようになっているが，変更したい場合は，

```
find_package(OptiX91)

if(OptiX9_FOUND)
    set(OPTIX_INCLUDE_DIR "${OPTIX9_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "OptiX SDK 9.1.0 not found.")
endif()
```
あたりの，OptiX91 を OptiX(バージョン) に変更すればよい．
対応しているバージョンは，```utils/cmake/FindOptiX*.cmake``` というファイルがあるもの．

対応していないバージョンに対応させたい場合は，```utils/cmake/FindOptiX(対応させたいver).cmake``` を自作する．
といっても，既存の .cmake ファイルをコピペして名前を変更し，

```
if (WIN32) 
    # Windows 環境だったら． 
    # WIN32 は Windows 環境だったら True を返すことに注意．　
    # 32bit OS でも 64bit OS でも True を返す 
    set(OPTIX8_INSTALL_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0" CACHE PATH "Path to OptiX installed location.")
endif()
```
で指定されているディレクトリを別のものに変更するだけ

</details>


### 3. ビルドに失敗したら
外部ライブラリのビルドに失敗することがある (``` fpng ```, ``` json ``` など？)．

この場合，当該ライブラリの ``` cmake_minimum_required() ``` を
```
cmake_minimum_required(VERSION 3.5)
```
に書き換えてしまうことで，ビルドを通すことができる．

## 実行

### シーンファイルを記述する
<details><summary>シーンファイルについて</summary>

JSON 形式のファイルを使用して，使用したいメッシュやカメラのパラメータ，レンダラの設定などをまとめて指定する．\
シーンファイルを ``` /scene ``` フォルダに置く．
### シーンファイルの構成
- ``` objects ``` : 使用したいモデルデータを列挙する．複数列挙可能
    - ``` name ``` : モデルの名前．特にレンダラでは使用されない．JSON ファイルを読み書きするユーザ向けの変数
    - ``` type ``` : モデルのタイプ．``` "mesh" ``` か ``` "volume" ``` で指定．
    - ``` file ``` : ファイル名．``` "mesh" ``` の場合，``` /model/ ``` 下のパスを記述．
    - ``` TRS ``` : オブジェクト全体を併進 (transform), 回転 (rotation), スケール (scale) するためのパラメータ．回転はクォータニオンで表現
- ``` camera ``` : カメラデータ
    - ``` from ``` : カメラの位置 
    - ``` at ``` : カメラが注目する座標
    - ``` up ``` : カメラの上向きを指定するベクトル
    - ``` focalLength ``` : 焦点距離．``` PINHOLE ``` モードでは無効
    - ``` fValue ``` : F値．``` PINHOLE ``` モードでは無効
    - ``` fov ``` : 視野角．``` THIN_LENS ``` モードでは無効
    - ``` sensitivity ``` : 疑似的な ISO 値．``` PINHOLE ``` モードでは無効
    - ``` pintDist ``` : 焦点を合わせたい物体までの距離．``` PINHOLE ``` モードでは無効
- ``` environment ``` : 環境マップに関する情報．
    - ``` file ``` : ファイル名．``` /envmap/ ``` 下のパスを記述．現在は ``` .hdr ``` ファイルのみ対応
- ``` integrator ``` : レンダラの設定
    - ``` type ``` : 光学計算のアルゴリズムの指定．現在は ``` "Path Tracing" ``` のみ対応
    - ``` applySpectralRendering ``` : スペクトラルレンダリングを実行するかどうかを boolean で指定
    - ``` spp ``` : カーネルが一回起動するたびに何サンプル行うか
    - ``` maxBounce ``` : パストレーシングでレイを追跡する際の最大反射回数

</details>

* ``` dam ``` 系など，既にカメラパラメータを調整済みのシーンファイルがあるため，類似したシーンでは類似したシーンを参照すると調整が楽．
* 新規シーンのカメラパラメータなどを探索したい場合は，低 spp に設定したうえでレンダラを実行し，理想のカメラ位置を探索する．GUI 上にカメラパラメータが表示されるのでそれを参考にシーンファイルを書き換える．
* 探索中，カメラ姿勢の変化が入力に対して敏感すぎると感じる場合は，キーボード上で ``` + / - ``` を連打することで sensitivity を調整できる

### シミュレーションデータを置く
レンダリングしたいシミュレーションデータ一式を ``` model ``` 下にそのまま置く．例えば，``` model/comp_going1_bunny_mesh_N50 ```


### レンダリングに必要な各種データを置く
環境マップのための HDR ファイルはデータが大きいため，Git で追跡していない．
[Poly Haven](https://polyhaven.com/hdris) でダウンロードして，```envmap``` 下に置く．

### ソースコードを一部変更する

#### 参照するフォルダのパスの変更（毎回必ず変更）
* ``` include/application.hpp ```: 1か所 (130 行目付近)
* ``` src/renderer.cpp ```: 2か所 (367行目, 1,749 行目付近)

#### マテリアルの変更（シーンのカテゴリごとに変更）
* ``` src/render.cpp ```: 1,018 行目付近．Diffuse から Disney Principled BRDF などに変更したいときに変更．シーンファイルに列挙した順に modelIndex が振られている.

* ``` shaders/entry/ch_radiance.cu ```: 110 行目付近．マテリアルの色味を変えたいときに変更．パーティクルを多数インスタンスするときは，ID をランダムな RGB 値にマッピングする関数が用意してある．

#### 光源の強さの変更（シーンにのカテゴリごとに変更）
* ``` square_light.mtl ```. 緑と紫のシーンでは 4.0, ダムなどのシーンでは 2,000 位に設定するとちょうどいい

#### パーティクルの位置の微調整（シミュレーションの格子数が変わるたびに変更）
シミュレーション結果のパーティクルの位置が，格子の解像度によって微妙に床から浮いてしまう現象を微調整する． ``` 1/(格子数) ``` だけ ``` inst.pos[2] ``` を下げる．
* ``` include/solid_particle_frame_reader.hpp ```:
　
    * ```static void makeOptixTransformRowMajor3x4``` (142 行目付近):

#### レンダリングするフレームの指定（偶数フレームのみなど，場合によって変更）
* ``` src/renderer.cpp ```: (1,095 行目付近)
    ```
    m_launchParams.frame.frameID += 1;
    ```
    を
    ```
    m_launchParams.frame.frameID += 2;
    ```
    とすれば偶数フレームのみなどに変更可能
### ビルドする

ビルドセクションを参照してビルドする．毎回クリーンビルドする必要はない．

### 低サンプル数で実行して様子を見る

描画したメッシュが，レンダリング後半で壁にめり込む場合がある．不整合がないか，``` 4spp ``` 程度で実行してみる．
例えば，
```
build\bin\Release\SDFGMMRT.exe "comp_going1_bunny_mesh_N50.json" 2>err.log
```
とすると，log を ``` err.log ``` に保存しつつ実行可能．

実行用の ``` execute.bat ``` も用意しているので，単にプロジェクトのルートディレクトリで
``` 
.\execute.bat 
``` 
を実行してもいい．

### 実行する

プロジェクトのルートディレクトリで
```
build\bin\Release\SDFGMMRT.exe "comp_going1_bunny_mesh_N50.json" 2>err.log
```
または
``` 
.\execute.bat 
``` 
を実行．


## 実行後にデータをまとめる（デノイズ・動画作成）
実行後のデータを集約し，デノイズする．
ルートディレクトリで
```
python .\process_output_sequence_oidn_pfm_windows.py `
--input-dir .\output `
--denoiser "C:\Tools\oidn-2.5.0.x64.windows\bin\oidnDenoise.exe" `
--srgb `
--overwrite `
--video-name "comp_going1_bunny_mesh_N50_v2.mp4" ` 
--zip-name "comp_going1_bunny_mesh_N50_result_v2.zip"
```
などと実行する．```--input-dir``` 下に zip ファイルが生成される

オプション：
* ```--input-dir```: 連番画像が入っているフォルダ
* ```--denoiser```: [Intel Open Image Denoise](https://www.openimagedenoise.org/) をダウンロードし，exe ファイルまでのパスを指定
* ```--srgb```: 色空間
* ```--overwrite```: 既に zip ファイルが存在する場合でも上書きする
* ```--video-name```: ビデオ名
* ```--zip-name```: zip ファイル名
* ```--keep-folders ```: zip 作成後も raw/ と denoised/ を残す

その他は help で確認

## レンダラの機能
Coming soon... 

## 実験開発用メモ
Coming soon... 

### デバッグ
Coming soon... 
