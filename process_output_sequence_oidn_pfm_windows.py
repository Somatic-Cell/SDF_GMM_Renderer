import argparse
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Frame:
    number: int
    width: int
    path: Path


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def find_executable(name_or_path: Optional[str], fallback_names: List[str]) -> str:
    if name_or_path:
        p = Path(name_or_path)
        if p.exists():
            return str(p)
        found = shutil.which(name_or_path)
        if found:
            return found
        fail(f"実行ファイルが見つかりません: {name_or_path}")

    for name in fallback_names:
        found = shutil.which(name)
        if found:
            return found

    fail("実行ファイルが PATH から見つかりません。候補: " + ", ".join(fallback_names))


def find_ffmpeg(ffmpeg_arg: Optional[str]) -> str:
    if ffmpeg_arg:
        return find_executable(ffmpeg_arg, ["ffmpeg.exe", "ffmpeg"])

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    found = shutil.which("ffmpeg.exe") or shutil.which("ffmpeg")
    if found:
        return found

    fail(
        "ffmpeg が見つかりません。"
        " python -m pip install imageio-ffmpeg を実行するか、"
        " --ffmpeg で ffmpeg.exe を指定してください。"
    )


def collect_frames(input_dir: Path) -> Tuple[List[Frame], int, int]:
    if not input_dir.exists():
        fail(f"入力ディレクトリが存在しません: {input_dir}")
    if not input_dir.is_dir():
        fail(f"入力パスがディレクトリではありません: {input_dir}")

    pattern = re.compile(r"^(\d+)\.png$", re.IGNORECASE)
    frames: List[Frame] = []

    for p in input_dir.iterdir():
        if not p.is_file():
            continue
        m = pattern.match(p.name)
        if not m:
            continue
        stem = m.group(1)
        frames.append(Frame(number=int(stem), width=len(stem), path=p))

    if not frames:
        fail(f"{input_dir} 直下に 000.png のような連番 PNG が見つかりません。")

    frames.sort(key=lambda f: f.number)

    widths = {f.width for f in frames}
    if len(widths) != 1:
        examples = ", ".join(f.path.name for f in frames[:10])
        fail("ゼロ埋め桁数が混在しています。例: " + examples)

    numbers = [f.number for f in frames]
    if len(numbers) != len(set(numbers)):
        fail("同じ番号の PNG が複数あります。例: 000.png と 0.png の混在など。")

    start = numbers[0]
    end = numbers[-1]
    missing = sorted(set(range(start, end + 1)) - set(numbers))
    if missing:
        fail("連番に欠番があります。欠番例: " + ", ".join(map(str, missing[:20])))

    return frames, start, frames[0].width


def ensure_outputs(
    raw_dir: Path,
    denoised_dir: Path,
    temp_dir: Path,
    video_path: Path,
    zip_path: Path,
    overwrite: bool,
    dry_run: bool,
) -> None:
    for d in [raw_dir, denoised_dir, temp_dir]:
        if d.exists():
            if overwrite:
                if not dry_run:
                    shutil.rmtree(d)
            else:
                fail(f"{d} が既に存在します。上書きする場合は --overwrite を付けてください。")

    for p in [video_path, zip_path]:
        if p.exists():
            if overwrite:
                if not dry_run:
                    p.unlink()
            else:
                fail(f"{p} が既に存在します。上書きする場合は --overwrite を付けてください。")

    if not dry_run:
        raw_dir.mkdir(parents=True, exist_ok=False)
        denoised_dir.mkdir(parents=True, exist_ok=False)
        temp_dir.mkdir(parents=True, exist_ok=False)


def show_command(cmd: Union[List[str], str]) -> str:
    if isinstance(cmd, list):
        return subprocess.list2cmdline([str(x) for x in cmd])
    return cmd


def run_command(cmd: Union[List[str], str], dry_run: bool = False) -> None:
    print("> " + show_command(cmd))
    if dry_run:
        return

    if isinstance(cmd, str):
        subprocess.run(cmd, check=True, shell=True)
    else:
        subprocess.run(cmd, check=True)


def read_png_as_float_rgb_and_alpha(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with Image.open(path) as im:
        has_alpha = im.mode in ("RGBA", "LA") or ("transparency" in im.info)

        if has_alpha:
            rgba = im.convert("RGBA")
            arr = np.asarray(rgba)
            rgb = arr[:, :, :3].astype(np.float32) / 255.0
            alpha = arr[:, :, 3].copy()
            return rgb, alpha

        rgb_im = im.convert("RGB")
        arr = np.asarray(rgb_im)
        rgb = arr.astype(np.float32) / 255.0
        return rgb, None


def write_pfm(path: Path, rgb: np.ndarray) -> None:
    if rgb.dtype != np.float32:
        rgb = rgb.astype(np.float32)

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        fail(f"PFM 書き出しには HxWx3 の RGB float32 配列が必要です: {path}")

    h, w, _ = rgb.shape

    with path.open("wb") as f:
        f.write(b"PF\n")
        f.write(f"{w} {h}\n".encode("ascii"))
        f.write(b"-1.0\n")  # negative scale means little-endian

        data = np.flipud(np.ascontiguousarray(rgb)).astype("<f4", copy=False)
        data.tofile(f)


def _read_non_comment_line(f) -> bytes:
    while True:
        line = f.readline()
        if not line:
            fail("PFM の読み込み中に予期せず EOF に到達しました。")
        line = line.strip()
        if not line.startswith(b"#"):
            return line


def read_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        header = _read_non_comment_line(f)
        if header == b"PF":
            channels = 3
        elif header == b"Pf":
            channels = 1
        else:
            fail(f"PFM ヘッダが不正です: {path}")

        dims = _read_non_comment_line(f).decode("ascii").split()
        if len(dims) != 2:
            fail(f"PFM の解像度行が不正です: {path}")
        w, h = int(dims[0]), int(dims[1])

        scale = float(_read_non_comment_line(f).decode("ascii"))
        endian = "<" if scale < 0 else ">"

        data = np.fromfile(f, dtype=endian + "f4")

    expected = w * h * channels
    if data.size != expected:
        fail(f"PFM のデータサイズが不正です: {path}: expected={expected}, actual={data.size}")

    if channels == 3:
        img = data.reshape((h, w, 3))
    else:
        gray = data.reshape((h, w, 1))
        img = np.repeat(gray, 3, axis=2)

    img = np.flipud(img)
    return np.ascontiguousarray(img.astype(np.float32, copy=False))


def save_float_rgb_to_png(path: Path, rgb: np.ndarray, alpha: Optional[np.ndarray]) -> None:
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb8 = np.round(rgb * 255.0).astype(np.uint8)

    if alpha is not None:
        if alpha.shape[:2] != rgb8.shape[:2]:
            fail(f"alpha の解像度がデノイズ結果と一致しません: {path}")
        rgba8 = np.dstack([rgb8, alpha])
        Image.fromarray(rgba8, mode="RGBA").save(path)
    else:
        Image.fromarray(rgb8, mode="RGB").save(path)


def convert_png_to_pfm(input_png: Path, output_pfm: Path) -> Optional[np.ndarray]:
    rgb, alpha = read_png_as_float_rgb_and_alpha(input_png)
    write_pfm(output_pfm, rgb)
    return alpha


def convert_pfm_to_png(input_pfm: Path, output_png: Path, alpha: Optional[np.ndarray]) -> None:
    rgb = read_pfm(input_pfm)
    save_float_rgb_to_png(output_png, rgb, alpha)


def copy_raw_frames(frames: List[Frame], raw_dir: Path, dry_run: bool) -> None:
    print(f"[copy raw] {raw_dir}")
    if dry_run:
        return

    for frame in frames:
        dst = raw_dir / frame.path.name
        if dst.exists():
            fail(f"コピー先に同名ファイルがあります: {dst}")
        shutil.copy2(frame.path, dst)


def build_oidn_command(
    args: argparse.Namespace,
    denoiser_exe: str,
    input_pfm: Path,
    output_pfm: Path,
) -> Union[List[str], str]:
    if args.denoise_cmd_template:
        return args.denoise_cmd_template.format(
            input=str(input_pfm),
            output=str(output_pfm),
        )

    if args.oidn_input_kind == "ldr":
        input_flag = "--ldr"
    elif args.oidn_input_kind == "hdr":
        input_flag = "--hdr"
    else:
        fail(f"未知の --oidn-input-kind です: {args.oidn_input_kind}")

    cmd: List[str] = [denoiser_exe]

    if args.device:
        cmd += ["--device", args.device]

    cmd += [input_flag, str(input_pfm)]

    if args.srgb:
        cmd.append("--srgb")

    if args.quality:
        cmd += ["--quality", args.quality]

    cmd += ["-o", str(output_pfm)]

    return cmd


def denoise_frames_via_pfm(
    args: argparse.Namespace,
    frames: List[Frame],
    denoised_dir: Path,
    temp_dir: Path,
) -> None:
    denoiser_exe = find_executable(args.denoiser, ["oidnDenoise.exe", "oidnDenoise"])

    total = len(frames)
    for i, frame in enumerate(frames, start=1):
        in_pfm = temp_dir / f"{frame.path.stem}_in.pfm"
        out_pfm = temp_dir / f"{frame.path.stem}_out.pfm"
        out_png = denoised_dir / frame.path.name

        print(f"[denoise {i}/{total}] {frame.path.name}")

        alpha = None
        if not args.dry_run:
            alpha = convert_png_to_pfm(frame.path, in_pfm)

        cmd = build_oidn_command(args, denoiser_exe, in_pfm, out_pfm)
        run_command(cmd, dry_run=args.dry_run)

        if not args.dry_run:
            if not out_pfm.exists():
                fail(f"OIDN の出力 PFM が生成されませんでした: {out_pfm}")

            convert_pfm_to_png(out_pfm, out_png, alpha)

            if not out_png.exists():
                fail(f"デノイズ後 PNG が生成されませんでした: {out_png}")

            if not args.keep_temp:
                try:
                    in_pfm.unlink()
                except FileNotFoundError:
                    pass
                try:
                    out_pfm.unlink()
                except FileNotFoundError:
                    pass


def make_video(
    args: argparse.Namespace,
    denoised_dir: Path,
    start_number: int,
    pad_width: int,
    video_path: Path,
) -> None:
    ffmpeg_exe = find_ffmpeg(args.ffmpeg)

    pattern = f"%0{pad_width}d.png" if pad_width > 1 else "%d.png"
    input_pattern = str(denoised_dir / pattern)

    cmd = [
        ffmpeg_exe,
        "-y" if args.overwrite else "-n",
        "-framerate",
        str(args.fps),
        "-start_number",
        str(start_number),
        "-i",
        input_pattern,
    ]

    if args.pad_even:
        cmd += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]

    cmd += [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        args.preset,
        "-crf",
        str(args.crf),
        "-r",
        str(args.fps),
        str(video_path),
    ]

    print(f"[video] {video_path.name}")
    run_command(cmd, dry_run=args.dry_run)

    if not args.dry_run and not video_path.exists():
        fail(f"動画ファイルが生成されませんでした: {video_path}")


def add_dir_to_zip(zf: zipfile.ZipFile, directory: Path, base_dir: Path) -> None:
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            zf.write(p, p.relative_to(base_dir).as_posix())


def make_zip(
    raw_dir: Path,
    denoised_dir: Path,
    video_path: Path,
    zip_path: Path,
    base_dir: Path,
    dry_run: bool,
) -> None:
    print(f"[zip] {zip_path.name}")
    if dry_run:
        return

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        add_dir_to_zip(zf, raw_dir, base_dir)
        add_dir_to_zip(zf, denoised_dir, base_dir)
        zf.write(video_path, video_path.relative_to(base_dir).as_posix())


def cleanup_dirs(raw_dir: Path, denoised_dir: Path, temp_dir: Path, args: argparse.Namespace) -> None:
    if args.keep_folders:
        print("[cleanup] --keep-folders が指定されたため raw/ と denoised/ を残します。")
    else:
        print("[cleanup] raw/ と denoised/ を削除します。元の連番 PNG は output/ 直下に残します。")
        if not args.dry_run:
            shutil.rmtree(raw_dir)
            shutil.rmtree(denoised_dir)

    if args.keep_temp:
        print("[cleanup] --keep-temp が指定されたため一時 PFM ディレクトリを残します。")
    else:
        print("[cleanup] 一時 PFM ディレクトリを削除します。")
        if not args.dry_run and temp_dir.exists():
            shutil.rmtree(temp_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "output/000.png, 001.png, ... を残したまま raw/ にコピーし、"
            "PNG -> PFM -> OIDN -> PFM -> PNG で denoised/ を生成し、"
            "50 fps MP4 と zip を作成します。"
            "既定では zip 作成後に raw/ と denoised/ と一時 PFM を削除します。"
        )
    )

    parser.add_argument("--input-dir", default="output", help="連番 PNG が入っているディレクトリ。既定値: output")
    parser.add_argument("--denoiser", required=True, help="oidnDenoise.exe のパス")
    parser.add_argument("--ffmpeg", default=None, help="ffmpeg.exe のパス。未指定なら imageio-ffmpeg または PATH から探索")

    parser.add_argument("--fps", type=int, default=50, help="出力 MP4 の fps。既定値: 50")
    parser.add_argument("--video-name", default="denoised_50fps.mp4", help="作成する MP4 ファイル名")
    parser.add_argument("--zip-name", default="result_package.zip", help="作成する zip ファイル名")
    parser.add_argument("--temp-dir-name", default="_pfm_tmp", help="一時 PFM ディレクトリ名")

    parser.add_argument(
        "--oidn-input-kind",
        choices=["ldr", "hdr"],
        default="ldr",
        help="OIDN に渡す入力種別。PNG 由来の 0..1 画像なら通常 ldr。既定値: ldr",
    )
    parser.add_argument(
        "--denoise-cmd-template",
        default=None,
        help=(
            "OIDN コマンドを完全に自分で指定する場合に使います。"
            " {input} と {output} が PFM パスに置換されます。"
            " 例: '\"C:\\path\\oidnDenoise.exe\" --ldr \"{input}\" --srgb -o \"{output}\"'"
        ),
    )
    parser.add_argument("--quality", default="high", help="公式 oidnDenoise 用 quality。例: high, balanced, fast")
    parser.add_argument("--device", default=None, help="公式 oidnDenoise 用 device。例: cpu, cuda, sycl, default")
    parser.add_argument("--srgb", action="store_true", help="PNG が sRGB エンコードの場合に指定。OIDN に --srgb を渡します。")

    parser.add_argument("--crf", type=int, default=18, help="libx264 の CRF。小さいほど高画質・大容量。既定値: 18")
    parser.add_argument("--preset", default="medium", help="libx264 の preset。既定値: medium")
    parser.add_argument("--no-pad-even", dest="pad_even", action="store_false", help="偶数解像度への padding を無効化")
    parser.set_defaults(pad_even=True)

    parser.add_argument("--overwrite", action="store_true", help="既存の raw/, denoised/, 一時 PFM, mp4, zip を上書き")
    parser.add_argument("--keep-folders", action="store_true", help="zip 作成後も raw/ と denoised/ を残す")
    parser.add_argument("--keep-temp", action="store_true", help="一時 PFM ディレクトリを残す")
    parser.add_argument("--dry-run", action="store_true", help="処理内容だけ表示し、実際には実行しない")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    raw_dir = input_dir / "raw"
    denoised_dir = input_dir / "denoised"
    temp_dir = input_dir / args.temp_dir_name
    video_path = input_dir / args.video_name
    zip_path = input_dir / args.zip_name

    frames, start_number, pad_width = collect_frames(input_dir)

    print(f"[input] {input_dir}")
    print(f"[frames] {len(frames)}")
    print(f"[range] {frames[0].path.name} .. {frames[-1].path.name}")
    print(f"[oidn] input kind = {args.oidn_input_kind}, srgb = {args.srgb}")
    print(f"[temp] {temp_dir}")

    ensure_outputs(
        raw_dir=raw_dir,
        denoised_dir=denoised_dir,
        temp_dir=temp_dir,
        video_path=video_path,
        zip_path=zip_path,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    copy_raw_frames(frames, raw_dir, args.dry_run)
    denoise_frames_via_pfm(args, frames, denoised_dir, temp_dir)
    make_video(args, denoised_dir, start_number, pad_width, video_path)
    make_zip(raw_dir, denoised_dir, video_path, zip_path, input_dir, args.dry_run)
    cleanup_dirs(raw_dir, denoised_dir, temp_dir, args)

    print("[done]")
    print(f"original frames remain in: {input_dir}")
    print(f"video: {video_path}")
    print(f"zip:   {zip_path}")
    if args.keep_folders:
        print(f"raw:      {raw_dir}")
        print(f"denoised: {denoised_dir}")
    if args.keep_temp:
        print(f"temp pfm: {temp_dir}")


if __name__ == "__main__":
    main()
