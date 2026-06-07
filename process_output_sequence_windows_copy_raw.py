import argparse
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union


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
    video_path: Path,
    zip_path: Path,
    overwrite: bool,
) -> None:
    if raw_dir.exists():
        if overwrite:
            shutil.rmtree(raw_dir)
        else:
            fail(f"{raw_dir} が既に存在します。上書きする場合は --overwrite を付けてください。")

    if denoised_dir.exists():
        if overwrite:
            shutil.rmtree(denoised_dir)
        else:
            fail(f"{denoised_dir} が既に存在します。上書きする場合は --overwrite を付けてください。")

    for p in [video_path, zip_path]:
        if p.exists():
            if overwrite:
                p.unlink()
            else:
                fail(f"{p} が既に存在します。上書きする場合は --overwrite を付けてください。")

    raw_dir.mkdir(parents=True, exist_ok=False)
    denoised_dir.mkdir(parents=True, exist_ok=False)


def show_command(cmd: Union[List[str], str]) -> str:
    if isinstance(cmd, list):
        return subprocess.list2cmdline([str(x) for x in cmd])
    return cmd


def run_command(cmd: Union[List[str], str], dry_run: bool = False) -> None:
    print("> " + show_command(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def build_denoise_command(
    args: argparse.Namespace,
    denoiser_exe: str,
    input_path: Path,
    output_path: Path,
) -> Union[List[str], str]:
    if args.denoise_cmd_template:
        return args.denoise_cmd_template.format(
            input=str(input_path),
            output=str(output_path),
        )

    if args.denoiser_kind == "oidn":
        cmd = [denoiser_exe, "--ldr", str(input_path)]
        if args.srgb:
            cmd.append("--srgb")
        if args.quality:
            cmd += ["--quality", args.quality]
        if args.device:
            cmd += ["--device", args.device]
        cmd += ["-o", str(output_path)]
        return cmd

    if args.denoiser_kind == "legacy":
        return [
            denoiser_exe,
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "-hdr",
            "0",
            "-srgb",
            "1" if args.srgb else "0",
        ]

    fail(f"未知の --denoiser-kind です: {args.denoiser_kind}")


def copy_raw_frames(frames: List[Frame], raw_dir: Path, dry_run: bool) -> None:
    print(f"[copy raw] {raw_dir}")
    if dry_run:
        return

    for frame in frames:
        dst = raw_dir / frame.path.name
        if dst.exists():
            fail(f"コピー先に同名ファイルがあります: {dst}")
        shutil.copy2(frame.path, dst)


def denoise_frames(args: argparse.Namespace, frames: List[Frame], denoised_dir: Path) -> None:
    denoiser_exe = find_executable(args.denoiser, ["oidnDenoise.exe", "oidnDenoise"])

    total = len(frames)
    for i, frame in enumerate(frames, start=1):
        out_path = denoised_dir / frame.path.name
        print(f"[denoise {i}/{total}] {frame.path.name}")

        cmd = build_denoise_command(args, denoiser_exe, frame.path, out_path)
        run_command(cmd, dry_run=args.dry_run)

        if not args.dry_run and not out_path.exists():
            fail(f"デノイズ後のファイルが生成されませんでした: {out_path}")


def make_video(
    args: argparse.Namespace,
    denoised_dir: Path,
    start_number: int,
    pad_width: int,
    video_path: Path,
) -> None:
    ffmpeg_exe = find_executable(args.ffmpeg, ["ffmpeg.exe", "ffmpeg"])

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
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
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


def cleanup_dirs(raw_dir: Path, denoised_dir: Path, keep_folders: bool, dry_run: bool) -> None:
    if keep_folders:
        print("[cleanup] --keep-folders が指定されたため raw/ と denoised/ を残します。")
        return

    print("[cleanup] raw/ と denoised/ を削除します。元の連番 PNG は output/ 直下に残します。")
    if dry_run:
        return

    shutil.rmtree(raw_dir)
    shutil.rmtree(denoised_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "output/000.png, 001.png, ... を残したまま raw/ にコピーし、"
            "denoised/ にデノイズ結果を作り、50 fps MP4 と zip を作成します。"
            "既定では zip 作成後に raw/ と denoised/ を削除します。"
        )
    )

    parser.add_argument("--input-dir", default="output", help="連番 PNG が入っているディレクトリ。既定値: output")
    parser.add_argument("--denoiser", default=None, help="oidnDenoise.exe などのパス。未指定なら PATH から探索")
    parser.add_argument("--ffmpeg", default=None, help="ffmpeg.exe のパス。未指定なら PATH から探索")
    parser.add_argument("--fps", type=int, default=50, help="出力 MP4 の fps。既定値: 50")
    parser.add_argument("--video-name", default="denoised_50fps.mp4", help="作成する MP4 ファイル名")
    parser.add_argument("--zip-name", default="result_package.zip", help="作成する zip ファイル名")

    parser.add_argument(
        "--denoiser-kind",
        choices=["oidn", "legacy"],
        default="oidn",
        help="oidn: 公式 oidnDenoise 形式。legacy: Denoiser.exe -i/-o 形式。",
    )
    parser.add_argument(
        "--denoise-cmd-template",
        default=None,
        help='任意のデノイズコマンド。例: \'"C:\\path\\oidnDenoise.exe" --ldr "{input}" --srgb -o "{output}"\'',
    )
    parser.add_argument("--quality", default="high", help="公式 oidnDenoise 用 quality。例: high, balanced, fast")
    parser.add_argument("--device", default=None, help="公式 oidnDenoise 用 device。例: cpu, cuda, sycl")
    parser.add_argument("--srgb", action="store_true", help="LDR PNG が sRGB エンコードの場合に指定")
    parser.add_argument("--overwrite", action="store_true", help="既存の raw/, denoised/, mp4, zip を上書き")
    parser.add_argument("--keep-folders", action="store_true", help="zip 作成後も raw/ と denoised/ を残す")
    parser.add_argument("--dry-run", action="store_true", help="処理内容だけ表示し、実際には実行しない")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    raw_dir = input_dir / "raw"
    denoised_dir = input_dir / "denoised"
    video_path = input_dir / args.video_name
    zip_path = input_dir / args.zip_name

    frames, start_number, pad_width = collect_frames(input_dir)

    print(f"[input] {input_dir}")
    print(f"[frames] {len(frames)}")
    print(f"[range] {frames[0].path.name} .. {frames[-1].path.name}")

    ensure_outputs(
        raw_dir=raw_dir,
        denoised_dir=denoised_dir,
        video_path=video_path,
        zip_path=zip_path,
        overwrite=args.overwrite,
    )

    copy_raw_frames(frames, raw_dir, args.dry_run)
    denoise_frames(args, frames, denoised_dir)
    make_video(args, denoised_dir, start_number, pad_width, video_path)
    make_zip(raw_dir, denoised_dir, video_path, zip_path, input_dir, args.dry_run)
    cleanup_dirs(raw_dir, denoised_dir, args.keep_folders, args.dry_run)

    print("[done]")
    print(f"original frames remain in: {input_dir}")
    print(f"video: {video_path}")
    print(f"zip:   {zip_path}")
    if args.keep_folders:
        print(f"raw:      {raw_dir}")
        print(f"denoised: {denoised_dir}")


if __name__ == "__main__":
    main()
