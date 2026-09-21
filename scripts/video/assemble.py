#!/usr/bin/env python3
"""成片组装器：逐帧烧录角标/条件条/字幕 → 九段拼接 → 编码 + 音轨合成。

设计约束（paper/video/plan.md §1/§4/§6）：
- 165 s / 4950 帧 @ 30 fps，槽位表与 make_narration.py 的 SEGS 完全一致
  （S3 末态驻留 +3 s、S9 结尾 +2 s，用户粗剪二审反馈）；
  音频起点 = 槽位起点 + 0.3 s（HEAD_GAP 约定，narration_timings.json）。
- 烧录规则（诚实呈现）：
  * 实拍段（MuJoCo 渲染）左上常驻 "MuJoCo simulation" 角标；
    动画段自带 "Schematic" 角标（底部条带），不再叠加。
  * 慢放段右上标注倍速（0.25× / 0.5× / 0.10× playback）。
  * S5 实拍顶部条件条：变体名 + 12 m/s serve · same seed。
  * 字幕全段烧录（subtitles.srt，白字黑描边，位于动画黑条带上方）。
  * S1 开头叠论文标题；S9 结尾匿名标题卡（论文指引 + 匿名代码链接 + 复现欢迎语）。
- 编码：H.264 yuv420p，CRF+VBV（默认 crf18 / maxrate 950k / bufsize 1900k），
  音频 AAC 64k，`-map_metadata -1` + faststart（双盲合规）。

用法：
    python scripts/video/assemble.py                     # 全片组装
    python scripts/video/assemble.py --crf 20            # 码率紧时提 crf
    python scripts/video/assemble.py --extract 5,30,60   # 额外抽 QC 帧
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "paper" / "video" / "assets"
VIDEO_DIR = REPO / "paper" / "video"
OUT = VIDEO_DIR / "icra2027_video.mp4"

FPS = 30
W, H = 1280, 720
HEAD_GAP = 0.3
SUB_SIZE = 24          # 字幕单行字号（用户粗剪反馈：30px 两行 → 24px 单行）
SUB_MAX_W = W - 56     # 字幕最大行宽（左右各留 28 px）
SUB_MIN_SIZE = 17      # 超宽 cue 自动缩字下限

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONT_B = str(FONT_DIR / "DejaVuSans-Bold.ttf")
FONT_R = str(FONT_DIR / "DejaVuSans.ttf")

TITLE = ("Relaxing the Point Target: A Cost-Relaxation Framework "
         "for Robust Robotic Ball Hitting")
ANON_URL = "https://anonymous.4open.science/r/Tennis_robot_789741"  # 与论文脚注同口径
INK = (13, 15, 19)
GREY_T = (222, 224, 229)

logger = logging.getLogger("assemble")


def ffmpeg_exe() -> str:
    """imageio-ffmpeg 自带二进制（系统无 ffmpeg）。"""
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_duration(path: Path) -> float:
    """解析媒体时长（无 ffprobe，从 ffmpeg -i stderr 提取）。"""
    r = subprocess.run([ffmpeg_exe(), "-i", str(path)], capture_output=True,
                       text=True, check=False)
    m = re.search(r"Duration: (\d+):(\d+):(\d+)\.(\d+)", r.stderr)
    if not m:
        raise RuntimeError(f"无法解析时长: {path}")
    h, mi, s, cs = (int(g) for g in m.groups())
    return h * 3600 + mi * 60 + s + cs / 100.0


# ── 槽位表：sid → [(素材路径, 烧录标志), ...]；"card" 为程序化结尾卡 ────────
@dataclass
class Clip:
    """一段源素材及其烧录标志（badge=仿真角标，speed=慢放倍速标签，bar=条件条）。"""

    path: Path | None = None       # None → 程序化生成帧（结尾卡）
    badge: bool = False
    speed: str | None = None
    bar: str | None = None
    title: bool = False            # S1 标题叠底
    frames: int = 0                # 运行时统计


def slot_clips() -> dict[str, list[Clip]]:
    """九段槽位的素材序列（时长见 probe，与分镜表 §3 对齐）。"""
    a = ASSETS
    return {
        "S1": [Clip(a / "s1_clean.mp4", badge=True, title=True),
               Clip(a / "s1_slow_clean.mp4", badge=True, speed="0.25× playback"),
               Clip(a / "s1_slow_contact_clean.mp4", badge=True,
                    speed="0.25× playback")],
        "S2": [Clip(a / "s2_method.mp4")],
        "S3": [Clip(a / "s3_window.mp4")],
        "S4": [Clip(a / "s4_tube.mp4")],
        "S5": [Clip(a / "s5_pt_contact_clean.mp4", badge=True,
                    speed="0.10× playback",
                    bar="point-target · 12 m/s serve · same seed"),
               Clip(a / "s5_full_contact_clean.mp4", badge=True,
                    speed="0.10× playback",
                    bar="full: window + tube · 12 m/s serve · same seed"),
               Clip(a / "s5_chart.mp4")],
        "S6": [Clip(a / "s6_perturb.mp4")],
        "S7": [Clip(a / "s7_limit.mp4")],
        "S8": [Clip(a / "s8_timing.mp4")],
        "S9": [Clip(a / "s9_clean.mp4", badge=True, speed="0.5× playback"),
               Clip(a / "s9_contact_clean.mp4", badge=True,
                    speed="0.14× playback"),
               Clip(None)],  # 结尾匿名标题卡（补齐槽位剩余时长）
    }


SLOTS = [("S1", 0.0, 12.0), ("S2", 12.0, 26.0), ("S3", 26.0, 46.0),
         ("S4", 46.0, 59.0), ("S5", 59.0, 87.0), ("S6", 87.0, 102.0),
         ("S7", 102.0, 129.0), ("S8", 129.0, 150.0), ("S9", 150.0, 165.0)]


# ── 字幕 ────────────────────────────────────────────────────────────────────
@dataclass
class Cue:
    """一条字幕 cue（秒，全片绝对时间）。"""

    t0: float
    t1: float
    text: str


def parse_srt(path: Path) -> list[Cue]:
    """解析 subtitles.srt（make_narration.py 生成的两行折行格式）。"""
    cues, block = [], []

    def ts(s: str) -> float:
        h, m, rest = s.split(":")
        sec, ms = rest.split(",")
        return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000

    for line in path.read_text(encoding="utf-8").splitlines() + [""]:
        if not line.strip():
            if len(block) >= 2:
                t0, t1 = block[1].split("-->")
                cues.append(Cue(ts(t0.strip()), ts(t1.strip()),
                                "\n".join(block[2:]).strip()))
            block = []
        else:
            block.append(line)
    return cues


# ── 叠加层（预渲染 RGBA，逐帧 paste）───────────────────────────────────────
def tag_layer(text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    """半透明黑底 + 白字的小角标图层。"""
    tmp = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    pad = 10
    bb = d.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    x, y = 22, 18
    d.rounded_rectangle((x, y, x + tw + 2 * pad, y + th + 2 * pad), radius=8,
                        fill=(10, 11, 14, 170))
    d.text((x + pad - bb[0], y + pad - bb[1]), text, font=font,
           fill=(255, 255, 255, 235))
    return tmp


def bar_layer(text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    """顶部居中条件条（半透明黑底白字）。"""
    tmp = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    pad = 12
    bb = d.textbbox((0, 0), text, font=font)
    tw = bb[2] - bb[0]
    x0 = (W - tw - 2 * pad) // 2
    d.rounded_rectangle((x0, 14, x0 + tw + 2 * pad, 14 + bb[3] - bb[1] + 2 * pad),
                        radius=8, fill=(10, 11, 14, 170))
    d.text((x0 + pad - bb[0], 14 + pad - bb[1]), text, font=font,
           fill=(255, 255, 255, 235))
    return tmp


def title_layer() -> Image.Image:
    """S1 标题叠底（两行，画面上部居中）。"""
    tmp = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    f = ImageFont.truetype(FONT_B, 31)
    l1, l2 = "Relaxing the Point Target:", \
        "A Cost-Relaxation Framework for Robotic Ball Hitting"
    for i, ln in enumerate((l1, l2)):
        bb = d.textbbox((0, 0), ln, font=f)
        d.text(((W - (bb[2] - bb[0])) // 2, 76 + i * 44), ln, font=f,
               fill=(255, 255, 255, 240), stroke_width=3,
               stroke_fill=(0, 0, 0, 200))
    return tmp


def card_frame(alpha: float) -> np.ndarray:
    """S9 结尾匿名标题卡（alpha 0→1 渐入）：标题 + 论文/匿名代码链接 + 复现欢迎语。"""
    img = Image.new("RGB", (W, H), (26, 27, 31))
    d = ImageDraw.Draw(img)
    a = int(255 * alpha)
    f1 = ImageFont.truetype(FONT_B, 34)
    f2 = ImageFont.truetype(FONT_R, 24)
    f3 = ImageFont.truetype(FONT_R, 20)
    f4 = ImageFont.truetype(FONT_R, 18)
    lines = ("Relaxing the Point Target:",
             "A Cost-Relaxation Framework for Robotic Ball Hitting")
    for i, ln in enumerate(lines):
        bb = d.textbbox((0, 0), ln, font=f1)
        d.text(((W - (bb[2] - bb[0])) // 2, 250 + i * 52), ln, font=f1,
               fill=(255, 255, 255, a))
    for txt, f, y in (("Details in the submitted paper", f2, 420),
                      (f"Code & data: {ANON_URL}", f3, 460),
                      ("One-command setup — reproduction welcome", f3, 496),
                      ("Simulated in MuJoCo", f4, 640)):
        bb = d.textbbox((0, 0), txt, font=f)
        d.text(((W - (bb[2] - bb[0])) // 2, y), txt, font=f,
               fill=(*GREY_T, a))
    return np.asarray(img)


# ── 主组装 ──────────────────────────────────────────────────────────────────
def build(crf: int, maxrate: int, extract: str) -> None:
    """逐帧烧录 + 编码 + 音轨合成 + 封装。"""
    timings = json.loads((ASSETS / "narration" / "narration_timings.json")
                         .read_text(encoding="utf-8"))
    assert timings["head_gap_s"] == HEAD_GAP, "HEAD_GAP 约定与音轨不一致"
    cues = parse_srt(VIDEO_DIR / "subtitles.srt")

    f_tag = ImageFont.truetype(FONT_B, 21)
    f_bar = ImageFont.truetype(FONT_B, 24)
    f_sub = ImageFont.truetype(FONT_B, SUB_SIZE)

    clips_map = slot_clips()
    # 预统计各素材帧数（以 30fps 网格取整）并算好每段尾部补帧数
    for sid, _, _ in SLOTS:
        for c in clips_map[sid]:
            if c.path is not None:
                c.frames = round(probe_duration(c.path) * FPS)

    pad: dict[str, int] = {}
    total = 0
    for sid, t0, t1 in SLOTS:
        have = sum(c.frames for c in clips_map[sid])
        want = round((t1 - t0) * FPS)
        if have > want:
            raise RuntimeError(f"{sid} 素材 {have} 帧超过槽位 {want} 帧")
        if sid != "S9" and have < want - 3 * FPS:
            logger.warning("  %s 素材 %d 帧较短，尾部保持末帧 %.1f s",
                           sid, have, (want - have) / FPS)
        pad[sid] = want - have
        total += want
    assert total == round(165 * FPS), f"总帧数 {total} ≠ 4950"
    clips_map["S9"][-1].frames = pad["S9"]  # 结尾卡帧数 = S9 槽位缺口
    logger.info("槽位检查通过：9 段共 %d 帧（165.0 s）", total)

    tmp_video = ASSETS / "_burned_silent.mp4"
    writer = imageio.get_writer(
        tmp_video, fps=FPS, codec="libx264", quality=None, bitrate=None,
        macro_block_size=16,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", str(crf),
                       "-maxrate", f"{maxrate}k", "-bufsize", f"{2 * maxrate}k",
                       "-preset", "medium"])

    t = 0.0
    title_ov = title_layer()
    sub_y = 622  # 字幕单行顶部 y（黑条带上缘 658 之上；动画安全线 616 px 以下禁区）

    for sid, _, _ in SLOTS:
        logger.info("烧录 %s …", sid)
        last_frame: np.ndarray | None = None
        for c in clips_map[sid]:
            layers: list[Image.Image] = []
            if c.badge:
                layers.append(tag_layer("MuJoCo simulation", f_tag))
            if c.speed:
                layers.append(speed_layer(c.speed, f_tag))
            if c.bar:
                layers.append(bar_layer(c.bar, f_bar))

            if c.path is None:  # 结尾卡
                for k in range(c.frames):
                    alpha = min(1.0, k / 20.0)
                    writer.append_data(card_frame(alpha))
                    t += 1 / FPS
                continue

            reader = imageio.get_reader(c.path)
            for k, frame in enumerate(reader):
                img = Image.fromarray(frame.copy())
                for lay in layers:
                    img.paste(lay, (0, 0), lay)
                if c.title:  # 标题 0.9 s 渐入、5.4 s 后渐出
                    a_in = min(1.0, (t - 0.5) / 0.9)
                    a_out = min(1.0, max(0.0, (5.9 - t) / 0.6))
                    a = min(a_in, a_out)
                    if a > 0.01:
                        a_ = a  # B023：绑定当前帧的透明度再进 lambda
                        faded = title_ov.copy()
                        faded.putalpha(faded.getchannel("A").point(
                            lambda v, av=a_: int(v * av)))
                        img.paste(faded, (0, 0), faded)
                draw_subtitle(img, cues, t, f_sub, sub_y)
                writer.append_data(np.asarray(img))
                last_frame = frame.copy()
                t += 1 / FPS
            reader.close()

        # 槽位尾部不足 → 保持末帧（S9 缺口已由结尾卡补齐）
        for _ in range(0 if sid == "S9" else pad[sid]):
            img = Image.fromarray(last_frame)
            draw_subtitle(img, cues, t, f_sub, sub_y)
            writer.append_data(np.asarray(img))
            t += 1 / FPS
    writer.close()
    logger.info("[done] 烧录视频 %.1f s", t)

    build_audio(timings)
    mux(tmp_video, crf, maxrate)
    report(extract)


def speed_layer(text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    """右上倍速标签（tag_layer 的右对齐版）。"""
    lay = tag_layer(text, font)
    bb = lay.getbbox()
    shifted = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shifted.paste(lay.crop(bb), (W - 22 - (bb[2] - bb[0]), 18))
    return shifted


def draw_subtitle(img: Image.Image, cues: list[Cue], t: float,
                  font: ImageFont.FreeTypeFont, y0: int) -> None:
    """在帧上画当前 cue（白字黑描边，单行居中；超宽自动缩字号）。"""
    cue = next((c for c in cues if c.t0 <= t < c.t1), None)
    if cue is None:
        return
    text = " ".join(cue.text.split())  # 单行化（兼容旧两行 SRT）
    d = ImageDraw.Draw(img)
    f = font
    while d.textlength(text, font=f) > SUB_MAX_W and f.size > SUB_MIN_SIZE:
        f = ImageFont.truetype(f.path, f.size - 1)
    bb = d.textbbox((0, 0), text, font=f)
    d.text(((W - (bb[2] - bb[0])) // 2, y0), text, font=f,
           fill=(255, 255, 255, 245), stroke_width=3,
           stroke_fill=(0, 0, 0, 210))


def build_audio(timings: dict) -> None:
    """九段 mp3 → adelay 定位 → amix → narration.wav（44.1k 立体声）。"""
    ff = ffmpeg_exe()
    segs = timings["segments"]
    args = [ff]
    filt = []
    for i, s in enumerate(segs):
        args += ["-i", str(ASSETS / s["mp3"])]
        ms = round((s["t0"] + HEAD_GAP) * 1000)
        filt.append(f"[{i}:a]adelay={ms}|{ms}[a{i}]")
    mix = "".join(f"[a{i}]" for i in range(len(segs)))
    filt.append(f"{mix}amix=inputs={len(segs)}:normalize=0,"
                "aformat=sample_rates=44100:channel_layouts=stereo[aout]")
    args += ["-filter_complex", ";".join(filt), "-map", "[aout]",
             "-c:a", "pcm_s16le", str(ASSETS / "_narration.wav"), "-y"]
    r = subprocess.run(args, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"音轨合成失败: {r.stderr[-800:]}")
    logger.info("[done] 音轨 %.1f s",
                probe_duration(ASSETS / "_narration.wav"))


def mux(tmp_video: Path, crf: int, maxrate: int) -> None:
    """烧录视频 + 音轨 → 成片（AAC 64k，清元数据，faststart）。"""
    ff = ffmpeg_exe()
    args = [ff, "-i", str(tmp_video), "-i", str(ASSETS / "_narration.wav"),
            "-c:v", "copy", "-c:a", "aac", "-b:a", "64k",
            "-map_metadata", "-1", "-movflags", "+faststart",
            str(OUT), "-y"]
    r = subprocess.run(args, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"封装失败: {r.stderr[-800:]}")
    logger.info("[done] %s", OUT.name)


def report(extract: str) -> None:
    """打印成片规格 + 可选抽帧。"""
    dur = probe_duration(OUT)
    size = OUT.stat().st_size
    print(f"\n[成片] {OUT}")
    print(f"  时长 {dur:.1f} s | 大小 {size / 1e6:.2f} MB "
          f"(预算 18.5) | 平均码率 {size * 8 / dur / 1e3:.0f} kbps")
    if size > 18.5e6:
        print("  [WARN] 超 18.5 MB —— 建议 --crf 20 或 --maxrate 850 重跑")
    if extract:
        reader = imageio.get_reader(OUT)
        for ts in extract.split(","):
            fr = reader.get_data(min(int(float(ts) * FPS),
                                     int(dur * FPS) - 1))
            p = ASSETS / f"qc_t{float(ts):07.2f}.png"
            Image.fromarray(fr).save(p)
            print(f"  QC 帧 → {p}")
        reader.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--maxrate", type=int, default=950)
    ap.add_argument("--extract", default="",
                    help="逗号分隔的抽帧时刻（秒），如 5,30,60")
    args = ap.parse_args()
    build(args.crf, args.maxrate, args.extract)


if __name__ == "__main__":
    main()
