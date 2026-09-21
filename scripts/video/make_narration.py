#!/usr/bin/env python3
"""旁白音轨生成器：edge-tts 逐段合成 + WordBoundary 精确字幕时间轴。

设计约束（paper/video/plan.md §3/§6，用户 2026-09-21 拍板）：
- 音色 en-US-ChristopherNeural，语速 rate=-10%（Christopher 实测口播 ~118 wpm，
  全片等效 ~99 wpm，落在 100–115 目标区间附近，从容不赶）。
- 九段旁白文本以本脚本 SEGS 为唯一事实来源，narration.md 每次运行重新生成（防漂移）。
- 每段音频须 ≤ 槽位长度 − 0.3 s（头部余量），违例以退出码 1 报错。
- 字幕用 edge-tts WordBoundary 事件（100 ns tick）对齐到句/子句级 cue；
  无边界事件时退化为按词数等分。
- 产物：assets/narration/S1..S9.mp3 + narration_timings.json + subtitles.srt
  + narration.md（SRT 时间基于「音频起点 = 槽位起点 + HEAD_GAP」约定，
  assemble.py 必须遵守同一约定）。

用法：
    python scripts/video/make_narration.py            # 全部九段
    python scripts/video/make_narration.py --only S5  # 只重合成一段
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import edge_tts

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "paper" / "video" / "assets" / "narration"
VIDEO_DIR = REPO / "paper" / "video"

VOICE = "en-US-ChristopherNeural"
RATE = "-10%"
HEAD_GAP = 0.3  # 音频起点相对槽位起点的头部余量（s），与槽位校验口径一致
TICK = 1e7  # WordBoundary offset 单位 = 100 ns

logger = logging.getLogger("make_narration")


@dataclass
class Seg:
    """一段旁白：槽位时间 + 中英文案。"""

    sid: str
    t0: float          # 槽位起点（s）
    t1: float          # 槽位终点（s）
    zh: str            # 中文文案（供 narration.md 确认用）
    en: str            # 英文台词（TTS 与字幕的实际文本）
    words: int = 0     # 英文词数（运行时统计）
    dur: float = 0.0   # 实测音频时长（s）
    cues: list = field(default_factory=list)  # [(rel_start, rel_end, text), ...]


SEGS: list[Seg] = [
    Seg("S1", 0.0, 12.0,
        "一台双臂机器人挥拍回击一记快速发球——挥拍全程由 MPC 规划。",
        "A dual-arm robot swings to return a fast serve — planned entirely by "
        "model predictive control."),
    Seg("S2", 12.0, 26.0,
        "内部是 iLQR 在滚动优化：终端击球代价加运行代价。"
        "我们只改动其中两项——每一项都可以独立开关。",
        "Inside, iLQR solves a rolling optimization: a terminal hitting cost "
        "plus running costs. We modify exactly two terms, each independently "
        "switchable."),
    Seg("S3", 26.0, 46.0,
        "第一项，终端代价：不再绑定单一预测触球点，而是瞄准一个 ±50 毫秒的候选窗口，"
        "用 softmin 聚合成同一个终端目标。",
        "First, the terminal cost. Instead of one predicted contact point, it "
        "targets a plus-or-minus fifty-millisecond window of candidates, "
        "aggregated by a softmin."),
    Seg("S4", 46.0, 59.0,
        "第二项，一条沿预测轨迹的 tube：横向偏离在 12 厘米半宽内零代价，"
        "出界后铰链代价随偏离增长；沿轨迹的超前/滞后不受罚。",
        "Second, a tube around the predicted trajectory: free inside a "
        "twelve-centimeter half-width, a hinge penalty outside — and "
        "along-track slack is free."),
    Seg("S5", 59.0, 87.0,
        "这是同一记 12 m/s 发球、同一随机种子的并排对比：point-target 漏球，"
        "完整方法回击成功。98 个种子下，仅候选窗口就把 active-hit rate 从 53.1% "
        "提到 85.7%——配对增益 +32.7 个百分点；9 m/s 下是 +17.4。",
        "Here is the same twelve-meter-per-second serve, same seed: "
        "point-target misses; the full method returns it. Across ninety-eight "
        "seeds, the candidate window alone lifts the active-hit rate from "
        "fifty-three point one to eighty-five point seven percent — a paired "
        "gain of plus thirty-two point seven points; plus seventeen point "
        "four at nine meters per second."),
    Seg("S6", 87.0, 102.0,
        "但在共享预测偏差下——9 m/s、20 格扰动网格——差距收窄：82.8% 对 73.9%。",
        "Under a shared prediction bias — a twenty-cell perturbation grid at "
        "nine meters per second — the gap narrows: eighty-two point eight, "
        "versus seventy-three point nine percent."),
    Seg("S7", 102.0, 129.0,
        "但把 TCP 限速收紧到 1.0 m/s（7 m/s 发球、394 个种子），窗口反而有害："
        "仅候选窗口 −11.9 个百分点，完整方法 −12.7；tube 单独仍 +5.8。"
        "在这个速度预算下，放宽目标没有帮助。",
        "But tighten the end-effector speed cap to one meter per second — "
        "seven-meter serves, three hundred ninety-four seeds — and the window "
        "turns harmful: minus eleven point nine points for the window alone, "
        "minus twelve point seven for the full stack. The tube alone still "
        "gains plus five point eight. Under this budget, widening the target "
        "does not help."),
    Seg("S8", 129.0, 150.0,
        "主结果在首个计划期间冻结仿真时间（928 毫秒）；稳态重规划中位数 4 毫秒，"
        "在 150 毫秒预算内。若不冻结，999 毫秒的计划让挥拍无暇完成。",
        "Main results freeze simulated time for the first plan: nine hundred "
        "twenty-eight milliseconds. Steady-state replanning runs at a "
        "four-millisecond median, within the one-hundred-fifty-millisecond "
        "budget. Without freezing, a nine-hundred-ninety-nine-millisecond "
        "plan leaves no time to swing."),
    Seg("S9", 150.0, 165.0,
        "哪种松弛有用，取决于可用的执行速度——代价设计应该跟随速度预算。",
        "Which relaxation helps depends on the available execution speed: "
        "cost design should follow the speed budget."),
]


def mp3_duration(path: Path) -> float:
    """用 imageio-ffmpeg 自带二进制解析 mp3 时长（无 ffprobe 环境）。"""
    import imageio_ffmpeg

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    r = subprocess.run([ff, "-i", str(path)], capture_output=True, text=True,
                       check=False)
    m = re.search(r"Duration: (\d+):(\d+):(\d+)\.(\d+)", r.stderr)
    if not m:
        raise RuntimeError(f"无法解析音频时长: {path}")
    h, mi, s, cs = (int(g) for g in m.groups())
    return h * 3600 + mi * 60 + s + cs / 100.0


async def synth_seg(seg: Seg, retries: int = 3) -> None:
    """合成一段：音频写盘 + 记录 WordBoundary 词级时间戳。"""
    out = OUT_DIR / f"{seg.sid}.mp3"
    for attempt in range(1, retries + 1):
        try:
            com = edge_tts.Communicate(seg.en, VOICE, rate=RATE)
            bounds: list[tuple[float, float]] = []  # [(词起点 s, 词终点 s)]
            audio = bytearray()
            async for chunk in com.stream():
                if chunk["type"] == "audio":
                    audio += chunk["data"]
                elif chunk["type"] == "WordBoundary":
                    start = chunk["offset"] / TICK
                    end = (chunk["offset"] + chunk["duration"]) / TICK
                    bounds.append((start, end))
            out.write_bytes(bytes(audio))
            seg.dur = mp3_duration(out)
            seg.cues = coalesce_sentences(build_cues(seg.en, bounds, seg.dur))
            return
        except Exception:  # noqa: BLE001 —— 网络类错误统一重试
            logger.warning("  %s 第 %d 次合成失败，重试…", seg.sid, attempt)
            await asyncio.sleep(3.0 * attempt)
    raise RuntimeError(f"{seg.sid} 连续 {retries} 次合成失败（edge-tts 网络？）")


def _balanced_split(cl: str, limit: int = 92) -> list[str]:
    """超长无标点子句：在中点最近空格处递归平衡二分（保序，短语边界自然）。"""
    if len(cl) <= limit:
        return [cl]
    mid = len(cl) // 2
    spaces = [i for i, ch in enumerate(cl) if ch == " "]
    if not spaces:
        return [cl]  # 无空格可切（极端），整体放行
    cut = min(spaces, key=lambda i: abs(i - mid))
    return _balanced_split(cl[:cut], limit) + _balanced_split(cl[cut + 1:], limit)


def split_pieces(text: str) -> list[str]:
    """切字幕 cue：句末标点断句 → 长句按 [;:,] 子句**贪心打包** ≤92 字符。

    破折号 " — " 先归一为逗号（切分后承接处自然带标点，不产生粘连句）。
    打包只在同一句内进行（不跨句回并，避免把两句粘进一条 cue）；
    单子句超长时按中点平衡二分（保序）。
    """
    text = text.replace(" — ", ", ")
    parts = [p.strip() for p in re.split(r"(?<=[.?!])\s+", text) if p.strip()]
    pieces: list[str] = []
    for sent in parts:
        if len(sent) <= 92:
            pieces.append(sent)
            continue
        # 子句切分（保留结尾标点）→ 超长子句平衡二分 → 贪心打包
        clauses: list[str] = []
        for c in re.split(r"(?<=[;:,])\s+", sent):
            c = c.strip()
            if c:
                clauses.extend(_balanced_split(c))
        cur = ""
        for cl in clauses:
            if cur and len(cur) + 1 + len(cl) <= 92:
                cur = f"{cur} {cl}"
            else:
                if cur:
                    pieces.append(cur)
                cur = cl
        if cur:
            pieces.append(cur)
    return pieces


def build_cues(text: str, bounds: list[tuple[float, float]],
               dur: float) -> list:
    """词级时间戳 → 句/子句级 cue（相对段起点，秒）。

    Args:
        text: 段英文台词。
        bounds: WordBoundary 词级 (start, end) 列表（与台词词序一致）。
        dur: 实测音频时长（秒；bounds 为空时用于比例兜底）。

    Returns:
        [(rel_start, rel_end, cue_text), ...]。
    """
    pieces = split_pieces(text)
    if not bounds:  # 兜底：无词边界事件时按词数比例等分实测时长
        total = sum(len(_words_of(p)) for p in pieces)
        cues, t = [], 0.0
        for p in pieces:
            d = dur * len(_words_of(p)) / max(total, 1)
            cues.append([t, t + d - 0.05, p])
            t += d
        return [tuple(c) for c in cues]

    cues: list = []
    i = 0
    for p in pieces:
        n = len(_words_of(p))
        wb = bounds[i: i + n]
        i += n
        if not wb:
            continue
        cues.append([wb[0][0], wb[-1][1], p])
    # 结束时间后延 0.3 s 并与下一条起点留 50 ms 缝，末条封顶段时长
    for k in range(len(cues)):
        nxt = cues[k + 1][0] if k + 1 < len(cues) else float("inf")
        cues[k][1] = min(cues[k][1] + 0.3, nxt - 0.05)
        cues[k][1] = max(cues[k][1], cues[k][0] + 0.8)
    return [tuple(c) for c in cues]


def _words_of(piece: str) -> list[str]:
    """子句中的实词（至少含一个字母/数字；破折号等符号不对应边界事件）。"""
    return [t for t in piece.split() if any(ch.isalnum() for ch in t)]


SENT_MAX_CHARS = 112  # 整句 cue 字符上限（112 字符 @17px ≈ 1128 px < 1224 可容纳）


def coalesce_sentences(cues: list) -> list:
    """把同一句内的子句 cue 合并为整句 cue（用户 2026-09-21 反馈：勿拆断句子）。

    规则：向后累积 cue，遇到以 .?! 结尾的 cue 结束一句；仅当整句
    ≤ SENT_MAX_CHARS 时合并（超长句保持切分，单行字幕过小不可读）。
    对已合并的 cue 幂等（单条整句自成一组，原样返回）。
    """
    out: list = []
    grp: list = []
    for c in cues:
        grp.append(c)
        if c[2].rstrip().endswith((".", "?", "!")):
            text = " ".join(x[2] for x in grp)
            if len(text) <= SENT_MAX_CHARS:
                out.append((grp[0][0], grp[-1][1], text))
            else:
                out.extend(grp)
            grp = []
    out.extend(grp)
    return out


def fmt_srt(t: float) -> str:
    """秒 → SRT 时间戳 HH:MM:SS,mmm。"""
    ms = round(t * 1000)
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segs: list[Seg]) -> None:
    """按「音频起点 = 槽位起点 + HEAD_GAP」约定生成全片 SRT。

    单行输出（用户 2026-09-21 粗剪反馈：两行字幕偏大且压动画）；
    超宽 cue 由 assemble.draw_subtitle 运行时自动缩字号兜底。
    """
    lines, idx = [], 1
    for seg in segs:
        base = seg.t0 + HEAD_GAP
        for rs, re_, txt in seg.cues:
            lines.append(f"{idx}\n{fmt_srt(base + rs)} --> {fmt_srt(base + re_)}\n"
                         f"{' '.join(txt.split())}\n")
            idx += 1
    (VIDEO_DIR / "subtitles.srt").write_text("\n".join(lines), encoding="utf-8")
    logger.info("[done] subtitles.srt（%d 条 cue，单行）", idx - 1)


def write_timings(segs: list[Seg]) -> None:
    """写 assemble.py 需要的时间约定与逐段时长（含词数/cue，供 --only 重跑回填）。"""
    import json

    data = {
        "voice": VOICE, "rate": RATE, "head_gap_s": HEAD_GAP,
        "segments": [
            {"sid": s.sid, "t0": s.t0, "t1": s.t1, "words": s.words,
             "dur": round(s.dur, 3), "mp3": f"narration/{s.sid}.mp3",
             "cues": [[round(a, 3), round(b, 3), c] for a, b, c in s.cues]}
            for s in segs
        ],
    }
    (OUT_DIR / "narration_timings.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_cached(segs: list[Seg]) -> None:
    """从上一轮 timings 回填未重合成段的词数/时长/cue（--only 场景防残缺输出）。"""
    import json

    f = OUT_DIR / "narration_timings.json"
    if not f.exists():
        return
    data = json.loads(f.read_text(encoding="utf-8"))
    by_id = {d["sid"]: d for d in data.get("segments", [])}
    for s in segs:
        d = by_id.get(s.sid)
        if d and not s.cues:
            s.words, s.dur = d.get("words", 0), d.get("dur", 0.0)
            s.cues = coalesce_sentences([tuple(c) for c in d.get("cues", [])])
            logger.info("  %s 用缓存时长 %.1f s（未重合成）", s.sid, s.dur)


def write_doc(segs: list[Seg]) -> None:
    """重新生成 narration.md（防脚本与文档漂移）。"""
    total_w = sum(s.words for s in segs)
    spoken_wpm = total_w / max(sum(s.dur for s in segs), 1e-6) * 60
    rows = []
    for s in segs:
        slot = s.t1 - s.t0
        rows.append(
            f"| {s.sid} | {int(s.t0)//60}:{int(s.t0)%60:02d}–{int(s.t1)//60}:"
            f"{int(s.t1)%60:02d} | {s.words} | {s.dur:.1f} / {slot - HEAD_GAP:.1f} |"
            f" {s.zh} |\n| | | | | {s.en} |")
    doc = (
        "# 旁白定稿（九段）\n\n"
        "> 由 `scripts/video/make_narration.py` 生成，勿手改（改文案请改脚本 SEGS）。\n"
        f"> 音色 `{VOICE}` · 语速 `{RATE}`（实测口播 ~{spoken_wpm:.0f} wpm / "
        f"全片等效 ~{total_w / 160 * 60:.0f} wpm）· 共 {total_w} 词 / 160 s。\n"
        "> 音频起点约定 = 槽位起点 + 0.3 s；SRT 同口径。\n\n"
        "| 段 | 时间 | 词数 | 音频/预算 s | 中文（确认稿）→ 英文（TTS/字幕） |\n"
        "|---|---|---|---|---|\n" + "\n".join(rows) + "\n"
    )
    (VIDEO_DIR / "narration.md").write_text(doc, encoding="utf-8")
    logger.info("[done] narration.md")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=[s.sid for s in SEGS], help="只重合成一段")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    segs = [s for s in SEGS if args.only in (None, s.sid)]
    for s in segs:
        s.words = len(_words_of(s.en))
        logger.info("合成 %s（%d 词）…", s.sid, s.words)
        asyncio.run(synth_seg(s))
    load_cached(SEGS)  # 未重合成的段回填缓存，写文件始终用全量九段

    # 槽位校验：音频 ≤ 槽位 − 0.3 s（始终对全量九段校验）
    print(f"\n{'段':<4}{'词数':>4}{'音频 s':>8}{'预算 s':>8}  状态")
    bad = []
    for s in SEGS:
        budget = (s.t1 - s.t0) - HEAD_GAP
        ok = s.dur <= budget
        print(f"{s.sid:<4}{s.words:>4}{s.dur:>8.1f}{budget:>8.1f}  "
              f"{'OK' if ok else '超槽 %.1fs' % (s.dur - budget)}")
        if not ok:
            bad.append(s.sid)
    total_w = sum(s.words for s in SEGS)
    print(f"\n合计 {total_w} 词；全片等效语速 ≈ {total_w / 160 * 60:.0f} wpm")

    write_srt(SEGS)
    write_timings(SEGS)
    write_doc(SEGS)
    if bad:
        print(f"[FAIL] 超槽段: {bad}（需砍词或压语速）")
        sys.exit(1)
    print("[PASS] 全部段落满足槽位预算")


if __name__ == "__main__":
    main()
