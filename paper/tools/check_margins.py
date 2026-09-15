#!/usr/bin/env python3
"""PaperCept 页边距合规自检（ICRA/IROS 投稿前，chat19 新增）。

规格来源：https://ras.papercept.net/conferences/support/page.php （Letter 纸）
  - 首页顶部 ≥ 72 pt（1.0 in）
  - 其余各页顶部 ≥ 54 pt（0.75 in）
  - 所有页面左/右/下边距 ≥ 54 pt（0.75 in）
  - 正文宽 ≤ 504 pt（7.0 in，由左右各 54 pt 推出）

判定口径：默认「光栅墨迹」——每页渲染成灰度位图，取非白像素的最小外接矩形，
再折算四边边距。为什么不用矢量提取：pdftex 以 Form XObject 嵌入图片（/BBox 裁剪），
而 PyMuPDF 的 get_drawings()/get_text() 不裁剪 form 的 /BBox，会把图片内部本被裁掉、
实际不落墨的几何算进包围盒（实测 Fig.1 的全幅白底矩形会假报右边界越界 ~38.8 pt）。
矢量口径可用 --vector 查看，仅作对照参考。

用法（解释器需带 PyMuPDF，例如 conda 环境 mujoco_tennis）：
  python tools/check_margins.py                          # 默认检查 build/main.pdf，8 页上限
  python tools/check_margins.py main.pdf --max-pages 8
  python tools/check_margins.py --vector                 # 矢量口径对照（含伪影说明）
  python tools/check_margins.py --thresh 200             # 更严格的墨迹阈值（默认 250）

退出码：0 = 全部合规；1 = 有页面不合规 / 页数超限 / 页尺寸不对。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np

# PaperCept 规格（pt）
REQ_FIRST_PAGE_TOP = 72.0  # 首页顶部
REQ_OTHER_TOP = 54.0  # 其余页顶部
REQ_SIDE_BOTTOM = 54.0  # 左右与底部
EXPECTED_SIZE = (612.0, 792.0)  # Letter


def _raster_margins(page: fitz.Page, dpi: int, thresh: int) -> dict[str, float]:
    """按光栅墨迹计算单页四边边距（pt）。

    取边界像素的「外沿」折算，即对墨迹范围作最不利（最保守）估计，
    单像素不确定性 ≤ 72/dpi pt。
    """
    rect = page.rect
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    ink = arr < thresh
    rows = np.flatnonzero(ink.any(axis=1))
    cols = np.flatnonzero(ink.any(axis=0))
    if rows.size == 0 or cols.size == 0:  # 整页空白（异常）
        return {"top": float("nan"), "bottom": float("nan"), "left": float("nan"), "right": float("nan")}
    sx = rect.width / pix.width
    sy = rect.height / pix.height
    return {
        "top": float(rows[0]) * sy,
        "bottom": float(pix.height - (rows[-1] + 1)) * sy,
        "left": float(cols[0]) * sx,
        "right": float(pix.width - (cols[-1] + 1)) * sx,
    }


def _vector_margins(page: fitz.Page) -> dict[str, float]:
    """按矢量几何（文本块 + 绘制 + 图像）计算单页四边边距（pt，对照口径）。

    注意：该口径不裁剪 Form XObject 的 /BBox，图片内部的隐藏几何会被计入，
    结果可能比真实墨迹更「越界」，仅用于排查。
    """
    rect = page.rect
    x0s: list[float] = []
    y0s: list[float] = []
    x1s: list[float] = []
    y1s: list[float] = []
    for b in page.get_text("blocks"):
        x0s.append(float(b[0])); y0s.append(float(b[1])); x1s.append(float(b[2])); y1s.append(float(b[3]))
    for dr in page.get_drawings():
        r = dr["rect"]
        x0s.append(float(r.x0)); y0s.append(float(r.y0)); x1s.append(float(r.x1)); y1s.append(float(r.y1))
    for im in page.get_images(full=True):
        try:
            r = page.get_image_rects(im[0])[0]
        except Exception:
            continue
        x0s.append(float(r.x0)); y0s.append(float(r.y0)); x1s.append(float(r.x1)); y1s.append(float(r.y1))
    if not x0s:
        return {"top": float("nan"), "bottom": float("nan"), "left": float("nan"), "right": float("nan")}
    return {
        "top": min(y0s),
        "bottom": rect.height - max(y1s),
        "left": min(x0s),
        "right": rect.width - max(x1s),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="PaperCept 页边距合规自检（默认光栅墨迹口径）")
    ap.add_argument("pdf", nargs="?", default="build/main.pdf", help="待检查 PDF（默认 build/main.pdf）")
    ap.add_argument("--max-pages", type=int, default=8, help="页数上限（默认 8）")
    ap.add_argument("--dpi", type=int, default=600, help="光栅化 DPI（默认 600，约 0.12 pt/像素）")
    ap.add_argument("--thresh", type=int, default=250, help="墨迹阈值：灰度 < 该值算落墨（默认 250，最严）")
    ap.add_argument("--vector", action="store_true", help="改用矢量几何口径（对照用，含 form 伪影）")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"[FAIL] 找不到文件：{pdf_path}")
        return 1

    doc = fitz.open(pdf_path)
    mode = "矢量几何（对照口径，含 form /BBox 伪影）" if args.vector else f"光栅墨迹 @{args.dpi}dpi，阈值<{args.thresh}"
    print(f"== PaperCept 页边距检查 ==\n文件: {pdf_path}\n口径: {mode}\n")
    print(f"{'页':>2} {'上':>8} {'下':>8} {'左':>8} {'右':>8}   要求(上/下/左/右)      判定")

    ok = True
    worst = {"top_first": float("inf"), "top_other": float("inf"), "bottom": float("inf"), "left": float("inf"), "right": float("inf")}

    for i, page in enumerate(doc, start=1):
        if (round(page.rect.width, 1), round(page.rect.height, 1)) != EXPECTED_SIZE:
            print(f"  [FAIL] 第 {i} 页尺寸 {page.rect.width:.1f}x{page.rect.height:.1f} != Letter 612x792")
            ok = False
        m = _vector_margins(page) if args.vector else _raster_margins(page, args.dpi, args.thresh)
        req_top = REQ_FIRST_PAGE_TOP if i == 1 else REQ_OTHER_TOP
        need = {"top": req_top, "bottom": REQ_SIDE_BOTTOM, "left": REQ_SIDE_BOTTOM, "right": REQ_SIDE_BOTTOM}
        bad = [k for k in need if not (m[k] >= need[k])]
        status = "OK" if not bad else "FAIL " + ",".join(bad)
        if bad:
            ok = False
        print(
            f"{i:>2} {m['top']:>8.2f} {m['bottom']:>8.2f} {m['left']:>8.2f} {m['right']:>8.2f}"
            f"   {req_top:.0f}/{REQ_SIDE_BOTTOM:.0f}/{REQ_SIDE_BOTTOM:.0f}/{REQ_SIDE_BOTTOM:.0f}      {status}"
        )
        worst["top_first" if i == 1 else "top_other"] = min(worst["top_first" if i == 1 else "top_other"], m["top"])
        for k in ("bottom", "left", "right"):
            worst[k] = min(worst[k], m[k])

    n = doc.page_count
    if n > args.max_pages:
        print(f"\n[FAIL] 页数 {n} 超过上限 {args.max_pages}")
        ok = False
    if worst["top_first"] != float("inf"):
        print(
            f"\n最小边距：首页顶 {worst['top_first']:.2f} / 其余页顶 {worst['top_other']:.2f} / "
            f"下 {worst['bottom']:.2f} / 左 {worst['left']:.2f} / 右 {worst['right']:.2f}  (pt)"
        )
    print(
        f"MARGIN_SUMMARY: pages={n} top1={worst['top_first']:.2f} top_other_min={worst['top_other']:.2f} "
        f"bottom_min={worst['bottom']:.2f} left_min={worst['left']:.2f} right_min={worst['right']:.2f} "
        f"verdict={'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
