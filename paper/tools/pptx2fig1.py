"""Fig.1 生成工具：ICRA.pptx → XML 补丁（文字/字体）→ LibreOffice 转 PDF → 裁内容 bbox。

背景（2026-09-15 chat17）：
  此前 Fig.1 走 PowerPoint 导出的 EMF（`fig1.emf`）→ LibreOffice 转 PDF，存在三个缺陷：
  ① 文本被 PowerPoint 按原字体算好原点拆成多 run，替代字体宽度不同 → 词内断缝（M ISS、H it）；
  ② 全部文本 run 的字体是主题东亚槽位 `+mn-ea`（→ 微软雅黑），本机缺失时回退到
     Noto Sans CJK，拉丁文字形/字距带"中文黑体味"；
  ③ 导出时图像被二次压缩（渲染图 72.9 KB → 48.7 KB），贴图进一步发糊。
  改为**直接从 PPTX 源头生成**：文字/字体/媒体补丁写在下方常量表里（原始 PPTX 字节不动），
  LibreOffice 按 OOXML 原生排版（无 run 原点错位），导出后再把被 LO 重编码的位图
  换回源文件原始字节（见 restore_image_bytes）。

用法（任意目录下执行均可）：
    python paper/tools/pptx2fig1.py                # 默认输入输出见 DEFAULT_* 常量
    python paper/tools/pptx2fig1.py --keep-temp    # 保留中间产物（排查用）

依赖：LibreOffice（soffice）、PyMuPDF、Pillow、NumPy。
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

# ---------------------------------------------------------------------------
# 常量配置
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]          # …/Tennis_robot_mujoco_sim
DEFAULT_PPTX = REPO_ROOT / "paper" / "figures" / "ICRA.pptx"
DEFAULT_OUT = REPO_ROOT / "paper" / "figures" / "fig1_handdrawn.pdf"

# 字体：英文论文常用拉丁字体（Arial 度量克隆，本机已装；粗体经 <a:b/> 得到 Bold 字重）
FONT_FACE = "Liberation Sans"

# 文字补丁：按顺序应用于 <a:t> 元素内部（只改可见文本，不碰 XML 结构）
TEXT_PATCHES: tuple[tuple[str, str, str], ...] = (
    ("canditates", "candidates", "拼写修正（源图/PowerPoint 均标了拼写错误 err=1）"),
    ("（", "(", "全角左括号→ASCII：全角字形 Liberation Sans 缺失，会触发中日韩字体回退"),
    ("）", ")", "全角右括号→ASCII：同上"),
)

# 媒体补丁：LibreOffice 对 PPTX 媒体的两个兼容性坑（实测缺失元素后逐一修复）
#   ① ppt/media/image3.webp（球精灵，本图用了 17 次）：LO 不导入 webp → 全部丢失。
#      处理：用 Pillow 转成 PNG（无损），并同步改写 rels 的 Target 与 [Content_Types].xml。
#   ② ppt/media/image4.svg（绿勾）：该 SVG 只有 viewBox、缺 width/height，
#      LO 无法确定尺寸而整块丢弃（同目录的 image2.svg 有 width/height，正常）。处理：补属性。
WEBP_TO_PNG: dict[str, str] = {
    "ppt/media/image3.webp": "ppt/media/image3.png",
}
SVG_ADD_SIZE: dict[str, tuple[int, int]] = {
    "ppt/media/image4.svg": (500, 500),          # 与 viewBox="0 0 500 500" 一致
}

# 需要打字体补丁的 XML 部件（幻灯片/母版/版式/主题）
XML_PARTS = ("ppt/slides/", "ppt/slideMasters/", "ppt/slideLayouts/", "ppt/theme/")

# 图像：PDF 导出不降采样、无损压缩（避免贴图二次损伤）
PDF_FILTER = (
    "pdf:impress_pdf_Export:"
    '{"UseLosslessCompression":{"type":"boolean","value":"true"},'
    '"ReduceImageResolution":{"type":"boolean","value":"false"},'
    '"Quality":{"type":"long","value":"100"}}'
)

MM = 72.0 / 25.4                       # 1 mm 对应的 PDF pt
REF_ASPECT = 1.5522                    # 旧版（EMF 路线）墨迹宽高比，用于漂移告警


def patch_text_runs(xml: str) -> tuple[str, list[str]]:
    """对 <a:t>…</a:t> 内的可见文本应用 TEXT_PATCHES，返回 (新 XML, 补丁报告)。

    Args:
        xml: slide XML 原文。

    Returns:
        (打补丁后的 XML, 每条补丁的命中次数描述列表)。
    """
    report: list[str] = []

    def _apply(match: re.Match[str]) -> str:
        """单条 <a:t> 的替换回调。"""
        text = match.group(1)
        for old, new, why in TEXT_PATCHES:
            if old in text:
                text = text.replace(old, new)
        return f"<a:t>{text}</a:t>"

    new_xml = re.sub(r"<a:t>(.*?)</a:t>", _apply, xml, flags=re.S)
    for old, new, why in TEXT_PATCHES:
        n = len(re.findall(re.escape(old), xml)) - len(re.findall(re.escape(old), new_xml))
        if n:
            report.append(f"文字：{old!r}→{new!r} ×{n}（{why}）")
    return new_xml, report


def patch_fonts(xml: str) -> tuple[str, int]:
    """把主题字体引用与 Arial/Calibri 统一替换为 FONT_FACE。

    覆盖三类写法：
      - `typeface="+mn-ea"` 等主题槽位引用（本图 68 处，指向东亚字体，是回退根因）
      - `typeface="Arial"`（母版 txStyles）
      - `typeface="Calibri"`（主题 majorFont/minorFont 的拉丁槽位）

    Args:
        xml: 部件 XML 原文。

    Returns:
        (替换后的 XML, 替换次数)。
    """
    xml, n1 = re.subn(r'typeface="\+[a-z]{2}-[a-z]{2}"', f'typeface="{FONT_FACE}"', xml)
    xml, n2 = re.subn(r'typeface="Arial"', f'typeface="{FONT_FACE}"', xml)
    xml, n3 = re.subn(r'typeface="Calibri"', f'typeface="{FONT_FACE}"', xml)
    return xml, n1 + n2 + n3


def _convert_webp_to_png(data: bytes) -> bytes:
    """webp 字节 → PNG 字节（无损，保留 alpha 通道）。"""
    import io

    im = Image.open(io.BytesIO(data))
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _add_svg_size(data: bytes, size: tuple[int, int]) -> bytes:
    """给只有 viewBox 的 SVG 补 width/height（LibreOffice 需要显式尺寸才能导入）。"""
    w, h = size
    svg = data.decode("utf-8")
    return re.sub(r"<svg\b", f'<svg width="{w}" height="{h}"', svg, count=1).encode("utf-8")


def patch_svg_blip(xml: str) -> tuple[str, int]:
    """给「只有 svgBlip 扩展、没有主 r:embed」的 <a:blip> 补上主引用。

    PowerPoint 的 SVG-only 写法：`<a:blip><a:extLst><a:ext uri="{96DAC541-…}">
    <asvg:svgBlip r:embed="rIdN"/></a:ext></a:extLst></a:blip>` —— 主 blip 没有任何
    引用属性，图片地址只在扩展里。LibreOffice 只读主引用，于是整块丢弃（实测：红叉、
    绿勾两个 SVG 图片全部消失）。补上 r:embed 后 LO 正常按矢量导入。

    Args:
        xml: slide XML 原文。

    Returns:
        (补好的 XML, 补引用次数)。
    """
    out: list[str] = []
    pos, n = 0, 0
    for m in re.finditer(r"<a:blip>", xml):
        end = xml.index("</a:blip>", m.end())
        block = xml[m.start():end]
        svg = re.search(r'<asvg:svgBlip[^>]*r:embed="([^"]+)"', block)
        if svg:
            out.append(xml[pos:m.start()])
            out.append(f'<a:blip r:embed="{svg.group(1)}">')
            out.append(xml[m.end():end])
            pos = end
            n += 1
    out.append(xml[pos:])
    return "".join(out), n


def build_patched_pptx(src: Path, dst: Path) -> list[str]:
    """解包 src，打补丁后重打包为 dst（原始文件绝不改动）。

    补丁分四类：文字（TEXT_PATCHES）、字体（FONT_FACE）、媒体格式（WEBP_TO_PNG）、
    媒体尺寸（SVG_ADD_SIZE）。rels 与 [Content_Types].xml 随重命名自动同步。

    Args:
        src: 原始 PPTX 路径。
        dst: 补丁版 PPTX 输出路径。

    Returns:
        补丁报告（逐条目列出命中次数与原因）。
    """
    report: list[str] = []
    font_hits: dict[str, int] = {}
    renamed = {k: v for k, v in WEBP_TO_PNG.items()}

    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            name = item.filename
            data = zin.read(name)

            if name in renamed:
                # webp → PNG（LO 不支持 webp，直接丢弃整个图片元素）
                data = _convert_webp_to_png(data)
                report.append(f"媒体：{name} → {renamed[name]}（webp 不受 LO 支持，转 PNG 无损保留 alpha）")
                item.filename = renamed[name]
                zout.writestr(item, data)
                continue

            if name in SVG_ADD_SIZE:
                data = _add_svg_size(data, SVG_ADD_SIZE[name])
                report.append(f"媒体：{name} 补 width/height={SVG_ADD_SIZE[name]}"
                              "（原文件只有 viewBox，LO 无法定尺寸而整块丢弃）")

            if name.endswith((".xml", ".rels")):
                text = data.decode("utf-8")
                if name.endswith(".rels"):
                    for old, new in renamed.items():
                        text = text.replace(old.split("ppt/")[-1], new.split("ppt/")[-1])
                elif name.startswith(XML_PARTS):
                    if name.startswith("ppt/slides/"):
                        text, text_rep = patch_text_runs(text)
                        report.extend(text_rep)
                        text, n_blip = patch_svg_blip(text)
                        if n_blip:
                            report.append(f"图片：给 {n_blip} 个只有 svgBlip 扩展的 <a:blip> "
                                          "补主 r:embed（LO 只读主引用，否则整图丢弃）")
                    text, n = patch_fonts(text)
                    if n:
                        font_hits[name] = n
                elif name == "[Content_Types].xml" and "Extension=\"png\"" not in text:
                    # 重命名后的 PNG 需要注册扩展名（本图已有 image5.png，通常已存在）
                    text = text.replace("<Default Extension=\"jpeg\"",
                                        "<Default Extension=\"png\" ContentType=\"image/png\"/>"
                                        "<Default Extension=\"jpeg\"", 1)
                    report.append("媒体：[Content_Types].xml 补注册 png 扩展名")
                data = text.encode("utf-8")

            zout.writestr(item, data)

    if font_hits:
        total = sum(font_hits.values())
        detail = "、".join(f"{k.split('/')[-1]}×{v}" for k, v in sorted(font_hits.items()))
        report.append(f"字体：统一替换为 {FONT_FACE} 共 {total} 处（{detail}）")
    return report


def pptx_to_pdf(pptx: Path, outdir: Path) -> Path:
    """调用 LibreOffice 把 PPTX 转成 PDF（无损图像选项）。

    Args:
        pptx: 待转换的 PPTX 路径。
        outdir: 输出目录。

    Returns:
        生成的 PDF 路径。

    Raises:
        RuntimeError: soffice 不可用或转换失败。
    """
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice is None:
        raise RuntimeError("未找到 LibreOffice（soffice），无法转换")
    cmd = [
        soffice, "--headless", "--norestore", "--nolockcheck",
        "--convert-to", PDF_FILTER, "--outdir", str(outdir), str(pptx),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    pdf = outdir / (pptx.stem + ".pdf")
    if not pdf.exists():
        raise RuntimeError(f"LibreOffice 转换失败：{proc.stdout}\n{proc.stderr}")
    return pdf


def _pptx_rasters(pptx: Path) -> dict[tuple[int, int], bytes]:
    """读取 PPTX 内所有位图，返回 {(宽, 高): 原始字节}（SVG 等非位图跳过）。"""
    import io

    out: dict[tuple[int, int], bytes] = {}
    with zipfile.ZipFile(pptx) as z:
        for name in z.namelist():
            if not name.startswith("ppt/media/"):
                continue
            data = z.read(name)
            try:
                with Image.open(io.BytesIO(data)) as im:
                    out.setdefault(im.size, data)
            except Exception:
                continue
    return out


def restore_image_bytes(page: pymupdf.Page, doc: pymupdf.Document,
                        sources: dict[tuple[int, int], bytes]) -> list[str]:
    """把 PDF 里被 LibreOffice 重编码的位图换回源文件原始字节。

    背景：LO 的 PDF 导出会以更低质量重编码 JPEG（本图 1040×693 渲染图 72.9 KB → 48.7 KB，
    实测命令行过滤选项 UseLosslessCompression/ReduceImageResolution/Quality 在本机 LO 版本
    上均不生效）。这里按**像素尺寸完全一致**匹配源图并原位替换（placement 不变）；
    LO 已降采样的小图（如 960→106 px 的球精灵）保持原样——那是按 300 dpi 打印最优降的。

    Args:
        page: PDF 首页。
        doc: PDF 文档（替换会写入其 xref 表）。
        sources: {(宽, 高): 原始字节}，由 _pptx_rasters 提供。

    Returns:
        替换报告（逐图列出字节数变化）。
    """
    notes: list[str] = []
    for info in page.get_image_info(xrefs=True):
        xref = info["xref"]
        raw = doc.extract_image(xref)
        key = (raw["width"], raw["height"])
        src = sources.get(key)
        if src is None or src == raw["image"]:
            continue
        page.replace_image(xref, stream=src)
        notes.append(f"位图：{key[0]}×{key[1]} 换回原始字节（{len(raw['image'])} → {len(src)} B）")
    return notes


def finalize_pdf(raw_pdf: Path, pptx: Path, dst: Path, dpi: int = 150,
                 thresh: int = 245, pad_mm: float = 0.0) -> tuple[pymupdf.Rect, list[str]]:
    """图像还原 + 裁剪到非白墨迹 bbox + 另存（一步完成，避免中间文件）。

    Args:
        raw_pdf: LibreOffice 导出的原始 PDF。
        pptx: 原始 PPTX（提供位图源字节）。
        dst: 输出 PDF。
        dpi: 墨迹扫描分辨率。
        thresh: 任一 RGB 通道低于该值即视为墨迹。
        pad_mm: 四周额外留白（毫米）。

    Returns:
        (裁剪后的内容矩形, 图像替换报告)。
    """
    doc = pymupdf.open(raw_pdf)
    page = doc[0]
    notes = restore_image_bytes(page, doc, _pptx_rasters(pptx))

    pix = page.get_pixmap(dpi=dpi)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    mask = (arr[..., :3] < thresh).any(axis=2)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("未检测到墨迹：整页为空？")
    s = 72.0 / dpi
    pad = pad_mm * MM
    ink = pymupdf.Rect(
        xs.min() * s - pad, ys.min() * s - pad,
        (xs.max() + 1) * s + pad, (ys.max() + 1) * s + pad,
    )
    page.set_cropbox(ink)
    doc.save(dst, garbage=4, deflate=True)
    return ink, notes


def main() -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="ICRA.pptx → Fig.1 矢量 PDF")
    parser.add_argument("--pptx", type=Path, default=DEFAULT_PPTX, help="输入 PPTX")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="输出 PDF")
    parser.add_argument("--keep-temp", action="store_true", help="保留中间产物目录")
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="fig1_pptx_"))
    print(f"[1/4] 工作目录 {tmp}")
    fixed_pptx = tmp / f"{args.pptx.stem}_fixed.pptx"
    report = build_patched_pptx(args.pptx, fixed_pptx)
    for line in report or ["（无补丁命中）"]:
        print(f"      补丁 → {line}")

    print("[2/4] LibreOffice 转换 …")
    raw_pdf = pptx_to_pdf(fixed_pptx, tmp)

    print("[3/4] 图像还原 + 裁剪内容 bbox …")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ink, img_notes = finalize_pdf(raw_pdf, args.pptx, args.out)
    for line in img_notes:
        print(f"      {line}")
    w_mm, h_mm = ink.width / MM, ink.height / MM
    aspect = ink.width / ink.height
    print(f"      输出 {args.out}")
    print(f"      墨迹 {w_mm:.1f}×{h_mm:.1f} mm，宽高比 {aspect:.4f}"
          f"（旧版参考 {REF_ASPECT}）")
    if abs(aspect - REF_ASPECT) / REF_ASPECT > 0.05:
        print("      ⚠ 宽高比偏离参考值 >5%，请检查版面是否变化")

    print("[4/4] 清理" + ("" if args.keep_temp else "（--keep-temp 可保留中间产物）"))
    if not args.keep_temp:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print(f"      中间产物保留于 {tmp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
