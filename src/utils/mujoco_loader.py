"""MuJoCo 模型安全加载工具。

MuJoCo 底层 C 函数 mj_loadXML 在 Windows 上不支持路径中包含非 ASCII 字符（如中文）。
本模块提供 load_mujoco_model() 函数，自动检测并处理该问题：
- Windows + 非 ASCII 路径 → 复制到临时 ASCII 目录后加载
- 其他情况 → 直接加载

同时处理 XML 文件中引用的相对路径资源（mesh、texture 等）。

Task 2.4 扩展：
- 缓存键 = (resolved 项目根, 受控资源集合内容摘要)。摘要覆盖
  src/robot 全部嵌套 XML、RM65 网格、assets/unitree_d2 模型资源，
  按文件相对路径 + 内容 sha256 判等，不依赖文件名或秒级 mtime。
- ``xml_text`` 关键字参数支持参数化 XML 加载：在与模板同目录的唯一
  临时文件中编译（保持 include/mesh 相对路径基准），编译完成或失败
  均清理该临时 XML，不修改磁盘模板，也不缓存 MjModel 实例。
"""

import atexit
import hashlib
import logging
import os
import shutil
import stat
import sys
import tempfile
import time
from pathlib import Path

import mujoco

logger = logging.getLogger(__name__)

# 模块级缓存：同一进程只复制一次
# 缓存键 = (resolved 项目根, 资源内容摘要)，任一变化即重新复制
_cached_tmp_dir: Path | None = None
_cached_src_hash: str | None = None
_cached_project_root: Path | None = None

# 崩溃残留目录的判定年龄（秒）：超过此值的 mj_model_* 目录视为上次崩溃残留
_STALE_AGE_S: int = 3600

# 参与内容摘要的资源集合：(相对项目根的目录, 递归匹配模式)
# - src/robot 全部 XML（含嵌套，覆盖 RM65 模型与 D2 片段 d2_arm/d2_assets）
# - RM65 模型实际引用的网格目录
# - Unitree D2 官方模型资源（Task 2.1 并行创建，目录可能尚不存在，允许为空）
# - Unitree R1D 官方模型资源（Task 2.5a 接入）
# 注意：不得把整个 assets/ 纳入摘要（内含大型视觉算法仓库）
_SIGNATURE_RESOURCE_DIRS: tuple[tuple[str, str], ...] = (
    ("src/robot", "*.xml"),
    ("assets/rm_65/urdf/meshes", "*"),
    ("assets/unitree_d2", "*"),
    ("assets/unitree_r1d", "*"),
)


def _is_ascii_path(path: Path) -> bool:
    """检查路径是否仅包含 ASCII 字符。"""
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _compute_dir_signature(project_root: Path) -> str:
    """计算受控模型资源集合的内容摘要（缓存键组成部分）。

    摘要对集合内每个文件的 (相对项目根路径, 内容 sha256) 依序累计哈希：
    文件名相同但内容变化、或元数据（mtime）被还原的修改都会改变摘要，
    杜绝仅按文件名或秒级 mtime 判等导致的旧副本命中。目录不存在时
    （如 assets/unitree_d2 尚未创建）按空集合处理，不报错。

    Args:
        project_root: 项目根目录。

    Returns:
        十六进制摘要字符串（截断 16 位）。
    """
    root = project_root.resolve()
    h = hashlib.sha256()
    for rel_dir, pattern in _SIGNATURE_RESOURCE_DIRS:
        base = root / rel_dir
        if not base.is_dir():
            continue
        for f in sorted(base.rglob(pattern), key=lambda p: p.as_posix()):
            if not f.is_file():
                continue
            h.update(f.relative_to(root).as_posix().encode("utf-8"))
            h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()[:16]


def _setup_temp_copy(project_root: Path) -> Path:
    """将 src/robot/ 和 assets/ 复制到系统临时目录下的唯一子目录。

    缓存键 = (resolved project_root, 资源内容摘要)：项目根不同或任一受控
    资源内容变化都会重新复制；键失效时顺带清理被替换的旧副本目录。
    每进程生成独立目录（PID 前缀 + mkdtemp 随机串）避免并发进程竞态；
    启动时清理超过 _STALE_AGE_S 的旧崩溃残留目录；atexit 正常退出时
    清理本进程目录。
    """
    global _cached_tmp_dir, _cached_src_hash, _cached_project_root

    resolved_root = project_root.resolve()
    sig = _compute_dir_signature(project_root)
    if (
        _cached_tmp_dir is not None
        and _cached_project_root == resolved_root
        and _cached_src_hash == sig
    ):
        if _cached_tmp_dir.exists():
            logger.debug("复用缓存的临时模型目录: %s", _cached_tmp_dir)
            return _cached_tmp_dir
        logger.warning("缓存临时目录已被删除: %s，重新复制", _cached_tmp_dir)
        _cached_tmp_dir = None

    if _cached_tmp_dir is not None:
        # 缓存键已失效：立即清理被替换的旧副本目录，减少残留
        logger.info("缓存键失效，清理旧临时目录: %s", _cached_tmp_dir)
        shutil.rmtree(_cached_tmp_dir, ignore_errors=True)
        _cached_tmp_dir = None

    # 清理崩溃残留：超过 _STALE_AGE_S 的同类 mj_model_* 目录
    tmp_root = Path(tempfile.gettempdir())
    stale_threshold = time.time() - _STALE_AGE_S
    for old in tmp_root.glob("mj_model_*"):
        try:
            if old.is_dir() and old.stat().st_mtime < stale_threshold:
                shutil.rmtree(old, ignore_errors=True)
        except OSError:
            pass

    # 每进程唯一目录：mkdtemp 自带随机串，PID 前缀便于人工识别/清理
    tmp = Path(tempfile.mkdtemp(prefix=f"mj_model_{os.getpid()}_"))
    logger.info("复制模型文件到临时目录: %s", tmp)

    robot_dst = tmp / "src" / "robot"
    robot_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(project_root / "src" / "robot", robot_dst)

    assets_src = project_root / "assets"
    if assets_src.exists():
        shutil.copytree(assets_src, tmp / "assets")

    # copytree 对每个复制目录执行 copystat：Unix 只读安装/容器挂载的
    # 0555 源目录会把权限位原样带进副本，导致 xml_text 降级路径在副本
    # 目录内创建临时 XML 再次失败、rmtree 也无法清理只读目录。仅对
    # 副本目录补足属主写位（不动源目录，不改文件权限）。
    _ensure_dirs_writable(tmp)

    _cached_tmp_dir = tmp
    _cached_src_hash = sig
    _cached_project_root = resolved_root
    return tmp


def _ensure_dirs_writable(root: Path) -> None:
    """给临时副本的全部目录补足属主写权限（只动副本，不动源目录）。

    shutil.copytree 递归复制时对每个目录执行 copystat：Unix 下 0555
    （r-x）的只读源目录会把权限位原样带进副本，于是 (1) xml_text 的
    只读目录降级路径在副本目录内 mkstemp 再次 PermissionError——
    恰是该降级想要服务的只读部署场景；(2) rmtree 清理副本与崩溃残留
    清扫都会在只读目录上失败，留下永久残留。此处仅对副本目录 OR 上
    属主写位；文件权限不动（副本中的只读文件不影响编译加载）。

    Args:
        root: 临时副本根目录。
    """
    for dirpath, _dirnames, _filenames in os.walk(root):
        path = Path(dirpath)
        try:
            mode = stat.S_IMODE(path.stat().st_mode)
            if not mode & stat.S_IWUSR:
                os.chmod(path, mode | stat.S_IWUSR)
        except OSError as exc:  # pragma: no cover - 副本被外部并发改动
            logger.warning("补足临时副本目录写权限失败: %s (%s)", path, exc)


def _cleanup_temp() -> None:
    """进程退出时清理临时目录。"""
    global _cached_tmp_dir, _cached_src_hash, _cached_project_root
    if _cached_tmp_dir is not None:
        shutil.rmtree(_cached_tmp_dir, ignore_errors=True)
        _cached_tmp_dir = None
    _cached_src_hash = None
    _cached_project_root = None


# 模块导入时注册一次（而非每次 _setup_temp_copy 注册），避免长生命周期进程
# 中签名变化导致多次注册。_cleanup_temp 在 _cached_tmp_dir=None 时安全 no-op。
atexit.register(_cleanup_temp)


def load_mujoco_model(model_path: Path, *, xml_text: str | None = None) -> mujoco.MjModel:
    """安全加载 MuJoCo 模型，自动处理 Windows 非 ASCII 路径问题。

    当在 Windows 上且路径包含非 ASCII 字符时，将 src/robot/ 和 assets/
    复制到临时 ASCII 目录后加载模型。其他情况直接使用原始路径。

    Args:
        model_path: MuJoCo XML 模板文件路径。``xml_text`` 非 None 时该
            文件本身不被读取内容，但其所在目录作为 include/mesh 等相对
            资源的解析基准（非 ASCII 分支下即临时副本中的对应目录）。
        xml_text: 参数化 XML 文本，仅用于本次实例化，不写回磁盘模板。
            非 None 时在与模板同目录的唯一临时 XML 文件中编译，编译
            完成或失败均清理该临时文件；若该目录不可写（只读安装、
            只读容器挂载），自动降级到 ASCII 临时副本中的对应目录
            编译，因此本 API 不要求源码目录可写。不做 MjModel 实例
            缓存，每次调用返回独立模型。必须以关键字参数传递（兼容
            既有位置传参）。

    Returns:
        加载完成的 MjModel 对象（每次调用均为新实例）。
    """
    path = Path(model_path).resolve()

    # 非 Windows 或纯 ASCII 路径：直接加载
    if sys.platform != "win32" or _is_ascii_path(path):
        base_xml = path
    else:
        # Windows + 非 ASCII 路径：需要复制到临时目录
        # 推断项目根目录（model_path 通常为 .../mujoco_sim/src/robot/xxx.xml）
        # 支持也放在 scripts/ 下直接调用的场景
        project_root = _infer_project_root(path)
        tmp = _setup_temp_copy(project_root)

        # 根据原始路径中的相对位置计算临时路径
        try:
            rel = path.relative_to(project_root)
            base_xml = tmp / rel
        except ValueError:
            # path 不在 project_root 下（不太可能），退回 src/robot/<文件名>
            base_xml = tmp / "src" / "robot" / path.name

        if not base_xml.exists():
            raise FileNotFoundError(
                f"临时目录中未找到模型文件: {base_xml} "
                f"(原始路径: {model_path})"
            )

        logger.info("从临时路径加载模型: %s (原路径: %s)", base_xml, model_path)

    if xml_text is None:
        return mujoco.MjModel.from_xml_path(str(base_xml))
    return _compile_xml_text_in_place(base_xml, xml_text)


def _compile_xml_text_in_place(
    base_xml: Path, xml_text: str, *, allow_temp_copy_fallback: bool = True
) -> mujoco.MjModel:
    """在与 base_xml 同目录的唯一临时文件中编译参数化 XML 文本。

    同目录放置保证 include/mesh/texture 等相对资源路径与原模板的解析
    基准一致（非 ASCII 分支下该目录即临时 ASCII 副本中的对应目录）。
    编译成功或失败都在 finally 中删除临时 XML；不覆盖 base_xml 指向的
    源模板或缓存模板，也不缓存编译出的 MjModel 实例。

    源目录不可写（只读安装、只读容器挂载）时，``mkstemp`` 会失败；
    此时降级到「ASCII 临时副本」中重试（``allow_temp_copy_fallback``），
    使只读部署仍可加载，而不是把"读模型"变成"必须能写源码目录"。

    Args:
        base_xml: 模板 XML 路径（仅取其所在目录作为资源解析基准）。
        xml_text: 待编译的完整 MJCF 文本。
        allow_temp_copy_fallback: 源目录不可写时是否允许降级到 ASCII
            临时副本；递归调用自身时置 False，避免无限降级。

    Returns:
        新编译的 MjModel 对象（每次调用独立）。
    """
    try:
        fd, name = tempfile.mkstemp(
            prefix=".mj_text_", suffix=".xml", dir=str(base_xml.parent)
        )
    except OSError as exc:
        if not allow_temp_copy_fallback:
            raise
        return _compile_xml_text_via_temp_copy(base_xml, xml_text, exc)
    tmp_xml = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(xml_text)
        logger.debug("从参数化临时 XML 编译模型: %s (基准模板: %s)", tmp_xml, base_xml)
        return mujoco.MjModel.from_xml_path(str(tmp_xml))
    finally:
        try:
            tmp_xml.unlink()
        except OSError as exc:  # pragma: no cover - Windows 极端文件占用
            logger.warning("清理参数化临时 XML 失败: %s (%s)", tmp_xml, exc)


def _compile_xml_text_via_temp_copy(
    base_xml: Path, xml_text: str, cause: OSError
) -> mujoco.MjModel:
    """源模板目录不可写时，改用 ASCII 临时副本的同目录文件编译。

    副本中的相对目录结构与项目根一致（``_setup_temp_copy`` 复制
    src/robot 与 assets），因此 include/mesh 的相对基准仍成立。

    Args:
        base_xml: 原模板 XML 路径（源目录不可写）。
        xml_text: 待编译的完整 MJCF 文本。
        cause: 触发降级的原始 OSError（副本内仍失败时原样抛出）。

    Returns:
        新编译的 MjModel 对象。
    """
    cached = _cached_tmp_dir
    if cached is not None:
        try:
            Path(base_xml).resolve().relative_to(Path(cached).resolve())
        except ValueError:
            pass
        else:  # 已在临时副本中仍不可写：无可再降级，抛原始错误
            raise cause

    project_root = _infer_project_root(base_xml)
    tmp = _setup_temp_copy(project_root)
    try:
        rel = Path(base_xml).resolve().relative_to(project_root.resolve())
    except ValueError:
        raise cause
    fallback_base = tmp / rel
    logger.warning(
        "模板目录不可写（%s），改用 ASCII 临时副本编译: %s → %s",
        cause,
        base_xml,
        fallback_base,
    )
    return _compile_xml_text_in_place(
        fallback_base, xml_text, allow_temp_copy_fallback=False
    )


def _infer_project_root(model_path: Path) -> Path:
    """从模型文件路径推断项目根目录。

    假设目录结构为:
        mujoco_sim/src/robot/xxx.xml  → mujoco_sim/
        mujoco_sim/assets/...         → mujoco_sim/

    Args:
        model_path: 模型文件路径。

    Returns:
        推断的项目根目录路径。
    """
    resolved = model_path.resolve()

    # 向上查找包含 src/robot/ 的目录
    for parent in resolved.parents:
        if (parent / "src" / "robot").is_dir() and (parent / "assets").is_dir():
            return parent

    # 回退：假设模型文件在 src/robot/ 下，往上两级
    return model_path.parent.parent.parent
