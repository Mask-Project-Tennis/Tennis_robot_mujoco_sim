"""R1D 机器人模型工厂（兼容 facade）。

A2 起模型装配实现按职责拆到 :mod:`src.robot.model_building`：

- ``template.py``：MJCF 模板生成（片段合并、mesh 重写、XML 注释）；
- ``racket.py``：球拍装配（``RacketProfile`` 与现代/旧版球拍 XML）；
- ``scene.py``：场景数值写入、球场白线、限位与装配后校验；
- ``loading.py``：``load_robot_model``、装配管线与依赖清单收集；
- ``types.py``：``RobotModelBundle`` / ``RacketAssembly`` 共享类型。

本模块只做同一对象重导出（``X as X``）：旧导入路径的
``inspect.signature``、``__module__`` 与 ``is`` 语义保持不变。
历史上本模块没有 ``__all__``，迁移后同样不新增。
"""

from __future__ import annotations

import logging

# A1：MJCF 依赖解析已迁至 ``src.robot.model_assets``；此处保留显式重导出，
# 使旧导入路径继续返回同一对象（is 相同，``__module__`` 指向新模块）。
from src.robot.model_assets import (
    DEPENDENCY_ROLE_ARM_FRAGMENT as DEPENDENCY_ROLE_ARM_FRAGMENT,  # noqa: PLC0414
)
from src.robot.model_assets import (
    DEPENDENCY_ROLE_INCLUDE as DEPENDENCY_ROLE_INCLUDE,  # noqa: PLC0414
)
from src.robot.model_assets import (
    DEPENDENCY_ROLE_MESH as DEPENDENCY_ROLE_MESH,  # noqa: PLC0414
)
from src.robot.model_assets import (
    DEPENDENCY_ROLE_MODEL as DEPENDENCY_ROLE_MODEL,  # noqa: PLC0414
)
from src.robot.model_assets import (
    DEPENDENCY_ROLE_TEXTURE as DEPENDENCY_ROLE_TEXTURE,  # noqa: PLC0414
)
from src.robot.model_assets import (
    collect_model_file_dependencies as collect_model_file_dependencies,  # noqa: PLC0414
)
from src.robot.model_assets import (
    mask_xml_comments as mask_xml_comments,  # noqa: PLC0414
)
from src.robot.model_assets.dependencies import (
    _dependency_entry as _dependency_entry,  # noqa: PLC0414
)
from src.robot.model_assets.dependencies import (
    _dependency_relative_path as _dependency_relative_path,  # noqa: PLC0414
)
from src.robot.model_assets.dependencies import (
    _dependency_repo_root as _dependency_repo_root,  # noqa: PLC0414
)
from src.robot.model_assets.dependencies import (
    _sha256_file as _sha256_file,  # noqa: PLC0414
)
from src.robot.model_assets.references import (
    _ASSET_DIR_ATTRS as _ASSET_DIR_ATTRS,  # noqa: PLC0414
)
from src.robot.model_assets.references import (
    _MAX_INCLUDE_DEPTH as _MAX_INCLUDE_DEPTH,  # noqa: PLC0414
)
from src.robot.model_assets.references import (
    _MJCF_TAG_PATTERN as _MJCF_TAG_PATTERN,  # noqa: PLC0414
)
from src.robot.model_assets.references import (
    _XML_COMMENT_PATTERN as _XML_COMMENT_PATTERN,  # noqa: PLC0414
)
from src.robot.model_assets.references import (
    DEPENDENCY_XML_ROLES as DEPENDENCY_XML_ROLES,  # noqa: PLC0414
)
from src.robot.model_assets.references import (
    _asset_candidate_paths as _asset_candidate_paths,  # noqa: PLC0414
)
from src.robot.model_assets.references import (
    _AssetReference as _AssetReference,  # noqa: PLC0414
)
from src.robot.model_assets.references import (
    _attribute_value as _attribute_value,  # noqa: PLC0414
)
from src.robot.model_assets.references import (
    _include_candidate_paths as _include_candidate_paths,  # noqa: PLC0414
)
from src.robot.model_assets.references import (
    _resolve_reference_path as _resolve_reference_path,  # noqa: PLC0414
)
from src.robot.model_assets.references import (
    _walk_model_file as _walk_model_file,  # noqa: PLC0414
)
from src.robot.model_building.loading import (
    _ROBOT_DIR as _ROBOT_DIR,  # noqa: PLC0414
)
from src.robot.model_building.loading import (
    _apply_fixed_joint_poses as _apply_fixed_joint_poses,  # noqa: PLC0414
)
from src.robot.model_building.loading import (
    _arm_fragment_path as _arm_fragment_path,  # noqa: PLC0414
)
from src.robot.model_building.loading import (
    _assemble_bound_model as _assemble_bound_model,  # noqa: PLC0414
)
from src.robot.model_building.loading import (
    collect_model_dependencies as collect_model_dependencies,  # noqa: PLC0414
)
from src.robot.model_building.loading import (
    load_robot_model as load_robot_model,  # noqa: PLC0414
)
from src.robot.model_building.racket import (
    _append_visual_geom as _append_visual_geom,  # noqa: PLC0414
)
from src.robot.model_building.racket import (
    _apply_legacy_racket as _apply_legacy_racket,  # noqa: PLC0414
)
from src.robot.model_building.racket import (
    _apply_racket_profile as _apply_racket_profile,  # noqa: PLC0414
)
from src.robot.model_building.racket import (
    _has_legacy_racket as _has_legacy_racket,  # noqa: PLC0414
)
from src.robot.model_building.racket import (
    _quat_aligning_z as _quat_aligning_z,  # noqa: PLC0414
)
from src.robot.model_building.racket import (
    _racket_assembly as _racket_assembly,  # noqa: PLC0414
)
from src.robot.model_building.racket import (
    _racket_inertial as _racket_inertial,  # noqa: PLC0414
)
from src.robot.model_building.racket import (
    _require_no_legacy_racket as _require_no_legacy_racket,  # noqa: PLC0414
)
from src.robot.model_building.scene import (
    _apply_court_profile as _apply_court_profile,  # noqa: PLC0414
)
from src.robot.model_building.scene import (
    _apply_scene_parameters as _apply_scene_parameters,  # noqa: PLC0414
)
from src.robot.model_building.scene import (
    _check_initial_state as _check_initial_state,  # noqa: PLC0414
)
from src.robot.model_building.scene import (
    _check_scene_landing as _check_scene_landing,  # noqa: PLC0414
)
from src.robot.model_building.scene import (
    _court_profile_of as _court_profile_of,  # noqa: PLC0414
)
from src.robot.model_building.scene import (
    _is_court_line_geom as _is_court_line_geom,  # noqa: PLC0414
)
from src.robot.model_building.scene import (
    _neutralize_court_ground as _neutralize_court_ground,  # noqa: PLC0414
)
from src.robot.model_building.scene import (
    _populate_joint_limits as _populate_joint_limits,  # noqa: PLC0414
)
from src.robot.model_building.scene import (
    _rebase_lift as _rebase_lift,  # noqa: PLC0414
)
from src.robot.model_building.scene import (
    _require_profile as _require_profile,  # noqa: PLC0414
)
from src.robot.model_building.scene import (
    _validate_model_shape as _validate_model_shape,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _ARM_ROOT_BODY as _ARM_ROOT_BODY,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _BASE_LANDING_HEIGHT_M as _BASE_LANDING_HEIGHT_M,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _CASTER_RADIUS as _CASTER_RADIUS,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _CASTER_XY as _CASTER_XY,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _CASTER_Z as _CASTER_Z,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _COURT_GROUND_RGBA as _COURT_GROUND_RGBA,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _COURT_GROUND_SIZE as _COURT_GROUND_SIZE,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _COURT_MATERIAL as _COURT_MATERIAL,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _COURT_MATERIAL_ATTRS as _COURT_MATERIAL_ATTRS,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _COURT_TEXTURE as _COURT_TEXTURE,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _COURT_TEXTURE_ATTRS as _COURT_TEXTURE_ATTRS,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _ENV_BAKED_LIFT_M as _ENV_BAKED_LIFT_M,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _ENV_CHAIN_BODIES as _ENV_CHAIN_BODIES,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _ENV_ROOT_BODY as _ENV_ROOT_BODY,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _ENV_TIP_BODY as _ENV_TIP_BODY,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _EXPORT_MESH_DIRNAME as _EXPORT_MESH_DIRNAME,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _FLANGE_SITE as _FLANGE_SITE,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _GROUND_GEOM as _GROUND_GEOM,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _LIFT_STAGE_BODIES as _LIFT_STAGE_BODIES,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _LIFT_STAGE_RANGE_M as _LIFT_STAGE_RANGE_M,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _LIGHT_ATTRS as _LIGHT_ATTRS,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _MOUNT_BODY as _MOUNT_BODY,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _PASSIVE_ARM_POSE_DEG as _PASSIVE_ARM_POSE_DEG,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _RACKET_BODY as _RACKET_BODY,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _RACKET_CONNECTOR_GEOM as _RACKET_CONNECTOR_GEOM,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _RACKET_CONNECTOR_RGBA as _RACKET_CONNECTOR_RGBA,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _RACKET_FACE_HALF_LENGTH as _RACKET_FACE_HALF_LENGTH,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _RACKET_FACE_HALF_THICKNESS as _RACKET_FACE_HALF_THICKNESS,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _RACKET_FACE_HALF_WIDTH as _RACKET_FACE_HALF_WIDTH,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _RACKET_FACE_VISUAL_GEOM as _RACKET_FACE_VISUAL_GEOM,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _RACKET_FACE_VISUAL_RGBA as _RACKET_FACE_VISUAL_RGBA,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _RACKET_FRAME_GEOM_FORMAT as _RACKET_FRAME_GEOM_FORMAT,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _RACKET_FRAME_GEOMS as _RACKET_FRAME_GEOMS,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _RACKET_FRAME_RADIUS as _RACKET_FRAME_RADIUS,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _RACKET_FRAME_RGBA as _RACKET_FRAME_RGBA,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _RACKET_GEOM as _RACKET_GEOM,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _RACKET_HANDLE_GEOM as _RACKET_HANDLE_GEOM,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _RACKET_HANDLE_LENGTH as _RACKET_HANDLE_LENGTH,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _RACKET_HANDLE_RADIUS as _RACKET_HANDLE_RADIUS,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _RACKET_HANDLE_RGBA as _RACKET_HANDLE_RGBA,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _RACKET_THROAT_GEOM_FORMAT as _RACKET_THROAT_GEOM_FORMAT,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _RACKET_THROAT_GEOMS as _RACKET_THROAT_GEOMS,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _RACKET_THROAT_RADIUS as _RACKET_THROAT_RADIUS,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _RACKET_THROAT_RGBA as _RACKET_THROAT_RGBA,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _TCP_SITE as _TCP_SITE,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _TORSO_BODY as _TORSO_BODY,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _TORSO_GEOM as _TORSO_GEOM,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _WHEEL_HALF_WIDTH as _WHEEL_HALF_WIDTH,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _WHEEL_RADIUS as _WHEEL_RADIUS,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _WHEEL_X as _WHEEL_X,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _WHEEL_Y as _WHEEL_Y,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _WHEEL_Z as _WHEEL_Z,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    DEPENDENCY_MANIFEST_VERSION as DEPENDENCY_MANIFEST_VERSION,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    SIDES as SIDES,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    SUPPORTED_ROBOT_ID as SUPPORTED_ROBOT_ID,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _add_landing_visuals as _add_landing_visuals,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _axis_angle_quat as _axis_angle_quat,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _build_template_element as _build_template_element,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _connected_excludes as _connected_excludes,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _find_body as _find_body,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _find_body_optional as _find_body_optional,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _find_body_with_parent as _find_body_with_parent,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _find_geom as _find_geom,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _flange_position as _flange_position,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _fmt_float as _fmt_float,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _fmt_vec as _fmt_vec,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _fragment_worldbody as _fragment_worldbody,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _model_side as _model_side,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _parse_fragment as _parse_fragment,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _passive_joint_angle as _passive_joint_angle,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _passive_side as _passive_side,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _quat_mul as _quat_mul,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _quat_to_matrix as _quat_to_matrix,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _read_arm_chain as _read_arm_chain,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _require_r1d as _require_r1d,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _require_scene as _require_scene,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _require_side as _require_side,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _retarget_mesh_paths as _retarget_mesh_paths,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _side_chain_bodies as _side_chain_bodies,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _static_arm_copy as _static_arm_copy,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    _template_comment as _template_comment,  # noqa: PLC0414
)
from src.robot.model_building.template import (
    write_r1d_model_template as write_r1d_model_template,  # noqa: PLC0414
)

# P1：共享类型已迁至 ``src.robot.model_building``；此处保留显式重导出，
# 使旧导入路径继续返回同一对象（is 相同）。
from src.robot.model_building.types import (
    RacketAssembly as RacketAssembly,  # noqa: PLC0414
)
from src.robot.model_building.types import (
    RobotModelBundle as RobotModelBundle,  # noqa: PLC0414
)

logger = logging.getLogger(__name__)