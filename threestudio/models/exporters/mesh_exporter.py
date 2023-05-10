from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.exporters.base import Exporter, ExporterOutput
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.mesh import Mesh
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *


@threestudio.register("mesh-exporter")
class MeshExporter(Exporter):
    @dataclass
    class Config(Exporter.Config):
        fmt: str = "obj-mtl"  # in ['obj-mtl', 'obj'], TODO: fbx
        save_normal: bool = False
        save_uv: bool = True
        save_texture: bool = True
        texture_size: int = 1024
        texture_format: str = "jpg"
        xatlas_chart_options: dict = field(default_factory=dict)
        xatlas_pack_options: dict = field(default_factory=dict)
        context_type: str = "gl"

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, self.device)

    def __call__(self) -> List[ExporterOutput]:
        mesh: Mesh = self.geometry.isosurface()

        if self.cfg.fmt == "obj-mtl":
            return self.export_obj_with_mtl(mesh)
        elif self.cfg.fmt == "obj":
            return self.export_obj(mesh)
        else:
            raise ValueError(f"Unsupported mesh export format: {self.cfg.fmt}")

    def export_obj_with_mtl(self, mesh: Mesh) -> List[ExporterOutput]:
        params = {
            "mesh": mesh,
            "save_mat": True,
            "save_normal": self.cfg.save_normal,
            "save_uv": self.cfg.save_uv,
            "save_vertex_color": False,
            "map_Kd": None,
            "map_Ks": None,
            "map_Bump": None,
            "map_format": self.cfg.texture_format,
        }

        if self.cfg.save_uv:
            mesh.unwrap_uv(self.cfg.xatlas_chart_options, self.cfg.xatlas_pack_options)

        if self.cfg.save_texture:
            threestudio.info("Exporting textures ...")
            assert self.cfg.save_uv, "save_uv must be True when save_texture is True"
            # clip space transform
            uv_clip = mesh.v_tex * 2.0 - 1.0
            # pad to four component coordinate
            uv_clip4 = torch.cat(
                (
                    uv_clip,
                    torch.zeros_like(uv_clip[..., 0:1]),
                    torch.ones_like(uv_clip[..., 0:1]),
                ),
                dim=-1,
            )
            # rasterize
            rast, _ = self.ctx.rasterize_one(
                uv_clip4, mesh.t_tex_idx, (self.cfg.texture_size, self.cfg.texture_size)
            )
            valid = rast[:, :, 3] > 0
            # Interpolate world space position

            gb_pos, _ = self.ctx.interpolate_one(
                mesh.v_pos, rast[None, ...], mesh.t_pos_idx
            )
            gb_pos = gb_pos[0]

            # Sample out textures from MLP
            geo_out = self.geometry.export(points=gb_pos)
            mat_out = self.material.export(points=gb_pos, **geo_out)

            if "normal" in geo_out:
                params["map_Bump"] = geo_out["normal"]

            if "albedo" in mat_out:
                params["map_Kd"] = mat_out["albedo"]
            else:
                threestudio.warn(
                    "save_texture is True but no albedo texture found, using default white texture"
                )
            # TODO: map_Ks
        return [ExporterOutput(save_name="model.obj", save_type="obj", params=params)]

    def export_obj(self, mesh: Mesh) -> List[ExporterOutput]:
        params = {
            "mesh": mesh,
            "save_mat": False,
            "save_normal": self.cfg.save_normal,
            "save_uv": self.cfg.save_uv,
            "save_vertex_color": False,
            "map_Kd": None,
            "map_Ks": None,
            "map_Bump": None,
            "map_format": self.cfg.texture_format,
        }

        if self.cfg.save_uv:
            mesh.unwrap_uv(self.cfg.xatlas_chart_options, self.cfg.xatlas_pack_options)

        if self.cfg.save_texture:
            threestudio.info("Exporting textures ...")
            geo_out = self.geometry.export(points=mesh.v_pos)
            mat_out = self.material.export(points=mesh.v_pos, **geo_out)

            if "albedo" in mat_out:
                mesh.set_vertex_color(mat_out["albedo"])
                params["save_vertex_color"] = True
            else:
                threestudio.warn(
                    "save_texture is True but no albedo texture found, not saving vertex color"
                )

        return [ExporterOutput(save_name="model.obj", save_type="obj", params=params)]
