from .dreamfusion import *


@threestudio.register("efficient-dreamfusion-system")
class EffDreamFusion(DreamFusion):
    @dataclass
    class Config(DreamFusion.Config):
        pass

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()

    def unmask(self, ind, subsampled_tensor, H, W):
        """
        ind: B,s_H,s_W
        subsampled_tensor: B,C,s_H,s_W
        """

        # Create a grid of coordinates for the original image size
        offset = [ind[0, 0] % H, ind[0, 0] // H]
        indices_all = torch.meshgrid(
            torch.arange(W, dtype=torch.float32, device=self.device),
            torch.arange(H, dtype=torch.float32, device=self.device),
            indexing="xy",
        )

        grid = torch.stack(
            [
                (indices_all[0] - offset[0]) * 4 / (3 * W),
                (indices_all[1] - offset[1]) * 4 / (H * 3),
            ],
            dim=-1,
        )
        grid = grid * 2 - 1
        grid = grid.repeat(subsampled_tensor.shape[0], 1, 1, 1)
        # Use grid_sample to upsample the subsampled tensor (B,C,H,W)
        upsampled_tensor = torch.nn.functional.grid_sample(
            subsampled_tensor, grid, mode="bilinear", align_corners=True
        )

        return upsampled_tensor.permute(0, 2, 3, 1)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        ### using mask to create image at original resolution during training
        (B, s_H, s_W, C) = out["comp_rgb"].shape
        comp_rgb = out["comp_rgb"].permute(0, 3, 1, 2)
        mask = batch["efficiency_mask"]
        comp_rgb = self.unmask(mask, comp_rgb, batch["height"], batch["width"])
        # comp_rgb = torch.zeros(B,batch["height"],batch["width"],C,device=self.device).view(B,-1,C)
        # comp_rgb[:,mask.view(-1)] = out["comp_rgb"].view(B,-1,C)
        out.update(
            {
                "comp_rgb": comp_rgb,
            }
        )

        prompt_utils = self.prompt_processor()
        guidance_out = self.guidance(
            out["comp_rgb"], prompt_utils, **batch, rgb_as_latents=False
        )

        loss = 0.0

        for name, value in guidance_out.items():
            if not (type(value) is torch.Tensor and value.numel() > 1):
                self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        if self.C(self.cfg.loss.lambda_orient) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            loss_orient = (
                out["weights"].detach()
                * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
            ).sum() / (out["opacity"] > 0).sum()
            self.log("train/loss_orient", loss_orient)
            loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        # z-variance loss proposed in HiFA: https://hifa-team.github.io/HiFA-site/
        if "z_variance" in out and "lambda_z_variance" in self.cfg.loss:
            loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
            self.log("train/loss_z_variance", loss_z_variance)
            loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}
