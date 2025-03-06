import torch
import torch.nn as nn
import numpy as np

from generative.networks.nets import AutoencoderKL
from . import layers as reg_layers


class AutoencoderKLTemplateRegistration(AutoencoderKL):
    def __init__(
        self,
        template_generator_type: bool,
        metadata_dim: int,
        *args,
        template_generator_final_activation=None,
        diffeomorphic=False,
        int_steps=7,
        int_downsize=2,
        bidir=False,
        template_shape=None,
        interpolation_type="bilinear",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.template_shape = template_shape
        img_size = template_shape[1:]

        if template_generator_type == "unetdecoderrelu":
            self.conditional_template_network = UNetDecoderReLU(
                self.spatial_dims,
                metadata_dim,
                self.template_shape[0],
                "instance",
            )
        elif template_generator_type == "unetdecoderfinal1x1":
            self.conditional_template_network = UNetDecoderFinal1x1(
                self.spatial_dims,
                metadata_dim,
                self.template_shape[0],
                "instance",
                final_activation=template_generator_final_activation,
            )
        elif template_generator_type == "unetdecoderhalfrelu":
            self.conditional_template_network = UNetDecoderHalfReLU(
                self.spatial_dims,
                metadata_dim,
                self.template_shape[0],
                "instance",
            )
        else:
            raise ValueError(f"{template_generator_type} not recognized.")

        self.default_metadata = nn.Parameter(torch.randn(1, metadata_dim))

        self.diffeomorphic = diffeomorphic
        self.interpolation_type = interpolation_type
        if diffeomorphic:
            # configure optional resize layers (downsize)
            if int_steps > 0 and int_downsize > 1:
                self.resize = reg_layers.ResizeTransform(
                    int_downsize, ndims=kwargs["spatial_dims"]
                )
            else:
                self.resize = None

            # resize to full res
            if int_steps > 0 and int_downsize > 1:
                self.fullsize = reg_layers.ResizeTransform(
                    1 / int_downsize, ndims=kwargs["spatial_dims"]
                )
            else:
                self.fullsize = None

            # configure bidirectional training
            self.bidir = bidir

            # configure optional integration layer for diffeomorphic warp
            down_shape = [int(dim / int_downsize) for dim in img_size]
            self.integrate = (
                reg_layers.VecInt(down_shape, int_steps) if int_steps > 0 else None
            )

    def _make_diffeomorphic(self, pos_flow):
        # resize flow for integration
        if self.resize:
            pos_flow = self.resize(pos_flow)

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        return pos_flow

    def displacement_field_to_registered_image(self, displacement_field, img):
        assert (
            img.shape[2:] == displacement_field.shape[2:]
        ), f"Image shape {img.shape} and displacement field shape {displacement_field.shape} must match"
        displacement_field = displacement_field.permute(0, 2, 3, 4, 1)
        flow_field = reg_layers.displacement2pytorchflow(
            displacement_field, input_space="norm"
        )
        return reg_layers.align_img(flow_field, img, mode=self.interpolation_type)

    def get_template_image(self, *args):
        del args
        return self.conditional_template_network(self.default_metadata)

    def forward(
        self, x: torch.Tensor, metadata=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        template = self.get_template_image(metadata)
        z_mu, z_sigma = self.encode(x, template)
        z = self.sampling(z_mu, z_sigma)
        reconstruction, displacement_field = self.decode(z, template)
        return reconstruction, z_mu, z_sigma, z, displacement_field

    def encode(self, x, template):
        """
        Forwards an image through the spatial encoder, obtaining the latent mean and sigma representations.

        Args:
            x: BxCx[SPATIAL DIMS] tensor

        """
        x = torch.cat([x, template], dim=1)
        return super().encode(x)

    def encode_stage_2_inputs(self, x, template):
        z_mu, z_sigma = self.encode(x, template)
        z = self.sampling(z_mu, z_sigma)
        return z

    def decode(self, z, template):
        """
        Based on a latent space sample, forwards it through the Decoder.

        Args:
            z: Bx[Z_CHANNELS]x[LATENT SPACE SHAPE]

        Returns:
            decoded image tensor
        """
        displacement_field = super().decode(z)
        if self.diffeomorphic:
            displacement_field = self._make_diffeomorphic(displacement_field)
        reconstruction = self.displacement_field_to_registered_image(
            displacement_field, template
        )
        return reconstruction, displacement_field

    def decode_stage_2_outputs(self, z, template):
        displacement_field = super().decode(z)
        if self.diffeomorphic:
            displacement_field = self._make_diffeomorphic(displacement_field)
        reconstruction = self.displacement_field_to_registered_image(
            displacement_field, template
        )
        return reconstruction


class AutoencoderKLConditionalTemplateRegistration(AutoencoderKLTemplateRegistration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.default_metadata

    def get_template_image(self, metadata):
        return self.conditional_template_network(metadata)


class UNetDecoderReLU(nn.Module):
    def __init__(self, dim, metadata_dim, out_dim, norm_type, reshape=(20, 24, 22)):
        super().__init__()
        self.h_dims = [32, 64, 64, 128, 128, 256, 256, 512]
        self.dim = dim
        self.reshape = reshape

        # Project metadata into a feature vector of h_dims[6] channels
        self.meta_proj = nn.Linear(metadata_dim, self.h_dims[6] * np.prod(reshape))

        # The decoding pipeline remains the same, starting from h_dims[6] channels at 1x1 resolution
        self.block1 = reg_layers.ConvBlockUp(
            self.h_dims[6], self.h_dims[6], 1, norm_type, False, dim
        )
        self.block2 = reg_layers.ConvBlockUp(
            self.h_dims[6], self.h_dims[5], 1, norm_type, True, dim
        )

        self.block3 = reg_layers.ConvBlockUp(
            self.h_dims[5], self.h_dims[4], 1, norm_type, False, dim
        )
        self.block4 = reg_layers.ConvBlockUp(
            self.h_dims[4], self.h_dims[3], 1, norm_type, True, dim
        )

        self.block5 = reg_layers.ConvBlockUp(
            self.h_dims[3], self.h_dims[2], 1, norm_type, False, dim
        )
        self.block6 = reg_layers.ConvBlockUp(
            self.h_dims[2], self.h_dims[1], 1, norm_type, True, dim
        )

        self.block7 = reg_layers.ConvBlockUp(
            self.h_dims[1], self.h_dims[0], 1, norm_type, False, dim
        )
        self.block9 = reg_layers.ConvBlockUp(
            self.h_dims[0], out_dim, 1, norm_type, False, dim
        )

    def forward(self, metadata):
        # Project the metadata into a feature map
        meta = self.meta_proj(metadata)  # shape: [B, h_dims[6]]
        meta = meta.reshape(
            -1, self.h_dims[6], *self.reshape
        )  # shape: [B, h_dims[6], 20, 22, 21]

        out = self.block1(meta)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.block9(out)

        return out


class UNetDecoderFinal1x1(nn.Module):
    def __init__(
        self,
        dim,
        metadata_dim,
        out_dim,
        norm_type,
        reshape=(20, 24, 22),
        final_activation=None,
    ):
        super().__init__()
        self.h_dims = [32, 64, 64, 128, 128, 256, 256, 512]
        self.dim = dim
        self.reshape = reshape
        if final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()

        # Project metadata into a feature vector of h_dims[6] channels
        self.meta_proj = nn.Linear(metadata_dim, self.h_dims[6] * np.prod(reshape))

        self.block1 = reg_layers.ConvBlockUp(
            self.h_dims[6], self.h_dims[6], 1, norm_type, False, dim
        )
        self.block2 = reg_layers.ConvBlockUp(
            self.h_dims[6], self.h_dims[5], 1, norm_type, True, dim
        )
        self.block3 = reg_layers.ConvBlockUp(
            self.h_dims[5], self.h_dims[4], 1, norm_type, False, dim
        )
        self.block4 = reg_layers.ConvBlockUp(
            self.h_dims[4], self.h_dims[3], 1, norm_type, True, dim
        )
        self.block5 = reg_layers.ConvBlockUp(
            self.h_dims[3], self.h_dims[2], 1, norm_type, False, dim
        )
        self.block6 = reg_layers.ConvBlockUp(
            self.h_dims[2], self.h_dims[1], 1, norm_type, True, dim
        )
        self.block7 = reg_layers.ConvBlockUp(
            self.h_dims[1], self.h_dims[0], 1, norm_type, False, dim
        )
        self.block9 = nn.Conv3d(self.h_dims[0], out_dim, kernel_size=1)

    def forward(self, metadata):
        # Project the metadata into a feature map
        meta = self.meta_proj(metadata)  # shape: [B, h_dims[6]]
        meta = meta.reshape(
            -1, self.h_dims[6], *self.reshape
        )  # shape: [B, h_dims[6], 20, 22, 21]

        out = self.block1(meta)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.block9(out)
        if hasattr(self, "final_activation"):
            out = self.final_activation(out)
        return out


class UNetDecoderHalfReLU(nn.Module):
    def __init__(
        self,
        dim,
        metadata_dim,
        out_dim,
        norm_type,
        reshape=(20, 24, 22),
        final_activation=None,
    ):
        super().__init__()
        self.h_dims = [32, 64, 64, 128, 128, 256, 256, 512]
        self.dim = dim
        self.reshape = reshape
        if final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()

        # Project metadata into a feature vector of h_dims[6] channels
        self.meta_proj = nn.Linear(metadata_dim, self.h_dims[6] * np.prod(reshape))

        self.block1 = reg_layers.ConvBlockUp(
            self.h_dims[6], self.h_dims[6], 1, norm_type, False, dim
        )
        self.block2 = reg_layers.ConvBlockUp(
            self.h_dims[6], self.h_dims[4], 1, norm_type, True, dim
        )
        self.block3 = reg_layers.ConvBlockUp(
            self.h_dims[4], self.h_dims[2], 1, norm_type, True, dim
        )
        self.block4 = reg_layers.ConvBlockUp(
            self.h_dims[2], self.h_dims[0], 1, norm_type, True, dim
        )
        self.block5 = reg_layers.ConvBlockUp(
            self.h_dims[0], out_dim, 1, norm_type, False, dim
        )

    def forward(self, metadata):
        # Project the metadata into a feature map
        meta = self.meta_proj(metadata)  # shape: [B, h_dims[6]]
        meta = meta.reshape(
            -1, self.h_dims[6], *self.reshape
        )  # shape: [B, h_dims[6], 20, 22, 21]

        out = self.block1(meta)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        if hasattr(self, "final_activation"):
            out = self.final_activation(out)
        return out
