import torch
import torch.nn as nn
import torch.nn.functional as nnf


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode="bilinear"):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer("grid", grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, "nsteps should be >= 0, found: %d" % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2**self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = "linear"
        if ndims == 2:
            self.mode = "bi" + self.mode
        elif ndims == 3:
            self.mode = "tri" + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x


def align_img(grid, x, mode="bilinear"):
    return nnf.grid_sample(
        x,
        grid=grid,
        mode=mode,
        padding_mode="border",
        align_corners=False,
    )


def displacement2pytorchflow(displacement_field, input_space="voxel"):
    """Converts displacement field in index coordinates into a flow-field usable by F.grid_sample.
    Assumes original space is in index (voxel) units, 256x256x256.
    Output will be in the [-1, 1] space.

    :param: displacement_field: (N, D, H, W, 3).
    """
    assert displacement_field.shape[-1] == 3, "Displacement field must have 3 channels"
    W, H, D = displacement_field.shape[1:-1]

    # Step 1: Create the original grid for 3D
    coords_x, coords_y, coords_z = torch.meshgrid(
        torch.linspace(-1, 1, W),
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, D),
        indexing="ij",
    )
    coords = torch.stack([coords_z, coords_y, coords_x], dim=-1)  # Shape: (D, H, W, 3)
    coords = coords.unsqueeze(0).to(displacement_field)  # Shape: (N, 3, D, H, W), N=1

    # Step 2: Normalize the displacement field
    # Convert physical displacement values to the [-1, 1] range
    # Assuming the displacement field is given in voxel units (physical coordinates)
    if input_space == "voxel":
        for i, dim_size in enumerate([W, H, D]):  # Note the order matches x, y, z as per the displacement_field
            # Normalize such that the displacement of 1 full dimension length corresponds to a move from -1 to 1
            displacement_field[..., i] = 2 * displacement_field[..., i] / (dim_size - 1)

    # Step 3: Add the displacement field to the original grid to get the transformed coordinates
    return coords + displacement_field


class Grad(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty="l1", loss_mult=None):
        super().__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [
                *range(d - 1, d + 1),
                *reversed(range(1, d - 1)),
                0,
                *range(d + 1, ndims + 2),
            ]
            df[i] = dfi.permute(r)

        return df

    def forward(self, _, y_pred):
        if self.penalty == "l1":
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == "l2", "penalty can only be l1 or l2. Got: %s" % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()


class MeanStream(nn.Module):
    """
    Maintain stream of data mean in PyTorch.

    `cap` refers to maintaining an approximation of up to that number of subjects,
    ensuring any incoming datapoint will have at least 1/cap weight.

    Adapted from:
        A.V. Dalca, M. Rakic, J. Guttag, M.R. Sabuncu.
        Learning Conditional Deformable Templates with Convolutional Networks
        NeurIPS: Advances in Neural Information Processing Systems. pp 804-816, 2019.
    """

    def __init__(self, cap=100):
        super(MeanStream, self).__init__()
        self.cap = float(cap)

        # Mean and count are registered as buffers to ensure they are saved and loaded with the model
        self.register_buffer("mean", None)  # Will be initialized dynamically
        self.register_buffer("count", torch.tensor(0.0))  # Effective count of data points seen

    def forward(self, x):
        """
        Forward pass for maintaining and updating the running mean.
        :param x: Input tensor of shape (batch_size, ...)
        :param training: Whether to update the mean and count or use the running stats (inference mode).
        :return: Scaled mean tensor of the same shape as `x`.
        """
        if self.mean is None:  # Lazy initialization of mean to match input dimensions
            self.mean = torch.zeros_like(x[0], device=x.device)

        batch_size = x.shape[0]

        if not self.training:  # Inference mode: use the current running stats
            scale = min(1.0, self.count.item() / self.cap)
            return scale * self.mean.unsqueeze(0).expand_as(x)

        # Compute new mean and count
        new_count = self.count + batch_size
        weight_old = min(1.0, self.count / (self.count + batch_size))
        weight_new = 1.0 - weight_old

        # Update mean
        new_mean = weight_old * self.mean + weight_new * x.mean(dim=0)

        # Update buffers
        self.count = min(new_count, self.cap)
        self.mean = new_mean

        # Scale the new mean
        scale = min(1.0, self.count.item() / self.cap)
        return scale * self.mean.unsqueeze(0).expand_as(x)

    def extra_repr(self):
        return f"cap={self.cap}, count={self.count.item()}"


class ConvBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, stride, norm_type, upsample=True, dim=2):
        super(ConvBlockUp, self).__init__()
        self.norm_type = norm_type
        self.up_sample = upsample

        if dim == 2:
            if norm_type == "none":
                self.norm = None
            elif norm_type == "instance":
                self.norm = nn.InstanceNorm2d(out_channels)
            elif norm_type == "batch":
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm_type == "group":
                self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
            else:
                raise NotImplementedError()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.upsample = Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        elif dim == 3:
            if norm_type == "none":
                self.norm = None
            elif norm_type == "instance":
                self.norm = nn.InstanceNorm3d(out_channels)
            elif norm_type == "batch":
                self.norm = nn.BatchNorm3d(out_channels)
            elif norm_type == "group":
                self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
            else:
                raise NotImplementedError()

            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.upsample = Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.activation = nn.ReLU(out_channels)

    def forward(self, x):
        out = self.conv(x)
        if self.norm:
            out = self.norm(out)
        out = self.activation(out)
        if self.up_sample:
            out = self.upsample(out)
        return out


class Upsample(nn.Module):
    """Upsample a multi-channel input image"""

    def __init__(self, scale_factor, mode, align_corners):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return nnf.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )
