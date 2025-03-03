import argparse
import json
import logging
import os
import sys
from pprint import pprint
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import L1Loss, MSELoss

# from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid


from monai.utils import set_determinism
from monai.bundle import ConfigParser
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import PatchDiscriminator

from stai_utils.datasets.dataset_utils import T1All
from stai_utils.datasets.bwm_sherlock import BWMSherlock
from stai_utils.plotting.visualize_image import (
    visualize_one_slice_in_3d_image,
    visualize_one_slice_in_3d_image_greyscale,
)

import wandb
from wandb import Image

import generative.networks.layers.registration_layers as reg_layers
from scripts.script_utils import summary


def define_instance(args, instance_def_key):
    parser = ConfigParser(vars(args))
    parser.parse(True)
    return parser.get_parsed_content(instance_def_key, instantiate=True)


def KL_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1,
        dim=list(range(1, len(z_sigma.shape))),
    )
    return torch.sum(kl_loss) / kl_loss.shape[0]


def get_slice_from_vol(volume_3d: np.ndarray, axis: int) -> np.ndarray:
    """
    Returns the middle slice of a 3D volume along the specified axis.

    volume_3d: a NumPy array of shape (D, H, W) or (some 3D shape).
    axis: which axis to slice along (0, 1, or 2).
    """
    idx = volume_3d.shape[axis] // 2
    if axis == 0:
        slice_2d = volume_3d[idx, :, :]
    elif axis == 1:
        slice_2d = volume_3d[:, idx, :]
    else:  # axis == 2
        slice_2d = volume_3d[:, :, idx]
    return slice_2d


def create_alignment_grid(
    moving_image: torch.Tensor, fixed_image: torch.Tensor, aligned_image: torch.Tensor, edge_crop: int = 0
) -> torch.Tensor:
    """
    Creates a single 3x3 grid:
      - 3 rows = axis in [0, 1, 2]
      - 3 columns = [moving, fixed, aligned]

    **Handles** differently sized slices by zero-padding them
    to the largest slice shape found among all 9 slices.

    Args:
        moving_image:   shape (B, C, D, H, W)
        fixed_image:    shape (B, C, D, H, W)
        aligned_image:  shape (B, C, D, H, W)

    Returns:
        final_grid: a torch.Tensor of shape (1, H_total, W_total)
                    suitable for passing to wandb.Image(final_grid).
    """

    # Convert to NumPy (take the first [batch, channel])
    moving_np = moving_image[0, 0, ...].cpu().detach().numpy()
    fixed_np = fixed_image[0, 0, ...].cpu().detach().numpy()
    aligned_np = aligned_image[0, 0, ...].cpu().detach().numpy()

    volumes = [moving_np, fixed_np, aligned_np]

    # 1) Collect all 9 slices
    #    (For each axis in [0,1,2], slice each volume.)
    slices_np = []
    for axis in range(3):
        for vol in volumes:
            slice_2d = get_slice_from_vol(vol, axis)
            slices_np.append(slice_2d[edge_crop:-edge_crop, edge_crop:-edge_crop])  # shape: (H, W) but might vary

    # 2) Find the maximum height and width among the 9 slices
    max_h = max(s.shape[0] for s in slices_np)
    max_w = max(s.shape[1] for s in slices_np)

    # 3) Pad each slice to (max_h, max_w)
    slices_tensors = []
    for s in slices_np:
        h, w = s.shape
        # Create a new padded array of shape (max_h, max_w)
        padded = np.zeros((max_h, max_w), dtype=s.dtype)
        padded[:h, :w] = s  # top-left corner
        # Convert to torch tensor with shape (1, max_h, max_w) for grayscale
        slice_t = torch.from_numpy(padded).unsqueeze(0).float()
        slices_tensors.append(slice_t)

    # 4) Make a single 3x3 grid using make_grid
    #    We have 9 slices total -> nrow=3 => 3 columns => 3 rows
    final_grid = make_grid(slices_tensors, nrow=3, padding=2, pad_value=1.0)  # white for padding between images
    return final_grid


def get_data(args):
    if args.dataset_type == "T1All":
        dataset = T1All(
            args.img_size,
            args.num_workers,
            age_normalization=args.age_normalization,
            rank=0,
            world_size=1,
            spacing=args.spacing,
            sample_balanced_age_for_training=args.sample_balanced_age_for_training,
        )
    elif args.dataset_type == "BWMSherlock":
        dataset = BWMSherlock(
            args.img_size,
            args.num_workers,
            age_normalization=args.age_normalization,
            rank=0,
            world_size=1,
            spacing=args.spacing,
            sample_balanced_age_for_training=args.sample_balanced_age_for_training,
        )
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")
    train_loader, val_loader = dataset.get_dataloaders(
        args.autoencoder_train["batch_size"],
    )
    print(f"Number of batches in train_loader: {len(train_loader)}")
    print(f"Number of batches in val_loader: {len(val_loader)}")
    return train_loader, val_loader, dataset


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch VAE-GAN training")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./config/environment.json",
        help="environment json file that stores environment path",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="./config/config_train_32g.json",
        help="config json file that stores hyper-parameters",
    )
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
    args = parser.parse_args()

    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)

    return args


def load_state_dict_partial(model, state_dict_path):
    """
    Loads a PyTorch state dict into a model, ignoring keys with size mismatches.

    Args:
        model (torch.nn.Module): The model to load the state dict into.
        state_dict_path (str): Path to the saved state dict file (.pth or .pt).

    Returns:
        None
    """
    print("Loading state dict from:", state_dict_path)
    # Load the state dict from the file
    state_dict = torch.load(state_dict_path, map_location=torch.device("cpu"))

    # Get the model's current state dict
    model_state_dict = model.state_dict()

    # Filter keys with size mismatches
    matched_state_dict = {}
    for key, value in state_dict.items():
        if key in model_state_dict and value.size() == model_state_dict[key].size():
            matched_state_dict[key] = value
        else:
            print(
                f"Skipping key '{key}' due to size mismatch: {value.size()} vs {model_state_dict[key].size() if key in model_state_dict else 'Key not found in model'}"
            )

    # Load the matched keys into the model
    model_state_dict.update(matched_state_dict)
    model.load_state_dict(model_state_dict)

    print("State dict loaded successfully (mismatched keys were skipped).")
    return model


def aggregate_dicts(dicts):
    result = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            result[k].append(v)
    return {k: sum(v) / len(v) for k, v in result.items()}


def train_one_epoch(
    train_loader, autoencoder, discriminator, optimizer_g, optimizer_d, intensity_loss, loss_perceptual, adv_loss, args
):
    autoencoder.train()
    discriminator.train()

    res = []

    autoencoder_warm_up_n_epochs = args.autoencoder_train["warm_up_n_epochs"]
    adv_weight = args.autoencoder_train["adv_weight"]
    for step, batch in enumerate(train_loader):
        if step == args.train_steps_per_epoch:
            break
        if step % 10 == 0:
            print("step: ", step)

        images = batch["image"].to(args.device).as_tensor()
        age = batch["age"][None].float().to(args.device)
        sex = batch["sex"][None].float().to(args.device)
        condition = torch.cat([age, sex], dim=-1)
        if args.dataset_type == "BWMSherlock":
            modality = batch["modality"][None].float().to(args.device)
            condition = torch.cat([age, sex, modality], dim=-1)
        else:
            condition = torch.cat([age, sex], dim=-1)
        del batch
        print(condition)

        # train Generator part
        optimizer_g.zero_grad(set_to_none=True)
        if args.autoencoder_def["_target_"] in [
            "generative.networks.nets.AutoencoderKLTemplateRegistrationInput",
            "generative.networks.nets.AutoencoderKLConditionalTemplateRegistrationInput",
        ]:
            reconstruction, z_mu, z_sigma, z, displacement_field = autoencoder(images, condition)
            kl_loss = KL_loss(z_mu, z_sigma)
            recons_loss = intensity_loss(reconstruction.float(), images.float())
            # p_loss = loss_perceptual(reconstruction.float(), images_fixed.float())
            p_loss = torch.tensor(0.0)
            displace_loss = F.mse_loss(displacement_field, torch.zeros_like(displacement_field))
            grad_loss = reg_layers.Grad(loss_mult=1.0)(None, displacement_field)
            loss_g = (
                recons_loss
                + args.autoencoder_train["kl_weight"] * kl_loss
                # + perceptual_weight * p_loss
                + args.autoencoder_train["displace_weight"] * displace_loss
                + args.autoencoder_train["grad_weight"] * grad_loss
            )

            train_metrics = {
                "train/recon_loss_iter": recons_loss.item(),
                "train/kl_loss_iter": kl_loss.item(),
                "train/perceptual_loss_iter": p_loss.item(),
                "train/displace_loss_iter": displace_loss.item(),
                "train/grad_loss_iter": grad_loss.item(),
            }
        else:
            reconstruction, z_mu, z_sigma, z = autoencoder(images)
            recons_loss = intensity_loss(reconstruction, images)
            kl_loss = KL_loss(z_mu, z_sigma)
            p_loss = loss_perceptual(reconstruction.float(), images.float())
            loss_g = (
                recons_loss
                + args.autoencoder_train["kl_weight"] * kl_loss
                + args.autoencoder_train["perceptual_weight"] * p_loss
            )

            train_metrics = {
                "train/recon_loss_iter": recons_loss.item(),
                "train/kl_loss_iter": kl_loss.item(),
                "train/perceptual_loss_iter": p_loss.item(),
            }

        if adv_weight > 0 and args.curr_epoch > autoencoder_warm_up_n_epochs:
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g = loss_g + adv_weight * generator_loss

        loss_g.backward()
        optimizer_g.step()

        # train Discriminator part
        if adv_weight > 0 and args.curr_epoch > autoencoder_warm_up_n_epochs:
            optimizer_d.zero_grad(set_to_none=True)
            logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
            logits_real = discriminator(images.contiguous().detach())[-1]
            loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
            loss_d = adv_weight * discriminator_loss

            loss_d.backward()
            optimizer_d.step()

            train_metrics.update(
                {
                    "train/adv_loss_iter": generator_loss.item(),
                    "train/fake_loss_iter": loss_d_fake.item(),
                    "train/real_loss_iter": loss_d_real.item(),
                }
            )

        # Convert metrics to numpy
        train_metrics = {k: torch.as_tensor(v).detach().cpu().numpy().item() for k, v in train_metrics.items()}
        res.append(train_metrics)
    return aggregate_dicts(res), images, reconstruction


def eval_one_epoch(val_loader, autoencoder, intensity_loss, loss_perceptual, args):
    autoencoder.eval()

    val_epoch_loss = 0
    val_recon_epoch_loss = 0
    val_grad_epoch_loss = 0
    val_kl_epoch_loss = 0
    for step, batch in enumerate(val_loader):
        images = batch["image"].to(args.device).as_tensor()
        age = batch["age"][None].float().to(args.device)
        sex = batch["sex"][None].float().to(args.device)
        if args.dataset_type == "BWMSherlock":
            modality = batch["modality"][None].float().to(args.device)
            condition = torch.cat([age, sex, modality], dim=-1)
        else:
            condition = torch.cat([age, sex], dim=-1)
        print(condition)

        if step == args.val_steps_per_epoch:
            break

        with torch.no_grad():
            if args.autoencoder_def["_target_"] in [
                "generative.networks.nets.AutoencoderKLTemplateRegistration",
                "generative.networks.nets.AutoencoderKLTemplateRegistrationInput",
                "generative.networks.nets.AutoencoderKLConditionalTemplateRegistration",
                "generative.networks.nets.AutoencoderKLConditionalTemplateRegistrationInput",
            ]:
                reconstruction, z_mu, z_sigma, z, displacement_field = autoencoder(images, condition)
                kl_loss = KL_loss(z_mu, z_sigma)
                recons_loss = intensity_loss(reconstruction.float(), images.float())
                # p_loss = loss_perceptual(reconstruction.float(), images_fixed.float())
                p_loss = torch.tensor(0.0)
                displace_loss = F.mse_loss(displacement_field, torch.zeros_like(displacement_field))
                grad_loss = reg_layers.Grad(loss_mult=1.0)(None, displacement_field)
                loss_g = (
                    recons_loss
                    + args.autoencoder_train["kl_weight"] * kl_loss
                    + args.autoencoder_train["perceptual_weight"] * p_loss
                    + args.autoencoder_train["displace_weight"] * displace_loss
                    + args.autoencoder_train["grad_weight"] * grad_loss
                )
            elif args.autoencoder_def["_target_"] == "generative.networks.nets.VQVAE":
                reconstruction, quantization_loss = autoencoder(images)
                recons_loss = intensity_loss(reconstruction, images)
                loss_g = recons_loss + quantization_loss
            else:
                reconstruction, z_mu, z_sigma, z = autoencoder(images)
                recons_loss = intensity_loss(reconstruction, images)
                kl_loss = KL_loss(z_mu, z_sigma)
                p_loss = loss_perceptual(reconstruction.float(), images.float())
                grad_loss = torch.tensor(0.0).to(args.device)
                loss_g = (
                    recons_loss
                    + args.autoencoder_train["kl_weight"] * kl_loss
                    + args.autoencoder_train["perceptual_weight"] * p_loss
                )

        val_epoch_loss += loss_g.item()
        val_recon_epoch_loss += recons_loss.item()
        val_grad_epoch_loss += grad_loss.item()
        val_kl_epoch_loss += kl_loss.item()

    val_epoch_loss = val_epoch_loss / (step + 1)
    val_recon_epoch_loss = val_recon_epoch_loss / (step + 1)
    val_grad_epoch_loss = val_grad_epoch_loss / (step + 1)
    val_kl_epoch_loss = val_kl_epoch_loss / (step + 1)
    return val_epoch_loss, val_recon_epoch_loss, val_grad_epoch_loss, val_kl_epoch_loss, images, reconstruction


def main():
    args = parse_args()
    pprint(vars(args))

    wandb.init(project=args.wandb_project_name_VAE, name=args.run_name, config=args)

    args.device = 0
    torch.cuda.set_device(args.device)
    print(f"Using device {args.device}")

    set_determinism(42)

    # Data
    train_loader, val_loader, dataset = get_data(args)

    # Define Autoencoder KL network and discriminator
    autoencoder = define_instance(args, "autoencoder_def").to(args.device)
    summary(autoencoder)
    discriminator = PatchDiscriminator(
        spatial_dims=args.spatial_dims, num_layers_d=3, num_channels=32, in_channels=1, out_channels=1, norm="INSTANCE"
    ).to(args.device)

    args.model_dir = os.path.join(args.base_model_dir, args.run_name)
    args.autoencoder_dir = os.path.join(args.model_dir, "autoencoder")
    args.diffusion_dir = os.path.join(args.model_dir, "diffuion")

    # Ensure directories exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.autoencoder_dir, exist_ok=True)
    os.makedirs(args.diffusion_dir, exist_ok=True)
    trained_g_path_best = os.path.join(args.autoencoder_dir, "autoencoder_best.pt")
    trained_d_path_best = os.path.join(args.autoencoder_dir, "discriminator_best.pt")

    # if args.autoencoder_train["pretrained_load_path"]:
    #     autoencoder = load_state_dict_partial(autoencoder, args.autoencoder_train["pretrained_load_path"])
    if args.resume_ckpt:
        autoencoder = load_state_dict_partial(autoencoder, trained_g_path_best)
        discriminator = load_state_dict_partial(discriminator, trained_d_path_best)

    # Losses
    if "recon_loss" in args.autoencoder_train and args.autoencoder_train["recon_loss"] == "l2":
        intensity_loss = MSELoss()
    else:
        intensity_loss = L1Loss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
    loss_perceptual.to(args.device)
    adv_weight = args.autoencoder_train["adv_weight"]

    # Optimizers
    optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=args.autoencoder_train["lr"])
    if adv_weight > 0:
        optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=args.autoencoder_train["lr"])

    # Training
    n_epochs = args.autoencoder_train["n_epochs"]
    val_interval = args.autoencoder_train["val_interval"]
    best_val_recon_epoch_loss = 100.0

    for epoch in range(1, n_epochs + 1):
        args.curr_epoch = epoch
        print("Epoch:", epoch)
        train_metrics, images, reconstruction = train_one_epoch(
            train_loader,
            autoencoder,
            discriminator,
            optimizer_g,
            optimizer_d,
            intensity_loss,
            loss_perceptual,
            adv_loss,
            args,
        )
        wandb.log(train_metrics, step=epoch)
        for axis in range(3):
            train_img = visualize_one_slice_in_3d_image_greyscale(images[0, 0, ...], axis)  # .transpose([2, 1, 0])
            train_recon = visualize_one_slice_in_3d_image_greyscale(
                reconstruction[0, 0, ...], axis
            )  # .transpose([2, 1, 0])

            wandb.log(
                {
                    f"train/image/gt_axis_{axis}": Image(train_img),
                    f"train/image/recon_axis_{axis}": Image(train_recon),
                },
                step=epoch,
            )
        if args.autoencoder_def["_target_"] in [
            "generative.networks.nets.AutoencoderKLTemplateRegistrationInput",
            "generative.networks.nets.AutoencoderKLConditionalTemplateRegistrationInput",
        ]:
            age = dataset.normalize_age([30.0])
            age = torch.tensor(age)[None].float().to(args.device)
            sex = torch.tensor([0.0])[None].float().to(args.device)
            if args.dataset_type == "BWMSherlock":
                modality = torch.tensor([0.0])[None].float().to(args.device)
                condition = torch.cat([age, sex, modality], dim=-1)
            else:
                condition = torch.cat([age, sex], dim=-1)
            images_moving = autoencoder.get_template_image(condition)

            grid_tensor = create_alignment_grid(images_moving, images, reconstruction, edge_crop=10)
            wandb.log(
                {"train/alignment_grid": wandb.Image(grid_tensor)},
                step=epoch,
            )

        # validation
        if epoch % val_interval == 0:
            val_epoch_loss, val_recon_epoch_loss, val_grad_epoch_loss, val_kl_epoch_loss, images, reconstruction = (
                eval_one_epoch(val_loader, autoencoder, intensity_loss, loss_perceptual, args)
            )

            # save last model
            print(f"Epoch {epoch} val_recon_loss: {val_recon_epoch_loss}")

            trained_g_path_epoch = os.path.join(args.autoencoder_dir, f"autoencoder_{epoch}.pt")
            trained_d_path_epoch = os.path.join(args.autoencoder_dir, f"discriminator_{epoch}.pt")

            torch.save(autoencoder.state_dict(), trained_g_path_epoch)
            torch.save(discriminator.state_dict(), trained_d_path_epoch)
            # save best model
            if val_recon_epoch_loss < best_val_recon_epoch_loss:
                best_val_recon_epoch_loss = val_recon_epoch_loss
                torch.save(autoencoder.state_dict(), trained_g_path_best)
                torch.save(discriminator.state_dict(), trained_d_path_best)
                print("Got best val recon loss.")
                print("Save trained autoencoder to", trained_g_path_best)
                print("Save trained discriminator to", trained_d_path_best)

            # write val loss for each epoch into wandb
            val_metrics = {
                "val/loss": val_epoch_loss,
                "val/recon_loss": val_recon_epoch_loss,
                "val/grad_loss": val_grad_epoch_loss,
                "val/kl_loss": val_kl_epoch_loss,
            }
            wandb.log(val_metrics, step=epoch)

            for axis in range(3):
                val_img = visualize_one_slice_in_3d_image_greyscale(images[0, 0, ...], axis)  # .transpose([2, 1, 0])
                val_recon = visualize_one_slice_in_3d_image_greyscale(
                    reconstruction[0, 0, ...], axis
                )  # .transpose([2, 1, 0])

                wandb.log(
                    {f"val/image/gt_axis_{axis}": Image(val_img), f"val/image/recon_axis_{axis}": Image(val_recon)},
                    step=epoch,
                )
            if args.autoencoder_def["_target_"] in [
                "generative.networks.nets.AutoencoderKLTemplateRegistrationInput",
                "generative.networks.nets.AutoencoderKLConditionalTemplateRegistrationInput",
            ]:
                age = dataset.normalize_age([30.0])
                age = torch.tensor(age)[None].float().to(args.device)
                sex = torch.tensor([0.0])[None].float().to(args.device)
                if args.dataset_type == "BWMSherlock":
                    modality = torch.tensor([0.0])[None].float().to(args.device)
                    condition = torch.cat([age, sex, modality], dim=-1)
                else:
                    condition = torch.cat([age, sex], dim=-1)
                images_moving = autoencoder.get_template_image(condition)
                grid_tensor = create_alignment_grid(images_moving, images, reconstruction, edge_crop=10)
                wandb.log(
                    {"val/alignment_grid": wandb.Image(grid_tensor)},
                    step=epoch,
                )

            if args.autoencoder_def["_target_"] in [
                "generative.networks.nets.AutoencoderKLConditionalTemplateRegistration",
                "generative.networks.nets.AutoencoderKLConditionalTemplateRegistrationInput",
            ]:
                if args.dataset_type == "BWMSherlock":
                    mod_vizs = [0, 1]
                else:
                    mod_vizs = [0]
                for age_viz in [0, 20, 40, 60, 80, 100]:
                    for sex_viz in [0.0, 1.0]:
                        for mod_viz in mod_vizs:
                            age = dataset.normalize_age([age_viz])
                            age = torch.tensor(age)[None].float().to(args.device)
                            sex = torch.tensor([sex_viz])[None].float().to(args.device)
                            if args.dataset_type == "BWMSherlock":
                                modality = torch.tensor([mod_viz])[None].float().to(args.device)
                                condition = torch.cat([age, sex, modality], dim=-1)
                            else:
                                condition = torch.cat([age, sex], dim=-1)
                            print(condition)
                            images_moving = autoencoder.get_template_image(condition)
                            for axis in range(3):
                                train_img_moving = visualize_one_slice_in_3d_image_greyscale(
                                    images_moving[0, 0, ...], axis
                                )  # .transpose([2, 1, 0])
                                wandb.log(
                                    {
                                        f"val/image/age_{age_viz}_sex_{sex_viz}_mod_{mod_viz}_img_moving_axis_{axis}": Image(
                                            train_img_moving
                                        ),
                                    },
                                    step=epoch,
                                )


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
    wandb.finish()