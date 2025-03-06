import argparse
import json
import logging
import os
import sys
from pprint import pprint
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.nn import L1Loss, MSELoss
import wandb
from monai.utils import set_determinism
from monai.bundle import ConfigParser
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import PatchDiscriminator

import morphldm.layers as reg_layers
from stai_utils.datasets.dataset_utils import T1All


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
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")
    train_loader, val_loader = dataset.get_dataloaders(
        args.autoencoder_train["batch_size"],
    )
    print(f"Number of batches in train_loader: {len(train_loader)}")
    print(f"Number of batches in val_loader: {len(val_loader)}")
    return train_loader, val_loader, dataset


def parse_args():
    parser = argparse.ArgumentParser(description="MorphLDM autoencoder training")
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
    parser.add_argument(
        "-g", "--gpus", default=1, type=int, help="number of gpus per node"
    )
    args = parser.parse_args()

    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)

    return args


def aggregate_dicts(dicts):
    result = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            result[k].append(v)
    return {k: sum(v) / len(v) for k, v in result.items()}


def train_one_epoch(
    train_loader,
    autoencoder,
    discriminator,
    optimizer_g,
    optimizer_d,
    intensity_loss,
    loss_perceptual,
    adv_loss,
    args,
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
        condition = torch.cat([age, sex], dim=-1)
        del batch
        print(condition)

        # train Generator part
        optimizer_g.zero_grad(set_to_none=True)
        if args.autoencoder_def["_target_"] in [
            "morphldm.autoencoderkl.AutoencoderKLTemplateRegistration",
            "morphldm.autoencoderkl.AutoencoderKLConditionalTemplateRegistration",
        ]:
            reconstruction, z_mu, z_sigma, z, displacement_field = autoencoder(
                images, condition
            )
            kl_loss = KL_loss(z_mu, z_sigma)
            recons_loss = intensity_loss(reconstruction.float(), images.float())
            # p_loss = loss_perceptual(reconstruction.float(), images_fixed.float())
            p_loss = torch.tensor(0.0)
            displace_loss = F.mse_loss(
                displacement_field, torch.zeros_like(displacement_field)
            )
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
            generator_loss = adv_loss(
                logits_fake, target_is_real=True, for_discriminator=False
            )
            loss_g = loss_g + adv_weight * generator_loss

        loss_g.backward()
        optimizer_g.step()

        # train Discriminator part
        if adv_weight > 0 and args.curr_epoch > autoencoder_warm_up_n_epochs:
            optimizer_d.zero_grad(set_to_none=True)
            logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = adv_loss(
                logits_fake, target_is_real=False, for_discriminator=True
            )
            logits_real = discriminator(images.contiguous().detach())[-1]
            loss_d_real = adv_loss(
                logits_real, target_is_real=True, for_discriminator=True
            )
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
        train_metrics = {
            k: torch.as_tensor(v).detach().cpu().numpy().item()
            for k, v in train_metrics.items()
        }
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
        condition = torch.cat([age, sex], dim=-1)

        if step == args.val_steps_per_epoch:
            break

        with torch.no_grad():
            if args.autoencoder_def["_target_"] in [
                "morphldm.autoencoderkl.AutoencoderKLTemplateRegistration",
                "morphldm.autoencoderkl.AutoencoderKLConditionalTemplateRegistration",
            ]:
                reconstruction, z_mu, z_sigma, z, displacement_field = autoencoder(
                    images, condition
                )
                kl_loss = KL_loss(z_mu, z_sigma)
                recons_loss = intensity_loss(reconstruction.float(), images.float())
                # p_loss = loss_perceptual(reconstruction.float(), images_fixed.float())
                p_loss = torch.tensor(0.0)
                displace_loss = F.mse_loss(
                    displacement_field, torch.zeros_like(displacement_field)
                )
                grad_loss = reg_layers.Grad(loss_mult=1.0)(None, displacement_field)
                loss_g = (
                    recons_loss
                    + args.autoencoder_train["kl_weight"] * kl_loss
                    + args.autoencoder_train["perceptual_weight"] * p_loss
                    + args.autoencoder_train["displace_weight"] * displace_loss
                    + args.autoencoder_train["grad_weight"] * grad_loss
                )
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
    return (
        val_epoch_loss,
        val_recon_epoch_loss,
        val_grad_epoch_loss,
        val_kl_epoch_loss,
        images,
        reconstruction,
    )


def main():
    args = parse_args()
    pprint(vars(args))

    wandb.init(project=args.wandb_project_name, name=args.run_name, config=args)

    args.device = 0
    torch.cuda.set_device(args.device)
    print(f"Using device {args.device}")

    set_determinism(42)

    # Data
    train_loader, val_loader, dataset = get_data(args)

    # Define Autoencoder KL network and discriminator
    autoencoder = define_instance(args, "autoencoder_def").to(args.device)
    discriminator = PatchDiscriminator(
        spatial_dims=args.spatial_dims,
        num_layers_d=3,
        num_channels=32,
        in_channels=1,
        out_channels=1,
        norm="INSTANCE",
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

    # Losses
    if (
        "recon_loss" in args.autoencoder_train
        and args.autoencoder_train["recon_loss"] == "l2"
    ):
        intensity_loss = MSELoss()
    else:
        intensity_loss = L1Loss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = PerceptualLoss(
        spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2
    )
    loss_perceptual.to(args.device)
    adv_weight = args.autoencoder_train["adv_weight"]

    # Optimizers
    optimizer_g = torch.optim.Adam(
        params=autoencoder.parameters(), lr=args.autoencoder_train["lr"]
    )
    if adv_weight > 0:
        optimizer_d = torch.optim.Adam(
            params=discriminator.parameters(), lr=args.autoencoder_train["lr"]
        )

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

        # validation
        if epoch % val_interval == 0:
            (
                val_epoch_loss,
                val_recon_epoch_loss,
                val_grad_epoch_loss,
                val_kl_epoch_loss,
                images,
                reconstruction,
            ) = eval_one_epoch(
                val_loader, autoencoder, intensity_loss, loss_perceptual, args
            )

            # save last model
            print(f"Epoch {epoch} val_recon_loss: {val_recon_epoch_loss}")

            trained_g_path_epoch = os.path.join(
                args.autoencoder_dir, f"autoencoder_{epoch}.pt"
            )
            trained_d_path_epoch = os.path.join(
                args.autoencoder_dir, f"discriminator_{epoch}.pt"
            )

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


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
    wandb.finish()
