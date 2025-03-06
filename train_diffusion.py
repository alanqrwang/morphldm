import argparse
import json
import logging
import os
import sys
from pprint import pprint
import torch
import torch.nn.functional as F
from generative.networks.schedulers import DDPMScheduler
from monai.utils import first, set_determinism
from monai.bundle import ConfigParser
from torch.amp import GradScaler
import wandb
from wandb import Image

from stai_utils.datasets.dataset_utils import T1All
from morphldm.inferer import LatentDiffusionInferer

def visualize_one_slice_in_3d_image(image, axis: int = 2):
    """
    Prepare a 2D image slice from a 3D image for visualization.
    Args:
        image: image numpy array, sized (H, W, D)
    """
    image = convert_to_numpy(image)
    # draw image
    center = image.shape[axis] // 2
    if axis == 0:
        draw_img = normalize_image_to_uint8(image[center, :, :])
    elif axis == 1:
        draw_img = normalize_image_to_uint8(image[:, center, :])
    elif axis == 2:
        draw_img = normalize_image_to_uint8(image[:, :, center])
    else:
        raise ValueError("axis should be in [0,1,2]")
    draw_img = np.stack([draw_img, draw_img, draw_img], axis=-1)
    return draw_img

def define_instance(args, instance_def_key):
    parser = ConfigParser(vars(args))
    parser.parse(True)
    return parser.get_parsed_content(instance_def_key, instantiate=True)


def get_data(args):
    dataset = T1All(
        args.img_size,
        args.num_workers,
        age_normalization=args.age_normalization,
        rank=0,
        world_size=1,
        spacing=args.spacing,
        sample_balanced_age_for_training=args.sample_balanced_age_for_training,
    )
    train_loader, val_loader = dataset.get_dataloaders(
        args.autoencoder_train["batch_size"], debug_one_sample=args.debug_one_sample
    )
    return train_loader, val_loader


def train_one_epoch(train_loader, unet, autoencoder, inferer, optimizer, noise_shape, args, scaler=None):
    unet.train()

    train_recon_epoch_loss = 0
    for step, batch in enumerate(train_loader):
        if step == args.train_steps_per_epoch:
            break
        if step % 10 == 0:
            print("Step:", step)

        images = batch["image"].to(args.device)
        if args.diffusion_def["with_conditioning"]:
            age = batch["age"][None].float().to(args.device)
            sex = batch["sex"][None].float().to(args.device)
            condition = torch.cat([age, sex], dim=-1).unsqueeze(1)  # for seq_len
        else:
            condition = None

        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        with torch.amp.autocast("cuda", enabled=args.use_amp):
            # Generate random noise
            noise = torch.randn(noise_shape, dtype=images.dtype).to(args.device)

            # Create timesteps
            timesteps = torch.randint(
                0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
            ).long()

            with torch.no_grad():
                if args.autoencoder_def["_target_"] in [
                    "morphldm.autoencoderkl.AutoencoderKLTemplateRegistration",
                    "morphldm.autoencoderkl.AutoencoderKLConditionalTemplateRegistration",
                ]:
                    template = autoencoder.get_template_image(condition).detach()
                else:
                    template = None

            noise_pred = inferer(
                inputs=images,
                autoencoder_model=autoencoder,
                diffusion_model=unet,
                noise=noise,
                timesteps=timesteps,
                condition=condition,
                template=template,
                plot_img=args.debug_mode and step == 0,
            )

            loss = F.mse_loss(noise_pred.float(), noise.float())
            train_recon_epoch_loss += loss.item()

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

    train_recon_epoch_loss = train_recon_epoch_loss / (step + 1)
    return train_recon_epoch_loss


def eval_one_epoch(val_loader, unet, autoencoder, inferer, noise_shape, args):
    autoencoder.eval()
    unet.eval()
    val_recon_epoch_loss = 0
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=args.use_amp):
            for step, batch in enumerate(val_loader):
                if step == args.val_steps_per_epoch:
                    break
                images = batch["image"].to(args.device)

                if args.diffusion_def["with_conditioning"]:
                    age = batch["age"][None].float().to(args.device)
                    sex = batch["sex"][None].float().to(args.device)
                    condition = torch.cat([age, sex], dim=-1).unsqueeze(1)  # for seq_len
                else:
                    condition = None

                noise = torch.randn(noise_shape, dtype=images.dtype).to(args.device)
                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                ).long()

                if args.autoencoder_def["_target_"] in [
                    "morphldm.autoencoderkl.AutoencoderKLTemplateRegistration",
                    "morphldm.autoencoderkl.AutoencoderKLConditionalTemplateRegistration",
                ]:
                    template = autoencoder.get_template_image(condition)
                else:
                    template = None

                noise_pred = inferer(
                    inputs=images,
                    autoencoder_model=autoencoder,
                    diffusion_model=unet,
                    noise=noise,
                    timesteps=timesteps,
                    condition=condition,
                    template=template,
                )

                val_loss = F.mse_loss(noise_pred.float(), noise.float())
                val_recon_epoch_loss += val_loss.item()
    val_recon_epoch_loss = val_recon_epoch_loss / (step + 1)
    return val_recon_epoch_loss


def synthesize_example_image(unet, autoencoder, inferer, scheduler, noise_shape, args):
    # Generate random noise
    noise = torch.randn(noise_shape).to(args.device)

    age = torch.tensor([10.0])[None].float().to(args.device)
    sex = torch.tensor([0.0])[None].float().to(args.device)
    condition = torch.cat([age, sex], dim=-1).unsqueeze(1)  # for seq_len
    if args.autoencoder_def["_target_"] in [
        "morphldm.autoencoderkl.AutoencoderKLTemplateRegistration",
        "morphldm.autoencoderkl.AutoencoderKLConditionalTemplateRegistration",
    ]:
        template = autoencoder.get_template_image(condition[:, 0])
    else:
        template = None
    synthetic_images = inferer.sample(
        input_noise=noise[0:1, ...],
        autoencoder_model=autoencoder,
        diffusion_model=unet,
        scheduler=scheduler,
        conditioning=condition,
        template=template,
    )
    return synthetic_images


def recon_example_image(x, autoencoder, template=None):
    if template is not None:
        z = autoencoder.encode_stage_2_inputs(x, template)
        return autoencoder.decode_stage_2_outputs(z, template)
    else:
        z = autoencoder.encode_stage_2_inputs(x)
        return autoencoder.decode_stage_2_outputs(z)


def parse_args():
    parser = argparse.ArgumentParser(description="MorphLDM Diffusion Training")
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
    parser.add_argument("--debug-mode", action=argparse.BooleanOptionalAction)
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
    args = parser.parse_args()

    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)

    return args


def main():
    args = parse_args()
    pprint(vars(args))

    wandb.init(
        project=args.wandb_project_name, name=args.run_name.replace("__auto__", "__diff__"), config=args
    )
    # Save the current training script to the wandb run
    wandb.save(__file__)

    args.device = 0
    torch.cuda.set_device(args.device)
    print(f"Using {args.device}")

    set_determinism(42)

    args.model_dir = os.path.join(args.base_model_dir, args.run_name)
    args.autoencoder_dir = os.path.join(args.model_dir, "autoencoder")
    args.diffusion_dir = os.path.join(args.model_dir, "diffuion")
    os.makedirs(args.diffusion_dir, exist_ok=True)

    # Data
    train_loader, val_loader = get_data(args)

    # Load Autoencoder KL network
    autoencoder = define_instance(args, "autoencoder_def").to(args.device)
    autoencoder_path = os.path.join(args.autoencoder_dir, f"autoencoder_{args.autoencoder_ckpt_name}.pt")
    autoencoder_state_dict = torch.load(autoencoder_path, map_location="cpu")
    autoencoder_state_dict.pop("template_image", None)
    autoencoder.load_state_dict(autoencoder_state_dict)
    print(f"Load trained autoencoder from {autoencoder_path}")

    # Compute Scaling factor
    # As mentioned in Rombach et al. [1] Section 4.3.2 and D.1, the signal-to-noise ratio (induced by the scale of the latent space) can affect the results obtained with the LDM,
    # if the standard deviation of the latent space distribution drifts too much from that of a Gaussian.
    # For this reason, it is best practice to use a scaling factor to adapt this standard deviation.
    # _Note: In case where the latent space is close to a Gaussian distribution, the scaling factor will be close to one,
    # and the results will not differ from those obtained when it is not used._

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=args.use_amp):
            check_data = first(train_loader)
            check_image = check_data["image"].float().to(args.device)
            check_age = check_data["age"][None].float().to(args.device)
            check_sex = check_data["sex"][None].float().to(args.device)
            if args.autoencoder_def["_target_"] in [
                "morphldm.autoencoderkl.AutoencoderKLTemplateRegistration",
                "morphldm.autoencoderkl.AutoencoderKLConditionalTemplateRegistration",
            ]:
                check_metadata = torch.cat([check_age, check_sex], dim=-1)
                check_template = autoencoder.get_template_image(check_metadata)
                z = autoencoder.encode_stage_2_inputs(check_image, check_template)
                recon_images = recon_example_image(check_image, autoencoder, check_template)
            else:
                z = autoencoder.encode_stage_2_inputs(check_image)
                recon_images = recon_example_image(check_image, autoencoder)
            print(f"Latent feature shape {z.shape}")

    scale_factor = 1 / torch.std(z)
    print(f"scale_factor: {scale_factor}")
    noise_shape = [check_data["image"].shape[0]] + [args.latent_channels, 40, 48, 48]  # list(z.shape[1:])

    # Define Diffusion Model
    unet = define_instance(args, "diffusion_def").to(args.device)

    trained_diffusion_path_best = os.path.join(args.diffusion_dir, "diffusion_unet_best.pt")

    if args.NoiseScheduler["schedule"] == "cosine":
        scheduler = DDPMScheduler(
            num_train_timesteps=args.NoiseScheduler["num_train_timesteps"],
            schedule=args.NoiseScheduler["schedule"],
            clip_sample=args.NoiseScheduler["clip_sample"],
        )
    else:
        scheduler = DDPMScheduler(
            num_train_timesteps=args.NoiseScheduler["num_train_timesteps"],
            schedule=args.NoiseScheduler["schedule"],
            beta_start=args.NoiseScheduler["beta_start"],
            beta_end=args.NoiseScheduler["beta_end"],
            clip_sample=args.NoiseScheduler["clip_sample"],
        )

    # We define the inferer using the scale factor:
    inferer = LatentDiffusionInferer(
        scheduler, scale_factor=scale_factor, ldm_latent_shape=(40, 48, 48), autoencoder_latent_shape=(40, 48, 44)
    )

    # Step 3: training config
    optimizer_diff = torch.optim.AdamW(
        unet.parameters(),
        lr=args.diffusion_train["lr"],
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_diff, milestones=args.diffusion_train["lr_scheduler_milestones"], gamma=0.1
    )

    n_epochs = args.diffusion_train["n_epochs"]
    val_interval = args.diffusion_train["val_interval"]
    autoencoder.eval()
    scaler = GradScaler("cuda")
    best_val_recon_epoch_loss = 100.0

    for epoch in range(1, n_epochs + 1):
        print("Epoch: ", epoch)
        train_epoch_loss = train_one_epoch(
            train_loader,
            unet,
            autoencoder,
            inferer,
            optimizer_diff,
            noise_shape,
            args,
            scaler=scaler,
        )

        # write to wandb
        wandb.log(
            {
                "train/diffusion_loss": train_epoch_loss,
                "train/lr": optimizer_diff.param_groups[0]["lr"],
            },
            step=epoch,
        )
        lr_scheduler.step()

        if epoch % val_interval == 0:
            val_epoch_loss = eval_one_epoch(
                val_loader,
                unet,
                autoencoder,
                inferer,
                noise_shape,
                args,
            )

            # write to wandb
            trained_diffusion_path_epoch = os.path.join(args.diffusion_dir, f"diffusion_unet_{epoch}.pt")
            wandb.log(
                {
                    "val/diffusion_loss": val_epoch_loss,
                },
                step=epoch,
            )
            print(f"Epoch {epoch} val_diffusion_loss: {val_epoch_loss}")

            # save last model
            ckpt_dict = {
                "state_dict": unet.state_dict(),
                "epoch": epoch,
            }
            torch.save(ckpt_dict, trained_diffusion_path_epoch)

            # save best model
            if val_epoch_loss < best_val_recon_epoch_loss:
                best_val_recon_epoch_loss = val_epoch_loss
                torch.save(ckpt_dict, trained_diffusion_path_best)
                print("Got best val noise pred loss.")
                print("Save trained latent diffusion model to", trained_diffusion_path_best)

            # visualize synthesized image
            synthetic_images = synthesize_example_image(unet, autoencoder, inferer, scheduler, noise_shape, args)

            for axis in range(3):
                synthetic_img = visualize_one_slice_in_3d_image(synthetic_images[0, 0, ...], axis)

                wandb.log(
                    {
                        f"val/image/syn_axis_{axis}": Image(synthetic_img),
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
