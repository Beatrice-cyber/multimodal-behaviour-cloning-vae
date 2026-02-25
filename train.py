def trainloop(
    vae,  # VAE model
    train_loader,  # DataLoader for training
    optimizer,  # Optimizer (e.g., Adam)
    device,  # Device (e.g., "cuda" or "cpu")
    epoch,
    beta
):

    vae.train()  # Set model to training mode
    reconstruction = {"re_front" : [],"re_mount":[] ,"re_pos" :[], "re_vel" :[], "re_joint" :[],
                      "original_front":[],"original_mount":[],"original_pos":[], "original_vel":[], "original_joint":[]}

    total_loss = 0
    # this is for choosing batch to store into reconstruction
    count = 0

    for batch in train_loader:
        count += 1
        # Access individual inputs
        front_images = batch.input["front_images_ob"]
        mount_images = batch.input["mount_images_ob"]
        pos_obs = batch.input["pos_obs"]
        vel_obs = batch.input["vel_obs"]
        joint_obs = batch.input["joint_obs"]
        # Move to device
        front_images, mount_images, pos_obs, vel_obs, joint_obs = (
            front_images.to(device),
            mount_images.to(device),
            pos_obs.to(device),
            vel_obs.to(device),
            joint_obs.to(device),
        )

        # Forward pass
        mu, logvar = vae.encoder(front_images, mount_images, pos_obs, vel_obs, joint_obs)
        logvar = torch.clamp(logvar, min=-3.0, max=2.0)  # Keep variance in a reasonable range
        z = reparameterize(mu, logvar)
        recon_front_images, recon_mount_images, recon_pos_obs, recon_vel_obs, recon_joint_obs = vae.decoder(z)
        if count % 10 == 0:
            # print(f"The reconstruction in batch {count} saved ")
            reconstruction["re_front"].append(recon_front_images)
            reconstruction["re_mount"].append(recon_mount_images)
            reconstruction["re_pos"].append(recon_pos_obs)
            reconstruction["re_vel"].append(recon_vel_obs)
            reconstruction["re_joint"].append(recon_joint_obs)
            reconstruction["original_front"].append(front_images)
            reconstruction["original_mount"].append(mount_images)
            reconstruction["original_pos"].append(pos_obs)
            reconstruction["original_vel"].append(vel_obs)
            reconstruction["original_joint"].append(joint_obs)

        # Compute loss
        recon_loss, kl_loss = vae_loss(
            recon_front_images, front_images,
            recon_mount_images, mount_images,
            recon_pos_obs, pos_obs,
            recon_vel_obs, vel_obs,
            recon_joint_obs, joint_obs,
            mu, logvar
        )

        loss = recon_loss + beta * kl_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

        # Log batch-level metrics
        wandb.log({
                "batch_reconstruction_loss": recon_loss.item(),
                "batch_kl_divergence": kl_loss.item(),
                "batch_total_loss": loss.item()
            })

    # Log epoch-level metrics
    avg_epoch_loss = total_loss / len(train_loader)
    wandb.log({"epoch_total_loss": avg_epoch_loss})

    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f}, Recon loss: {recon_loss:.4f}, KL: {kl_loss:.4f}")


    return avg_epoch_loss, reconstruction