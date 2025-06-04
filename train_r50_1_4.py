import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from roboflow import Roboflow
from transformers import DetrImageProcessor, DetrForObjectDetection
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from datetime import datetime
from config import *
from data.dataset import CocoDetection, collate_fn


class Detr(pl.LightningModule):
    def __init__(self, num_labels=91, lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=CHECKPOINT,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

        # Initialize loss tracking
        self.train_losses = []
        self.val_losses = []

        # Create directory for saving logs
        self.log_dir = Path('training_logs') / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created log directory: {self.log_dir}")

    def training_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        # Log losses
        loss_dict = {
            'epoch': self.current_epoch,
            'step': self.global_step,
            'total_loss': outputs.loss.item(),
            'loss_ce': outputs.loss_dict['loss_ce'].item(),
            'loss_bbox': outputs.loss_dict['loss_bbox'].item(),
            'loss_giou': outputs.loss_dict['loss_giou'].item()
        }
        self.train_losses.append(loss_dict)

        # Log to TensorBoard or other logger
        for name, value in outputs.loss_dict.items():
            self.log(f"train_{name}", value.item())

        return outputs.loss

    def validation_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        # Log validation losses
        val_loss_dict = {
            'epoch': self.current_epoch,
            'step': self.global_step,
            'total_loss': outputs.loss.item(),
            'loss_ce': outputs.loss_dict['loss_ce'].item(),
            'loss_bbox': outputs.loss_dict['loss_bbox'].item(),
            'loss_giou': outputs.loss_dict['loss_giou'].item()
        }
        self.val_losses.append(val_loss_dict)

        # Log to TensorBoard or other logger
        for name, value in outputs.loss_dict.items():
            self.log(f"val_{name}", value.item())

    def on_train_epoch_end(self):
        # Convert list of losses to DataFrame
        train_df = pd.DataFrame(self.train_losses)

        # Save training losses to CSV
        train_csv_path = self.log_dir / 'training_losses.csv'
        train_df.to_csv(train_csv_path, index=False)

        # Calculate and print epoch statistics
        epoch_stats = train_df[train_df['epoch'] == self.current_epoch].mean()
        print(f"\nEpoch {self.current_epoch} statistics:")
        print(f"Total Loss: {epoch_stats['total_loss']:.4f}")
        print(f"Classification Loss: {epoch_stats['loss_ce']:.4f}")
        print(f"Bbox Loss: {epoch_stats['loss_bbox']:.4f}")
        print(f"GIoU Loss: {epoch_stats['loss_giou']:.4f}")

        # Visualize training losses
        self.plot_losses(train_df, 'training')

    def on_validation_epoch_end(self):
        # Convert list of validation losses to DataFrame
        val_df = pd.DataFrame(self.val_losses)

        # Save validation losses to CSV
        val_csv_path = self.log_dir / 'validation_losses.csv'
        val_df.to_csv(val_csv_path, index=False)

        # Visualize validation losses
        self.plot_losses(val_df, 'validation')

    def plot_losses(self, df, phase):
        plt.figure(figsize=(10, 5))
        plt.plot(df['step'], df['total_loss'], label="Total Loss", color='blue')
        plt.plot(df['step'], df['loss_ce'], label="Classification Loss", color='red')
        plt.plot(df['step'], df['loss_bbox'], label="Bbox Loss", color='green')
        plt.plot(df['step'], df['loss_giou'], label="GIoU Loss", color='purple')

        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title(f"{phase.capitalize()} Loss (Epoch {self.current_epoch})")
        plt.legend()
        plt.grid(True)

        plot_path = self.log_dir / f'{phase}_loss_epoch_{self.current_epoch}.png'
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved {phase} loss plot: {plot_path}")

    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad],
                "weight_decay": 1e-4,  # Weight decay untuk detektor
            },
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr)

        # Scheduler dengan warmup
        def lr_lambda(epoch):
            if epoch < 10:  # Warmup selama 10 epoch
                return epoch / 10
            else:
                return 0.1 ** ((epoch - 10) // 30)  # Decay learning rate setiap 30 epoch

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [scheduler]

    @classmethod
    def from_pretrained(cls, model_path, num_labels=91):
        model = cls(num_labels=num_labels)
        model.model = DetrForObjectDetection.from_pretrained(
            model_path,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        return model


def setup_data():
    # Download dataset dari Roboflow
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace("fire-qdubq").project("detr_api")
    version = project.version(1)
    dataset = version.download("coco")

    # Setup direktori dataset
    train_dir = os.path.join(dataset.location, "train")
    val_dir = os.path.join(dataset.location, "valid")
    test_dir = os.path.join(dataset.location, "test")

    image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)

    # Buat dataset
    train_dataset = CocoDetection(train_dir, image_processor, train=True)
    val_dataset = CocoDetection(val_dir, image_processor, train=False)
    test_dataset = CocoDetection(test_dir, image_processor, train=False)

    # Buat DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=lambda b: collate_fn(b, image_processor),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    val_dataloader = DataLoader(
        val_dataset,
        collate_fn=lambda b: collate_fn(b, image_processor),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=lambda b: collate_fn(b, image_processor),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    return train_dataloader, val_dataloader, test_dataloader, train_dataset.coco.cats


def main():
    # Setup data
    train_dataloader, val_dataloader, test_dataloader, categories = setup_data()
    id2label = {k: v['name'] for k, v in categories.items()}

    # Initialize model
    model = Detr(
        num_labels=len(id2label),
        lr=1e-4,
        lr_backbone=1e-5,
        weight_decay=1e-4
    )

    # Train model
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        max_epochs=MAX_EPOCHS,
        gradient_clip_val=0.1,
        accumulate_grad_batches=8,
        log_every_n_steps=5,
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )
    # Save model
    model.model.save_pretrained(MODEL_PATH)


if __name__ == "__main__":
    main()