
# Fire and Smoke Detection with DETR

This repository provides an implementation for training an object detection model using the DETR (DEtection TRansformers) architecture with custom backbones and dataset integration. The implementation is based on PyTorch Lightning for efficient training and logging.

## Project Structure

```
├── config.py               # Configuration settings (paths, model parameters, etc.)
├── train.py                # Main training script
├── train_r50.py            # Training script with ResNet50 backbone (Normal Configuration)
├── train_r50_3_4.py        # Training script with ResNet50 backbone (DC Layer 3-4)
├── train_r50_4.py          # Training script with ResNet50 backbone (DC Layer 1-4)
├── train_r101.py           # Training script with ResNet101 backbone (Normal Configuration)
├── train_r101_3_4.py       # Training script with ResNet101 backbone (DC Layer 3-4)
├── train_r101_4.py         # Training script with ResNet101 backbone (DC Layer 1-4)
├── train_r50_1_4.py        # Training script with ResNet50 backbone (DC Layer 1-4)
├── testing.py              # Testing Script for evaluation
├── data/
│   ├── dataset.py          # Dataset handler for COCO-style datasets
├── utils/
│   ├── helpers.py          # Helper functions for data processing and utility
├── logging_config.py       # Logging configurations
```

## Requirements

Ensure you have the following dependencies installed:

- Python 3.x
- PyTorch
- PyTorch Lightning
- Hugging Face Transformers
- Roboflow API (for dataset management)
- Matplotlib
- Pandas
- NumPy

You can install the necessary dependencies via pip:

```bash
pip install -r requirements.txt
```

## Setup

1. **Download the Dataset:**
   - Use the `Roboflow` API to download a custom dataset or use the pre-defined `COCO` dataset.
   - The datasets are downloaded using the `Roboflow` API key. Replace the API key in `config.py` with your own.

2. **Configure Training Settings:**
   - Set the appropriate model checkpoint, batch size, number of epochs, and other training parameters in the `config.py` file.

3. **Modify the Model Backbone (Optional):**
   - The model can use different backbones like `ResNet50` or `ResNet101`, depending on which script is used:
     - `train_r50.py`: Normal configuration with ResNet50.
     - `train_r50_1_4.py`: ResNet50 with DC Layer 1-4.
     - `train_r50_3_4.py`: ResNet50 with DC Layer 3-4.
     - `train_r101.py`: Normal configuration with ResNet101.
     - `train_r101_1_4.py`: ResNet101 with DC Layer 1-4.
     - `train_r101_3_4.py`: ResNet101 with DC Layer 3-4.

## Training

1. **Run the training script:**
   You can start the training using the following command, depending on the backbone and configuration you want to use:

   ```bash
   python train_r50.py           # ResNet50 with normal configuration
   python train_r50_3_4.py       # ResNet50 with DC Layer 3-4
   python train_r50_1_4.py       # ResNet50 with DC Layer 1-4
   python train_r101.py          # ResNet101 with normal configuration
   python train_r101_3_4.py      # ResNet101 with DC Layer 3-4
   python train_r101_1_4.py      # ResNet101 with DC Layer 1-4
   ```

2. **TensorBoard for Visualization:**
   - TensorBoard logging is set up by default to track the training losses and metrics.
   - The logs are stored in the `training_logs` directory and can be visualized using:

   ```bash
   tensorboard --logdir=training_logs
   ```

## Model Checkpoint

- Once training completes, the model will be saved in the directory defined by `MODEL_PATH` in `config.py`.

## Notes

- The training scripts are designed for high-performance training on GPUs. Ensure that your system has CUDA enabled if you're training on GPU.
- The scripts include visualization of loss values at the end of each epoch, allowing for better monitoring of training progress.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
