"""
Visualize test set predictions for Task 1: Flower Recognition.

Generates:
1. Sample predictions (random images with predictions and confidence)
2. Confusion matrix heatmap
3. Error analysis (misclassified samples)

Usage:
    python visualize_predictions.py --checkpoint checkpoints/baseline_r18_best.pth
    python visualize_predictions.py --checkpoint checkpoints/se_resnet18_best.pth
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import DATA_ROOT, NUM_CLASSES
from models import resnet18, resnet34, se_resnet18, cbam_resnet18
from utils.dataset import Flower102Dataset, get_transforms

MODEL_REGISTRY = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "se_resnet18": se_resnet18,
    "cbam_resnet18": cbam_resnet18,
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def denormalize(tensor):
    """Denormalize image tensor for display."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return tensor * std + mean


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    # Detect model type from checkpoint name
    if "baseline_r34" in checkpoint_path:
        model = resnet34(num_classes=NUM_CLASSES, pretrained=False)
    elif "se_resnet18" in checkpoint_path:
        model = se_resnet18(num_classes=NUM_CLASSES, pretrained=False)
    elif "cbam_resnet18" in checkpoint_path:
        model = cbam_resnet18(num_classes=NUM_CLASSES, pretrained=False)
    else:
        model = resnet18(num_classes=NUM_CLASSES, pretrained=False)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Extract model name for display
    if "baseline_r34" in checkpoint_path:
        model_name = "ResNet-34"
    elif "se_resnet18" in checkpoint_path:
        model_name = "SE-ResNet-18"
    elif "cbam_resnet18" in checkpoint_path:
        model_name = "CBAM-ResNet-18"
    else:
        model_name = "ResNet-18"
    
    return model, model_name


def visualize_predictions(model, dataset, device, model_name, save_path="predictions.png"):
    """Visualize random test samples with predictions."""
    model.eval()
    
    # Randomly sample 9 images
    np.random.seed(42)
    indices = np.random.choice(len(dataset), size=9, replace=False)
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    correct_count = 0
    
    for idx, ax in zip(indices, axes):
        image, label = dataset[idx]
        
        # Get prediction
        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(device)
            output = model(image_tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()
        
        # Denormalize and display
        image_display = denormalize(image)
        image_display = np.clip(image_display.numpy().transpose(1, 2, 0), 0, 1)
        
        ax.imshow(image_display)
        is_correct = pred == label
        correct_count += is_correct
        
        color = 'green' if is_correct else 'red'
        ax.set_title(f'Pred: {pred}\nConf: {confidence:.1%}', 
                    color=color, fontsize=10, fontweight='bold')
        ax.set_xlabel(f'True: {label}', fontsize=9)
        ax.axis('off')
    
    fig.suptitle(f'{model_name} - Test Set Predictions (Accuracy: {correct_count}/9)', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved prediction visualization to {save_path}")


def plot_confusion_matrix(model, dataset, device, model_name, save_path="confusion_matrix.png"):
    """Plot confusion matrix heatmap."""
    model.eval()
    
    # Get all predictions
    all_preds = []
    all_labels = []
    
    for i in range(len(dataset)):
        image, label = dataset[i]
        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(device)
            output = model(image_tensor)
            pred = torch.argmax(output, dim=1).item()
        
        all_preds.append(pred)
        all_labels.append(label)
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(NUM_CLASSES))
    
    # Normalize by row
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot top 20 classes (most frequent in confusion)
    # For 102 classes, we'll show the full matrix but it might be dense
    # Let's create a focused view showing main diagonal and some off-diagonal
    fig, ax = plt.subplots(figsize=(20, 18))
    
    im = ax.imshow(cm_normalized, cmap='Blues', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(range(NUM_CLASSES), fontsize=8)
    ax.set_yticklabels(range(NUM_CLASSES), fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Count', rotation=270, labelpad=20)
    
    # Add text annotations for high values
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if cm_normalized[i, j] > 0.1:  # Only annotate significant values
                ax.text(j, i, f'{cm_normalized[i, j]:.2f}',
                       ha='center', va='center', fontsize=6,
                       color='red' if cm_normalized[i, j] < 0.5 else 'black')
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name} - Confusion Matrix (102 Classes)\n'
                f'Overall Accuracy: {np.sum(np.diag(cm))/np.sum(cm):.2%}',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrix to {save_path}")
    
    return cm, cm_normalized


def visualize_errors(model, dataset, device, model_name, save_path="error_analysis.png", max_errors=12):
    """Visualize misclassified samples."""
    model.eval()
    
    # Find all misclassified samples
    errors = []
    for i in range(len(dataset)):
        image, label = dataset[i]
        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(device)
            output = model(image_tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()
        
        if pred != label:
            errors.append({
                'idx': i,
                'image': image,
                'true_label': label,
                'pred_label': pred,
                'confidence': confidence
            })
    
    print(f"Found {len(errors)} misclassified samples out of {len(dataset)}")
    
    # Randomly sample errors to display
    if len(errors) > max_errors:
        np.random.seed(42)
        sampled_errors = np.random.choice(errors, size=max_errors, replace=False)
    else:
        sampled_errors = errors
    
    # Determine grid size
    n = len(sampled_errors)
    rows = (n + 2) // 3
    cols = min(3, n)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if n == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for idx, (error, ax) in enumerate(zip(sampled_errors, axes)):
        image = error['image']
        true_label = error['true_label']
        pred_label = error['pred_label']
        confidence = error['confidence']
        
        # Denormalize and display
        image_display = denormalize(image)
        image_display = np.clip(image_display.numpy().transpose(1, 2, 0), 0, 1)
        
        ax.imshow(image_display)
        ax.set_title(f'True: {true_label} | Pred: {pred_label}\n'
                    f'Confidence: {confidence:.1%}',
                    fontsize=10, color='red', fontweight='bold')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'{model_name} - Error Analysis ({n} Misclassified Samples)', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved error analysis to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize test set predictions")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint (.pth file)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for inference")
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, model_name = load_model(args.checkpoint, args.device)
    print(f"✓ Model loaded: {model_name}")
    
    # Load test dataset
    print("Loading test dataset...")
    _, val_transform = get_transforms()
    test_dataset = Flower102Dataset(DATA_ROOT, split="test", transform=val_transform)
    print(f"✓ Loaded {len(test_dataset)} test samples")
    
    # Generate visualizations
    checkpoint_name = os.path.basename(args.checkpoint).replace('.pth', '')
    
    print("\nGenerating visualizations...")
    
    # 1. Sample predictions
    visualize_predictions(
        model, test_dataset, args.device, model_name,
        save_path=f"{checkpoint_name}_predictions.png"
    )
    
    # 2. Confusion matrix
    plot_confusion_matrix(
        model, test_dataset, args.device, model_name,
        save_path=f"{checkpoint_name}_confusion_matrix.png"
    )
    
    # 3. Error analysis
    visualize_errors(
        model, test_dataset, args.device, model_name,
        save_path=f"{checkpoint_name}_error_analysis.png"
    )
    
    print("\n✓ All visualizations generated successfully!")
    print(f"  - {checkpoint_name}_predictions.png")
    print(f"  - {checkpoint_name}_confusion_matrix.png")
    print(f"  - {checkpoint_name}_error_analysis.png")


if __name__ == "__main__":
    main()
