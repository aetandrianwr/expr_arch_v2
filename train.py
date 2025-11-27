import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from tqdm import tqdm
import json
from datetime import datetime

from model import RecurrentTransformer, count_parameters
from dataset import get_dataloaders
from metrics import calculate_correct_total_prediction, get_performance_dict


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, train_loader, criterion, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        loc = batch['loc'].to(device)
        user = batch['user'].to(device)
        weekday = batch['weekday'].to(device)
        start_min = batch['start_min'].to(device)
        duration = batch['duration'].to(device)
        diff = batch['diff'].to(device)
        mask = batch['mask'].to(device)
        target = batch['target'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(loc, user, weekday, start_min, duration, diff, mask)
        loss = criterion(logits, target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, data_loader, device):
    model.eval()
    
    results = {
        "correct@1": 0,
        "correct@3": 0,
        "correct@5": 0,
        "correct@10": 0,
        "rr": 0,
        "ndcg": 0,
        "f1": 0,
        "total": 0
    }
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            loc = batch['loc'].to(device)
            user = batch['user'].to(device)
            weekday = batch['weekday'].to(device)
            start_min = batch['start_min'].to(device)
            duration = batch['duration'].to(device)
            diff = batch['diff'].to(device)
            mask = batch['mask'].to(device)
            target = batch['target'].to(device)
            
            logits = model(loc, user, weekday, start_min, duration, diff, mask)
            
            batch_results, _, _ = calculate_correct_total_prediction(logits, target)
            
            results["correct@1"] += batch_results[0]
            results["correct@3"] += batch_results[1]
            results["correct@5"] += batch_results[2]
            results["correct@10"] += batch_results[3]
            results["rr"] += batch_results[4]
            results["ndcg"] += batch_results[5]
            results["total"] += batch_results[6]
    
    results["f1"] = 0
    perf = get_performance_dict(results)
    
    return perf


def train_model(
    num_epochs=100,
    batch_size=128,
    learning_rate=0.001,
    embed_dim=128,
    num_heads=4,
    num_layers=2,
    num_cycles=3,
    num_refinements=16,
    dropout=0.1,
    max_len=50,
    patience=15,
    device='cuda'
):
    set_seed(42)
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size, max_len=max_len)
    
    # Create model
    print("Creating model...")
    model = RecurrentTransformer(
        num_locations=1200,
        num_users=50,
        num_weekdays=7,
        num_start_min_bins=1440,
        num_diff_bins=100,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_cycles=num_cycles,
        num_refinements=num_refinements,
        dropout=dropout
    ).to(device)
    
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    if num_params >= 500000:
        print(f"WARNING: Model has {num_params:,} parameters, exceeding 500K limit!")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_val_acc = 0
    best_test_acc = 0
    epochs_without_improvement = 0
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        
        # Evaluate
        val_perf = evaluate(model, val_loader, device)
        test_perf = evaluate(model, test_loader, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val - Acc@1: {val_perf['acc@1']:.2f}%, Acc@5: {val_perf['acc@5']:.2f}%, MRR: {val_perf['mrr']:.2f}%")
        print(f"Test - Acc@1: {test_perf['acc@1']:.2f}%, Acc@5: {test_perf['acc@5']:.2f}%, MRR: {test_perf['mrr']:.2f}%")
        
        # Save best model
        if val_perf['acc@1'] > best_val_acc:
            best_val_acc = val_perf['acc@1']
            best_test_acc = test_perf['acc@1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'test_acc': best_test_acc,
            }, 'best_model.pt')
            print(f"✓ New best model saved! Val Acc@1: {best_val_acc:.2f}%, Test Acc@1: {best_test_acc:.2f}%")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping after {patience} epochs without improvement")
            break
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best Val Acc@1: {best_val_acc:.2f}%")
    print(f"Best Test Acc@1: {best_test_acc:.2f}%")
    print(f"{'='*60}")
    
    # Load best model and evaluate
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_perf = evaluate(model, test_loader, device)
    print(f"\nFinal Test Performance:")
    print(f"Acc@1: {test_perf['acc@1']:.2f}%")
    print(f"Acc@5: {test_perf['acc@5']:.2f}%")
    print(f"Acc@10: {test_perf['acc@10']:.2f}%")
    print(f"MRR: {test_perf['mrr']:.2f}%")
    print(f"NDCG: {test_perf['ndcg']:.2f}%")
    
    return test_perf


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    test_perf = train_model(
        num_epochs=100,
        batch_size=128,
        learning_rate=0.001,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        num_cycles=3,
        num_refinements=16,
        dropout=0.1,
        max_len=50,
        patience=15,
        device=device
    )
