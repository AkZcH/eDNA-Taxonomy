# Training cell with detailed epoch output
def evaluate(loader):
    model.eval()
    total = 0
    correct_tax = 0
    correct_role = 0
    total_loss = 0
    with torch.no_grad():
        for padded, src_key_padding_mask, tax_idx, role_idx, novel in loader:
            padded = padded.to(device); src_key_padding_mask = src_key_padding_mask.to(device)
            tax_idx = tax_idx.to(device); role_idx = role_idx.to(device); novel = novel.to(device)
            out = model(padded, src_key_padding_mask=src_key_padding_mask)
            
            # Calculate losses
            tax_loss = ce(out['tax_logits'], tax_idx)
            role_loss = ce(out['role_logits'], role_idx)
            novel_loss = bce(out['novel_logits'], novel)
            loss = tax_loss + role_loss + 1.5*novel_loss
            total_loss += loss.item() * tax_idx.size(0)
            
            # Calculate accuracies
            correct_tax += (out['tax_logits'].argmax(dim=1) == tax_idx).sum().item()
            correct_role += (out['role_logits'].argmax(dim=1) == role_idx).sum().item()
            total += tax_idx.size(0)
    return correct_tax/total, correct_role/total, total_loss/total

print("Starting training...")
print("=" * 80)
for epoch in range(1,11):
    model.train()
    running_loss = 0.0
    running_tax_loss = 0.0
    running_role_loss = 0.0
    running_novel_loss = 0.0
    
    for batch_idx, (padded, src_key_padding_mask, tax_idx, role_idx, novel) in enumerate(train_loader):
        padded = padded.to(device); src_key_padding_mask = src_key_padding_mask.to(device)
        tax_idx = tax_idx.to(device); role_idx = role_idx.to(device); novel = novel.to(device)
        
        out = model(padded, src_key_padding_mask=src_key_padding_mask)
        
        # Individual losses
        tax_loss = ce(out['tax_logits'], tax_idx)
        role_loss = ce(out['role_logits'], role_idx)
        novel_loss = bce(out['novel_logits'], novel)
        total_loss = tax_loss + role_loss + 1.5*novel_loss
        
        opt.zero_grad()
        total_loss.backward()
        opt.step()
        
        # Accumulate losses
        batch_size = tax_idx.size(0)
        running_loss += total_loss.item() * batch_size
        running_tax_loss += tax_loss.item() * batch_size
        running_role_loss += role_loss.item() * batch_size
        running_novel_loss += novel_loss.item() * batch_size
    
    # Calculate average training losses
    avg_train_loss = running_loss / len(train)
    avg_tax_loss = running_tax_loss / len(train)
    avg_role_loss = running_role_loss / len(train)
    avg_novel_loss = running_novel_loss / len(train)
    
    # Validation metrics
    tax_acc, role_acc, val_loss = evaluate(val_loader)
    
    # Display detailed output
    print(f"Epoch {epoch:2d}/10:")
    print(f"  Train Loss: {avg_train_loss:.4f} (Tax: {avg_tax_loss:.4f}, Role: {avg_role_loss:.4f}, Novel: {avg_novel_loss:.4f})")
    print(f"  Val Loss:   {val_loss:.4f}")
    print(f"  Val Acc:    Tax: {tax_acc:.3f}, Role: {role_acc:.3f}")
    print("-" * 60)

print("Training completed!")