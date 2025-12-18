#!/usr/bin/env python3
"""
TriX 6502: ATOMIC

Each operation is an ATOM. Trained separately to 100%.
Routing COMPOSES atoms.

ADC_C1 = ADC_C0 → INC (composition, not learning)
"""

import sys
sys.path.insert(0, '/workspace/trix_latest/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

# Atomic operations
ATOMS = ['ADD', 'AND', 'ORA', 'EOR', 'ASL', 'LSR', 'INC', 'DEC']


def compute_atom(atom, a, b=0):
    """Pure atomic computation. No carry input - that's composition."""
    if atom == 'ADD':
        return (a + b) & 0xFF
    elif atom == 'AND':
        return a & b
    elif atom == 'ORA':
        return a | b
    elif atom == 'EOR':
        return a ^ b
    elif atom == 'ASL':
        return (a << 1) & 0xFF
    elif atom == 'LSR':
        return a >> 1
    elif atom == 'INC':
        return (a + 1) & 0xFF
    elif atom == 'DEC':
        return (a - 1) & 0xFF


class Atom(nn.Module):
    """Single atomic operation. Tiny, specialized, perfect."""
    
    def __init__(self, needs_b=True):
        super().__init__()
        input_dim = 16 if needs_b else 8  # a_bits + b_bits or just a_bits
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.Sigmoid(),
        )
    
    def forward(self, a_bits, b_bits=None):
        if b_bits is not None:
            x = torch.cat([a_bits, b_bits], dim=-1)
        else:
            x = a_bits
        return self.net(x)


def train_atom(atom_name, needs_b, epochs=100, device='cuda'):
    """Train single atom to 100%."""
    atom = Atom(needs_b=needs_b).to(device)
    opt = torch.optim.Adam(atom.parameters(), lr=0.001)
    
    # Generate exhaustive data for this atom
    data = []
    for a in range(256):
        if needs_b:
            for b in range(256):
                result = compute_atom(atom_name, a, b)
                data.append((a, b, result))
        else:
            result = compute_atom(atom_name, a)
            data.append((a, 0, result))
    
    # To tensors
    a_vals = torch.tensor([d[0] for d in data], device=device)
    b_vals = torch.tensor([d[1] for d in data], device=device)
    results = torch.tensor([d[2] for d in data], device=device)
    
    a_bits = torch.stack([(a_vals >> i) & 1 for i in range(8)], dim=1).float()
    b_bits = torch.stack([(b_vals >> i) & 1 for i in range(8)], dim=1).float()
    result_bits = torch.stack([(results >> i) & 1 for i in range(8)], dim=1).float()
    
    best_acc = 0
    for epoch in range(epochs):
        atom.train()
        
        if needs_b:
            pred = atom(a_bits, b_bits)
        else:
            pred = atom(a_bits)
        
        loss = F.binary_cross_entropy(pred, result_bits)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        # Check accuracy
        with torch.no_grad():
            pred_vals = sum((pred[:, i] > 0.5).long() << i for i in range(8))
            acc = (pred_vals == results).float().mean().item() * 100
            best_acc = max(best_acc, acc)
        
        if acc == 100.0:
            print(f"  {atom_name}: 100% at epoch {epoch+1}")
            break
    
    if best_acc < 100:
        print(f"  {atom_name}: {best_acc:.1f}% (best)")
    
    return atom, best_acc


class AtomicRouter(nn.Module):
    """Routes to atoms and composes them."""
    
    def __init__(self, atoms: dict, d_model=64):
        super().__init__()
        self.atoms = nn.ModuleDict(atoms)
        self.atom_names = list(atoms.keys())
        
        # Freeze atoms
        for atom in self.atoms.values():
            for p in atom.parameters():
                p.requires_grad = False
        
        # Router: given opcode + state, select atom sequence
        self.opcode_embed = nn.Embedding(8, 32)
        self.state_embed = nn.Embedding(2, 16)  # carry state
        
        # Output: which atom(s) to use and whether to chain
        self.router = nn.Sequential(
            nn.Linear(48, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.atom_names) + 1),  # +1 for "chain INC"
        )
    
    def forward(self, opcode, a_bits, b_bits, carry):
        """
        Route to atoms based on opcode and carry state.
        
        For ADC with carry=1: route to ADD, then INC
        """
        B = opcode.shape[0]
        device = opcode.device
        
        # Router decision
        op_emb = self.opcode_embed(opcode)
        state_emb = self.state_embed(carry)
        router_input = torch.cat([op_emb, state_emb], dim=-1)
        router_logits = self.router(router_input)
        
        # Soft routing for training, hard for inference
        atom_weights = F.softmax(router_logits[:, :-1], dim=-1)
        chain_prob = torch.sigmoid(router_logits[:, -1])
        
        # Compute each atom's output
        atom_outputs = {}
        for name, atom in self.atoms.items():
            needs_b = name in ['ADD', 'AND', 'ORA', 'EOR']
            if needs_b:
                atom_outputs[name] = atom(a_bits, b_bits)
            else:
                atom_outputs[name] = atom(a_bits)
        
        # Weighted combination of atom outputs
        output = torch.zeros(B, 8, device=device)
        for i, name in enumerate(self.atom_names):
            output = output + atom_weights[:, i:i+1] * atom_outputs[name]
        
        # Chain with INC if needed (for ADC with carry)
        inc_input_bits = (output > 0.5).float()
        inc_output = self.atoms['INC'](inc_input_bits)
        
        # Blend based on chain probability
        final_output = (1 - chain_prob.unsqueeze(-1)) * output + chain_prob.unsqueeze(-1) * inc_output
        
        return final_output, atom_weights, chain_prob


def train_router(router, epochs=50, device='cuda'):
    """Train router to compose atoms correctly."""
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, router.parameters()), lr=0.002)
    
    # Generate 6502 ADC data (the hard case)
    data = []
    for _ in range(10000):
        op = 0  # ADC
        a = np.random.randint(0, 256)
        b = np.random.randint(0, 256)
        c = np.random.randint(0, 2)
        result = (a + b + c) & 0xFF
        data.append((op, a, b, c, result))
    
    # Add other ops for balance
    for op_idx, op_name in enumerate(['ADD', 'AND', 'ORA', 'EOR', 'ASL', 'LSR', 'INC', 'DEC']):
        for _ in range(1000):
            a = np.random.randint(0, 256)
            b = np.random.randint(0, 256) if op_name in ['ADD', 'AND', 'ORA', 'EOR'] else 0
            result = compute_atom(op_name, a, b)
            data.append((op_idx, a, b, 0, result))
    
    np.random.shuffle(data)
    
    opcodes = torch.tensor([d[0] for d in data], device=device)
    a_vals = torch.tensor([d[1] for d in data], device=device)
    b_vals = torch.tensor([d[2] for d in data], device=device)
    carries = torch.tensor([d[3] for d in data], device=device)
    results = torch.tensor([d[4] for d in data], device=device)
    
    a_bits = torch.stack([(a_vals >> i) & 1 for i in range(8)], dim=1).float()
    b_bits = torch.stack([(b_vals >> i) & 1 for i in range(8)], dim=1).float()
    result_bits = torch.stack([(results >> i) & 1 for i in range(8)], dim=1).float()
    
    batch_size = 256
    
    print("\nTraining router...")
    for epoch in range(epochs):
        router.train()
        perm = torch.randperm(len(data), device=device)
        total_loss = 0
        
        for i in range(0, len(data) - batch_size, batch_size):
            idx = perm[i:i+batch_size]
            
            pred, atom_weights, chain_prob = router(
                opcodes[idx], a_bits[idx], b_bits[idx], carries[idx]
            )
            
            loss = F.binary_cross_entropy(pred, result_bits[idx])
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            # Check ADC accuracy
            router.eval()
            with torch.no_grad():
                adc_mask = (opcodes == 0)
                pred, _, chain = router(opcodes[adc_mask], a_bits[adc_mask], b_bits[adc_mask], carries[adc_mask])
                pred_vals = sum((pred[:, i] > 0.5).long() << i for i in range(8))
                adc_acc = (pred_vals == results[adc_mask]).float().mean().item() * 100
                
                # Check chain usage
                c1_mask = adc_mask & (carries == 1)
                c0_mask = adc_mask & (carries == 0)
                chain_c1 = chain[carries[adc_mask] == 1].mean().item() if (carries[adc_mask] == 1).any() else 0
                chain_c0 = chain[carries[adc_mask] == 0].mean().item() if (carries[adc_mask] == 0).any() else 0
                
            print(f"  Epoch {epoch+1}: loss={total_loss/(len(data)//batch_size):.4f}, "
                  f"ADC={adc_acc:.1f}%, chain[C=1]={chain_c1:.2f}, chain[C=0]={chain_c0:.2f}")


def verify_atomic(router, device='cuda'):
    """Verify atomic composition."""
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    router.eval()
    
    # Test ADC specifically
    results = {'ADC_C0': 0, 'ADC_C1': 0}
    totals = {'ADC_C0': 0, 'ADC_C1': 0}
    
    with torch.no_grad():
        for c in [0, 1]:
            for _ in range(500):
                a = np.random.randint(0, 256)
                b = np.random.randint(0, 256)
                expected = (a + b + c) & 0xFF
                
                opcode = torch.tensor([0], device=device)
                carry = torch.tensor([c], device=device)
                a_bits = torch.tensor([[(a >> i) & 1 for i in range(8)]], device=device).float()
                b_bits = torch.tensor([[(b >> i) & 1 for i in range(8)]], device=device).float()
                
                pred, weights, chain = router(opcode, a_bits, b_bits, carry)
                pred_val = sum((pred[0, i] > 0.5).long().item() << i for i in range(8))
                
                key = f'ADC_C{c}'
                totals[key] += 1
                if pred_val == expected:
                    results[key] += 1
    
    for key in ['ADC_C0', 'ADC_C1']:
        acc = results[key] / totals[key] * 100
        bar = '█' * int(acc / 5)
        print(f"  {key}: {bar:20s} {acc:.1f}%")
    
    overall = (results['ADC_C0'] + results['ADC_C1']) / (totals['ADC_C0'] + totals['ADC_C1']) * 100
    print(f"\n  ADC Overall: {overall:.1f}%")
    
    return overall


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("TriX 6502: ATOMIC COMPOSITION")
    print("=" * 60)
    print(f"Device: {device}")
    print("\nPhilosophy: Atoms are perfect. Routing composes them.")
    print("ADC_C1 = ADD → INC (not learned, composed)")
    
    # Step 1: Train each atom to 100%
    print("\n" + "-" * 60)
    print("TRAINING ATOMS (each to 100%)")
    print("-" * 60)
    
    atoms = {}
    atom_configs = {
        'ADD': True,   # needs_b
        'AND': True,
        'ORA': True,
        'EOR': True,
        'ASL': False,
        'LSR': False,
        'INC': False,
        'DEC': False,
    }
    
    for name, needs_b in atom_configs.items():
        atom, acc = train_atom(name, needs_b, epochs=100, device=device)
        atoms[name] = atom
    
    # Step 2: Build router
    print("\n" + "-" * 60)
    print("TRAINING ROUTER (composition)")
    print("-" * 60)
    
    router = AtomicRouter(atoms).to(device)
    train_router(router, epochs=50, device=device)
    
    # Step 3: Verify
    acc = verify_atomic(router, device)
    
    print("\n" + "=" * 60)
    if acc > 95:
        print(f"SUCCESS: ADC {acc:.1f}% via atomic composition")
    else:
        print(f"PROGRESS: ADC {acc:.1f}% (needs tuning)")
    print("=" * 60)


if __name__ == "__main__":
    main()
