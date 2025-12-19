#!/usr/bin/env python3
"""
TriX 6502: ATOMIC with EXACT atoms

Atoms are NOT learned. They ARE the operations.
Routing composes them.

ADC_C1 = ADD → INC
"""

import sys
sys.path.insert(0, '/workspace/trix_latest/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# EXACT atomic operations - not learned, perfect by definition
class ExactADD(nn.Module):
    """A + B, no carry"""
    def forward(self, a, b):
        return (a + b) % 256

class ExactINC(nn.Module):
    """A + 1"""
    def forward(self, a):
        return (a + 1) % 256

class ExactAND(nn.Module):
    def forward(self, a, b):
        return a & b

class ExactORA(nn.Module):
    def forward(self, a, b):
        return a | b

class ExactEOR(nn.Module):
    def forward(self, a, b):
        return a ^ b

class ExactASL(nn.Module):
    def forward(self, a):
        return (a << 1) & 0xFF

class ExactLSR(nn.Module):
    def forward(self, a):
        return a >> 1

class ExactDEC(nn.Module):
    def forward(self, a):
        return (a - 1) % 256


class AtomicComposer(nn.Module):
    """
    Routes inputs to exact atoms and composes them.
    
    The router is learned. The atoms are exact.
    
    For ADC:
      C=0: route to ADD
      C=1: route to ADD, then INC
    """
    
    def __init__(self):
        super().__init__()
        
        # Exact atoms (frozen, perfect)
        self.add = ExactADD()
        self.inc = ExactINC()
        self.and_op = ExactAND()
        self.ora = ExactORA()
        self.eor = ExactEOR()
        self.asl = ExactASL()
        self.lsr = ExactLSR()
        self.dec = ExactDEC()
        
        # Router: opcode + carry -> composition path
        # 8 opcodes, 2 carry states = 16 possible paths
        self.router = nn.Sequential(
            nn.Linear(10, 32),  # 8 opcode one-hot + 2 (a_high, b_high hints)
            nn.ReLU(),
            nn.Linear(32, 2),  # [use_primary, chain_inc]
        )
    
    def forward(self, opcode, a, b, carry):
        """
        Compose atoms based on opcode and carry.
        
        The insight: For ADC, we just need to know:
        - Always do ADD
        - If carry=1, also do INC
        """
        batch_size = opcode.shape[0]
        device = opcode.device
        
        # Opcode one-hot
        op_onehot = F.one_hot(opcode, 8).float()
        
        # Simple hints about operand ranges
        a_high = (a > 127).float().unsqueeze(1)
        b_high = (b > 127).float().unsqueeze(1)
        
        router_input = torch.cat([op_onehot, a_high, b_high], dim=1)
        router_out = self.router(router_input)
        
        # For ADC (opcode 0), we want:
        #   chain_inc = carry (if carry is 1, chain INC)
        # The router should learn this
        
        chain_inc_prob = torch.sigmoid(router_out[:, 1])
        
        # Compute all atoms
        results = {
            'add': self.add(a, b),
            'and': self.and_op(a, b),
            'ora': self.ora(a, b),
            'eor': self.eor(a, b),
            'asl': self.asl(a),
            'lsr': self.lsr(a),
            'inc': self.inc(a),
            'dec': self.dec(a),
        }
        
        # Select primary result based on opcode
        # 0=ADC->add, 1=AND->and, 2=ORA->ora, 3=EOR->eor, 
        # 4=ASL->asl, 5=LSR->lsr, 6=INC->inc, 7=DEC->dec
        primary = torch.zeros(batch_size, device=device)
        primary = torch.where(opcode == 0, results['add'].float(), primary)
        primary = torch.where(opcode == 1, results['and'].float(), primary)
        primary = torch.where(opcode == 2, results['ora'].float(), primary)
        primary = torch.where(opcode == 3, results['eor'].float(), primary)
        primary = torch.where(opcode == 4, results['asl'].float(), primary)
        primary = torch.where(opcode == 5, results['lsr'].float(), primary)
        primary = torch.where(opcode == 6, results['inc'].float(), primary)
        primary = torch.where(opcode == 7, results['dec'].float(), primary)
        
        # Chain INC if needed (for ADC with carry)
        inc_result = self.inc(primary.long()).float()
        
        # For ADC (opcode 0): blend based on chain probability
        # For others: just use primary
        is_adc = (opcode == 0).float()
        
        # Final result:
        # If ADC and router says chain: use inc_result
        # Otherwise: use primary
        final = primary * (1 - is_adc * chain_inc_prob) + inc_result * (is_adc * chain_inc_prob)
        
        return final.long(), chain_inc_prob


def train_composer(composer, epochs=100, device='cuda'):
    """Train the router to compose correctly."""
    composer = composer.to(device)
    opt = torch.optim.Adam(composer.router.parameters(), lr=0.01)
    
    print("Training router to compose atoms...")
    
    for epoch in range(epochs):
        # Generate ADC training data (the only thing router needs to learn)
        batch_size = 1024
        
        a = torch.randint(0, 256, (batch_size,), device=device)
        b = torch.randint(0, 256, (batch_size,), device=device)
        carry = torch.randint(0, 2, (batch_size,), device=device)
        opcode = torch.zeros(batch_size, dtype=torch.long, device=device)  # ADC
        
        # Ground truth: A + B + C
        target = ((a + b + carry) % 256).float()
        
        pred, chain_prob = composer(opcode, a, b, carry)
        
        # Loss: how close is prediction to target?
        loss = F.mse_loss(pred.float(), target)
        
        # Regularizer: chain_prob should match carry for ADC
        carry_loss = F.binary_cross_entropy(chain_prob, carry.float())
        
        total_loss = loss + carry_loss
        
        opt.zero_grad()
        total_loss.backward()
        opt.step()
        
        if (epoch + 1) % 20 == 0:
            # Check accuracy
            with torch.no_grad():
                acc = (pred == target.long()).float().mean().item() * 100
                chain_c1 = chain_prob[carry == 1].mean().item() if (carry == 1).any() else 0
                chain_c0 = chain_prob[carry == 0].mean().item() if (carry == 0).any() else 0
                
            print(f"  Epoch {epoch+1}: loss={total_loss.item():.4f}, acc={acc:.1f}%, "
                  f"chain[C=1]={chain_c1:.2f}, chain[C=0]={chain_c0:.2f}")


def verify_composer(composer, device='cuda'):
    """Exhaustive verification."""
    composer.eval()
    
    print("\n" + "=" * 60)
    print("EXHAUSTIVE VERIFICATION")
    print("=" * 60)
    
    errors = {'ADC_C0': 0, 'ADC_C1': 0, 'AND': 0, 'ORA': 0, 'EOR': 0, 'ASL': 0, 'LSR': 0, 'INC': 0, 'DEC': 0}
    totals = {'ADC_C0': 0, 'ADC_C1': 0, 'AND': 0, 'ORA': 0, 'EOR': 0, 'ASL': 0, 'LSR': 0, 'INC': 0, 'DEC': 0}
    
    with torch.no_grad():
        # Test ADC exhaustively (all 256*256*2 = 131072 cases)
        for c in [0, 1]:
            for a_val in range(256):
                a = torch.full((256,), a_val, device=device)
                b = torch.arange(256, device=device)
                carry = torch.full((256,), c, device=device)
                opcode = torch.zeros(256, dtype=torch.long, device=device)
                
                expected = ((a + b + carry) % 256)
                pred, _ = composer(opcode, a, b, carry)
                
                key = f'ADC_C{c}'
                totals[key] += 256
                errors[key] += (pred != expected).sum().item()
        
        # Test other ops (sample)
        ops = [('AND', 1, lambda a, b: a & b, True),
               ('ORA', 2, lambda a, b: a | b, True),
               ('EOR', 3, lambda a, b: a ^ b, True),
               ('ASL', 4, lambda a, _: (a << 1) & 0xFF, False),
               ('LSR', 5, lambda a, _: a >> 1, False),
               ('INC', 6, lambda a, _: (a + 1) % 256, False),
               ('DEC', 7, lambda a, _: (a - 1) % 256, False)]
        
        for name, op_idx, op_fn, needs_b in ops:
            for a_val in range(256):
                a = torch.full((256,), a_val, device=device)
                b = torch.arange(256, device=device) if needs_b else torch.zeros(256, device=device)
                carry = torch.zeros(256, device=device)
                opcode = torch.full((256,), op_idx, dtype=torch.long, device=device)
                
                expected = torch.tensor([op_fn(a_val, b_val.item()) for b_val in b], device=device)
                pred, _ = composer(opcode, a, b, carry)
                
                totals[name] += 256
                errors[name] += (pred != expected).sum().item()
    
    print("\nResults:")
    for key in ['ADC_C0', 'ADC_C1', 'AND', 'ORA', 'EOR', 'ASL', 'LSR', 'INC', 'DEC']:
        acc = (totals[key] - errors[key]) / totals[key] * 100
        bar = '█' * int(acc / 5)
        err_str = f"({errors[key]} errors)" if errors[key] > 0 else ""
        print(f"  {key:8s}: {bar:20s} {acc:.1f}% {err_str}")
    
    # Overall ADC
    adc_total = totals['ADC_C0'] + totals['ADC_C1']
    adc_errors = errors['ADC_C0'] + errors['ADC_C1']
    adc_acc = (adc_total - adc_errors) / adc_total * 100
    print(f"\n  ADC Overall: {adc_acc:.1f}% ({adc_errors} errors out of {adc_total})")
    
    return adc_errors == 0


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("TriX 6502: ATOMIC COMPOSITION with EXACT ATOMS")
    print("=" * 60)
    print(f"Device: {device}")
    print("\nPhilosophy:")
    print("  - Atoms are EXACT (not learned)")
    print("  - Router learns to COMPOSE")
    print("  - ADC_C1 = ADD → INC")
    
    composer = AtomicComposer()
    train_composer(composer, epochs=100, device=device)
    
    success = verify_composer(composer, device)
    
    print("\n" + "=" * 60)
    if success:
        print("SUCCESS: 100% accuracy via atomic composition!")
    else:
        print("Composition learned but not perfect - router needs tuning")
    print("=" * 60)


if __name__ == "__main__":
    main()
