# Demo Guide

**"From Spec to Swarm" — The Flagship Demonstration**

---

## The Core Superpower

> **You can train computation, freeze it, prove it correct, and then compose it live while watching every step.**

Most systems can do ONE of these. Almost none can do all four at once.

---

## Demo #1: Learn → Freeze → Verify → Compose (Flagship)

**Duration:** 10 minutes
**Audience:** Systems people, ML people, investors

### Setup

```python
from trix.hsquares_os import HSquaresOS, SquaresShell
from trix.spline6502 import SplineTrainer, SplineVerifier

os = HSquaresOS(num_workers=8)
os.boot()
shell = SquaresShell(os)
```

### Act 1: Training (2 min)

```python
# Define a weird operation: saturating add
def saturating_add(a, b):
    return min(a + b, 255)

# Generate training data
data = [(a, b, saturating_add(a, b)) 
        for a in range(256) for b in range(256)]

# Train
trainer = SplineTrainer()
model = trainer.train(data)
print(f"Training loss: {trainer.final_loss:.6f}")
```

**Say:** *"This is learning. But we won't trust it yet."*

### Act 2: Freeze (1 min)

```python
# Freeze to lookup table
frozen = model.freeze()
print(f"Size: {frozen.size_bytes} bytes")  # ~3KB
```

**Say:** *"This is now a deterministic artifact. Same input, same output, forever."*

### Act 3: Verify (2 min)

```python
# Exhaustive verification
verifier = SplineVerifier()
result = verifier.verify(frozen, saturating_add)

print(f"Tested: {result.total_cases:,} cases")
print(f"Accuracy: {result.accuracy * 100:.1f}%")
print(f"Mismatches: {result.mismatches}")
```

**Say:** *"100% accuracy on 65,536 test cases. This is now as trustworthy as hand-written logic."*

### Act 4: Compose (3 min)

```python
# Inject into Hollywood Squares
os.set_neural_processor(node=1, processor=frozen)

# Execute through the network
print(shell.execute('nodes'))
```

**Say:** *"No retraining. No recompiling. The learned primitive is now part of the distributed system."*

### Act 5: Single-Step (2 min)

```python
# Queue a computation
from trix.hsquares_os.message import compute_msg
msg = compute_msg(0, 1, 99, op=0x10, a=200, b=100)
os.master.send_message(msg)

# Step through
print("Step 1:")
print(shell.execute('step'))

print("Step 2:")
print(shell.execute('step'))

print("Step 3:")
print(shell.execute('step'))

# Result should be 255 (saturated)
```

**Say:** *"You're watching a learned computation execute across a distributed system, one message at a time. Every step is deterministic and inspectable."*

---

## Demo #2: Deterministic Replay

**Duration:** 5 minutes
**Audience:** Systems people, debugging enthusiasts

### The Demo

```python
# Start recording
os.start_recording()

# Run some computations
os.exec(1, OpCode.ADD, 50, 10)
os.exec(2, OpCode.SUB, 100, 30)
os.exec(3, OpCode.ADD, 200, 55)

# Take snapshot
snap1 = os.snapshot()
print("Snapshot 1:", snap1)

# Stop recording
log = os.stop_recording()
print(f"Recorded {len(log)} messages")

# Reset and replay
os.replay(log)

# Take second snapshot
snap2 = os.snapshot()
print("Snapshot 2:", snap2)

# Verify identical
assert snap1['workers'] == snap2['workers']
print("✓ Bit-for-bit identical!")
```

**Say:** *"This is what happens when learning is bounded, frozen, and composed correctly. Complete deterministic replay across a distributed system."*

---

## Demo #3: The Shell Experience

**Duration:** 5 minutes
**Audience:** Enthusiasts, developers

### Interactive Session

```
$ python -m trix.hsquares_os.shell

Hollywood Squares OS - sqsh
Type 'help' for commands, 'exit' to quit

Booting...
8/8 workers online

> nodes
● node0  master   IDLE     msgs:8
● node1  worker   IDLE     msgs:1
● node2  worker   IDLE     msgs:1
...

> topo
root (master)
 ├─ ● node1
 ├─ ● node2
 ├─ ● node3
 ...
 └─ ● node8

> run all add 50 10
[60, 60, 60, 60, 60, 60, 60, 60]

> route add 100 55
→ node2: 155

> step
[  15] 1 msgs | active: n1

> inspect 1
node_id: 1
status: IDLE
tick: 15
msgs_received: 2
msgs_sent: 2

> trace show
[     1] Node 0: SEND (PING #1)
[     2] Node 1: RECV (PING #1)
[     2] Node 1: SEND (PONG #1)
...
```

**Say:** *"Feels like a shell. But instead of processes and files, it's nodes and messages. And you can see everything."*

---

## Demo #4: Scale Preview

**Duration:** 3 minutes
**Audience:** Business, investors

### The Numbers

```python
# What fits on Grace Blackwell
processors = 200_000
size_each = 3  # KB
total_size = processors * size_each / 1024  # MB

print(f"200,000 processors × 3KB = {total_size:.0f}MB")
print(f"Grace Blackwell has: hundreds of GB")
print(f"Headroom: massive")
print()
print("200,000 complex cognitive acts per cycle")
print("Each one: verified, deterministic, addressable")
print()
print("That's not AI that might be right.")
print("That's cognitive fabric that is PROVEN right.")
```

**Say:** *"This isn't incremental improvement. This is a different kind of machine."*

---

## Demo Anti-Patterns (Avoid These)

### Don't Say:
- "It beats GPUs"
- "It's smarter than GPT"
- "It scales to millions"
- "It replaces Unix"

### Do Say:
- "It's deterministic and verified"
- "You can single-step through it"
- "Learning produces artifacts, not behavior"
- "It's inspectable end-to-end"

---

## Audience-Specific Angles

### For Systems Reviewers
- Emphasize: Message passing, determinism, replay
- Demo: Trace a message through the network
- Key sentence: *"Distributed microkernel with message-passing syscalls"*

### For ML Reviewers
- Emphasize: Train → Freeze → Verify pipeline
- Demo: Exhaustive verification of learned function
- Key sentence: *"Learning as manufacturing, not inference"*

### For Network Silicon Contacts
- Emphasize: Packet classification, line rate, audit trail
- Demo: TCP classifier with 100% accuracy
- Key sentence: *"Semantic micro-plane for network devices"*

### For Investors
- Emphasize: First-mover in verified learned compute
- Demo: The full pipeline in 10 minutes
- Key sentence: *"Deterministic AI for safety-critical systems"*

### For Enthusiasts
- Emphasize: The shell, single-stepping, topology
- Demo: Interactive session with `step` and `inspect`
- Key sentence: *"A machine you can watch thinking"*

---

## The One-Liner

> **"We built a distributed system where learned computation can be frozen, proven correct, and composed live — and you can single-step it across a network."**

---

## Checklist Before Demo

- [ ] System boots with 8/8 workers
- [ ] Basic operations work (add, sub)
- [ ] Single-step shows meaningful output
- [ ] Trace shows message flow
- [ ] Snapshot works
- [ ] Neural processor injection works (if showing)
- [ ] Shell is responsive

---

## Recovery Moves

**If boot fails:**
```python
os = HSquaresOS(num_workers=4)  # Try fewer workers
```

**If node times out:**
```python
os.reset(node)
```

**If shell hangs:**
```python
shell.execute('stats')  # Check system state
```

---

## The Closer

End every demo with:

> *"This is what's possible when learning produces artifacts, not behavior. Train it. Freeze it. Prove it. Compose it. Step through it. That's the whole thing."*
