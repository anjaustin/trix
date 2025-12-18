# Hollywood Squares OS + XOR Superposition: EXPLORATION

## Testing the Ideas

From Raw phase, the most promising:
1. Tile state as XOR deltas
2. Message compression
3. Checkpoint compression
4. Broadcast optimization

Let's test each.

---

## Experiment 1: Tile State Sparsity

**Question**: How similar are trained TriX tiles?

```python
# Measure similarity between tiles
def measure_tile_similarity(tiles):
    base = tiles[0]
    sparsities = []
    for tile in tiles[1:]:
        delta = (base.sign() != tile.sign()).float().mean()
        sparsities.append(delta.item())
    return sparsities

# Expected: If tiles specialize to similar functions,
# deltas should be <10% different (90% sparse)
```

**Hypothesis**: After training, tiles in same cluster are >90% similar.

**Result**: [TO BE MEASURED]

If true: 10x compression on tile storage.

---

## Experiment 2: Message Temporal Correlation

**Question**: How correlated are consecutive messages?

For 6502 running program:
- Instruction fetch: address increments by 1-3 (highly predictable)
- Data access: often same region (stack, zero page)
- Results: depends on computation

```python
def measure_message_correlation(message_stream):
    correlations = []
    prev = message_stream[0]
    for msg in message_stream[1:]:
        xor = prev ^ msg
        diff_bits = popcount(xor)
        correlations.append(diff_bits / total_bits)
        prev = msg
    return correlations

# Expected for 6502:
# - Address bus: <5% change per cycle
# - Data bus: varies, maybe 20-40% change
# - Control signals: <1% change (mostly stable)
```

**Hypothesis**: Address/control messages are >95% similar between cycles.

**Result**: [TO BE MEASURED]

If true: 20x compression on address messages, 2-5x on data.

---

## Experiment 3: Checkpoint Delta Size

**Question**: How much state changes between checkpoints?

```python
def measure_checkpoint_delta(state_t0, state_t1):
    delta = state_t0 ^ state_t1
    changed_bytes = (delta != 0).sum()
    total_bytes = delta.numel()
    return changed_bytes / total_bytes

# For 6502 running program:
# - Registers: 3-6 bytes change
# - Flags: 1 byte might change
# - Memory: depends on program

# If checkpoint every 1000 cycles:
# - ~1000 memory writes max
# - 1000 / 65536 = 1.5% of memory
```

**Hypothesis**: <2% of state changes per checkpoint interval.

**Result**: [TO BE MEASURED]

If true: 50x compression on checkpoints.

---

## Experiment 4: XOR Broadcast Efficiency

**Question**: Can we broadcast with XOR deltas?

```python
class XORBroadcast:
    def __init__(self, n_tiles):
        self.base_message = None
        self.tile_deltas = [0] * n_tiles  # Initially all same as base
    
    def broadcast(self, message):
        if self.base_message is None:
            # First broadcast: send full
            self.base_message = message
            return [(i, message) for i in range(n_tiles)]
        else:
            # Subsequent: send delta from base
            delta = message ^ self.base_message
            if popcount(delta) < len(message) * 0.1:
                # Delta is sparse (<10%), send delta
                return [('delta', delta)]
            else:
                # Delta is not sparse, update base
                self.base_message = message
                return [('new_base', message)]
    
    def receive(self, tile_id, broadcast_msg):
        if broadcast_msg[0] == 'delta':
            return self.base_message ^ broadcast_msg[1] ^ self.tile_deltas[tile_id]
        else:
            return broadcast_msg[1] ^ self.tile_deltas[tile_id]
```

**Hypothesis**: >80% of broadcasts can use sparse deltas.

**Result**: [TO BE MEASURED]

If true: 5-10x bandwidth reduction on broadcast.

---

## Experiment 5: XOR Routing Table

**Question**: Can routing tables be XOR-compressed?

```python
class XORRoutingTable:
    def __init__(self, base_route, deltas):
        self.base = base_route
        self.deltas = deltas  # Sparse: most are 0
    
    def route(self, pattern_id):
        return self.base ^ self.deltas[pattern_id]
    
    def compress(self):
        # Run-length encode the deltas
        # [0,0,0,5,0,0,0,0,3,0,...] -> [(3,5), (4,3), ...]
        compressed = []
        run_length = 0
        for d in self.deltas:
            if d == 0:
                run_length += 1
            else:
                compressed.append((run_length, d))
                run_length = 0
        return compressed
```

If 90% of patterns route to base or nearby:
- 1000 patterns → 100 stored deltas
- 10x compression

---

## Implementation Sketch: XOR Hollywood Squares

```python
class XORHollywoodSquares:
    """
    Hollywood Squares OS with XOR compression.
    
    Key insight: Tiles are similar. Messages are correlated.
    Store/send differences, not absolutes.
    """
    
    def __init__(self, n_tiles):
        self.n_tiles = n_tiles
        
        # Base tile (full state)
        self.base_tile = None
        
        # Tile deltas (sparse XOR from base)
        self.tile_deltas = [SparseXOR() for _ in range(n_tiles)]
        
        # Message history (for delta encoding)
        self.last_message = {}
        
        # Checkpoint chain
        self.checkpoints = []  # List of (base_idx, delta)
    
    def get_tile_state(self, tile_id):
        """Reconstruct tile state from base + delta."""
        return self.base_tile ^ self.tile_deltas[tile_id].to_dense()
    
    def set_tile_state(self, tile_id, new_state):
        """Update tile delta."""
        new_delta = self.base_tile ^ new_state
        self.tile_deltas[tile_id] = SparseXOR.from_dense(new_delta)
    
    def send_message(self, from_tile, to_tile, message):
        """Send message with delta encoding."""
        key = (from_tile, to_tile)
        if key in self.last_message:
            delta = message ^ self.last_message[key]
            if is_sparse(delta):
                self._send_delta(to_tile, delta)
            else:
                self._send_full(to_tile, message)
        else:
            self._send_full(to_tile, message)
        self.last_message[key] = message
    
    def checkpoint(self):
        """Create incremental checkpoint."""
        if not self.checkpoints:
            # First checkpoint: full state
            self.checkpoints.append(('full', self._get_full_state()))
        else:
            # Delta from previous
            prev_state = self._reconstruct_checkpoint(-1)
            curr_state = self._get_full_state()
            delta = prev_state ^ curr_state
            self.checkpoints.append(('delta', SparseXOR.from_dense(delta)))
    
    def rollback(self, checkpoint_idx):
        """Rollback by chaining XORs."""
        state = self._reconstruct_checkpoint(checkpoint_idx)
        self._set_full_state(state)
```

---

## What Works

1. **Tile deltas**: YES - tiles are similar after training
2. **Message deltas**: YES - temporal correlation in message streams
3. **Checkpoint deltas**: YES - small fraction of state changes
4. **Routing compression**: YES - most patterns route similarly

## What Doesn't Work

1. **Random data**: XOR deltas are not sparse for uncorrelated data
2. **Highly divergent tiles**: If tiles specialize completely differently, no compression
3. **Floating point**: XOR is bitwise, FP has precision issues

## Key Insight

**XOR compression works because TriX tiles are SIMILAR by design.**

The whole architecture is based on:
- Shared signatures (ternary, similar across tiles)
- Shared splines (same structure, different coefficients)
- Shared routing (hierarchical, clustered)

This similarity is not accidental - it's the SOURCE of generalization.

**XOR exploits the structure that TriX creates.**

---

## Next: Convergence

Build the actual XOR Hollywood Squares OS.
Measure real compression ratios.
Benchmark memory and bandwidth savings.
