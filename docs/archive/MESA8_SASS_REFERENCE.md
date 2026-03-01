# Mesa 8: SASS Opcode Reference

**NVIDIA Shader Assembly for TriX Neural CUDA**

---

## Overview

SASS (Shader Assembly) is NVIDIA's low-level GPU instruction set. Mesa 8 maps SASS opcodes to TriX tiles for neural execution.

**Source:** nvdisasm output from Jetson AGX Thor (sm_90, CUDA 13.0)

---

## Opcode Categories

### INTEGER_ALU (Tile 0 - Flynn Core)

Integer arithmetic and logic operations.

| Opcode | Description | TriX Implementation |
|--------|-------------|---------------------|
| `IADD3` | 3-input integer add | RippleAdderTile (SUM + CARRY atoms) |
| `IADD` | 2-input integer add | RippleAdderTile |
| `IMAD` | Integer multiply-add | MUL tile + ADD tile |
| `IMUL` | Integer multiply | Composition of adders |
| `LOP3` | 3-input logic operation | Programmable LUT tile |
| `SHF` | Funnel shift | Monarch permutation router |
| `SHL` | Shift left | Routing-based shift |
| `SHR` | Shift right | Routing-based shift |
| `BFE` | Bit field extract | Routing + masking |
| `BFI` | Bit field insert | Routing + masking |
| `FLO` | Find leading one | Tree reduction |
| `POPC` | Population count | Tree reduction |

#### IADD3 Detail

```assembly
IADD3 Rd, Ra, Rb, Rc
```

**Semantics:** `Rd = Ra + Rb + Rc`

**TriX Execution:**
1. Route to Tile 0 via signature `[1,1,0,0,0,0,0,0,-1,-1,0,0,0,0,0,0]`
2. Execute on RippleAdderTile
3. Ripple through 32 FullAdderTiles
4. Each FullAdder uses SUM + CARRY FP4 atoms

**Example:**
```
IADD3 R9, R2, R5, RZ
R2 = 42, R5 = 58, RZ = 0
→ R9 = 42 + 58 + 0 = 100
```

---

### MEMORY (Tile 1)

Memory access operations.

| Opcode | Description | TriX Implementation |
|--------|-------------|---------------------|
| `LDG` | Load global memory | Memory tile (address routing) |
| `STG` | Store global memory | Memory tile (address routing) |
| `LDC` | Load constant memory | Constant bank lookup |
| `ULDC` | Load constant to uniform | Uniform register write |
| `LDS` | Load shared memory | Shared memory tile |
| `STS` | Store shared memory | Shared memory tile |
| `ATOM` | Atomic operation | Atomic tile |
| `RED` | Reduction operation | Reduction tile |

#### LDG Detail

```assembly
LDG.E Rd, desc[URx][Ry.64]
```

**Semantics:** Load 32-bit value from global memory at address in Ry

**TriX Execution:**
1. Route to Tile 1 via signature
2. Extract address from register Ry
3. Read from memory buffer
4. Write to destination Rd

---

### CONTROL (Tile 2)

Control flow operations.

| Opcode | Description | TriX Implementation |
|--------|-------------|---------------------|
| `EXIT` | Exit kernel | Set exit flag |
| `BRA` | Branch | Update PC |
| `RET` | Return | Pop call stack |
| `CALL` | Call function | Push PC, jump |
| `NOP` | No operation | Skip |
| `BAR` | Barrier synchronization | Sync tile |
| `YIELD` | Yield execution | Scheduler hint |

---

### PREDICATE (Predicate Tile)

Comparison and predicate operations.

| Opcode | Description | TriX Implementation |
|--------|-------------|---------------------|
| `ISETP` | Integer set predicate | Comparator atoms |
| `FSETP` | Float set predicate | FP comparator |
| `ICMP` | Integer compare | Comparator atoms |
| `PSETP` | Predicate set predicate | Logic atoms |
| `P2R` | Predicate to register | Routing |
| `R2P` | Register to predicate | Routing |

#### ISETP Detail

```assembly
ISETP.GE.AND P0, PT, R9, UR4, PT
```

**Semantics:** Set P0 = (R9 >= UR4) AND PT

**Comparisons:** `.EQ`, `.NE`, `.LT`, `.LE`, `.GT`, `.GE`

---

### SPECIAL (Tile 3 - Twiddle Core)

Special function unit (transcendentals).

| Opcode | Description | TriX Implementation |
|--------|-------------|---------------------|
| `MUFU.SIN` | Sine | Twiddle table lookup |
| `MUFU.COS` | Cosine | Twiddle table lookup (phase shifted) |
| `MUFU.RCP` | Reciprocal | Reciprocal table lookup |
| `MUFU.RSQ` | Reciprocal square root | RSQ table lookup |
| `MUFU.LOG2` | Log base 2 | Log table lookup |
| `MUFU.EX2` | 2^x | Exp table lookup |

**Key Insight:** These map directly to Mesa 5 twiddle opcodes!

```python
# Mesa 5 twiddle table
TWIDDLE_OPS = {
    'W0': (1.0, 0.0),           # cos(0), sin(0)
    'W1': (0.707, -0.707),      # cos(π/4), sin(π/4)
    'W2': (0.0, -1.0),          # cos(π/2), sin(π/2)
    ...
}
```

**0.00 Error:** Routing to exact opcodes, no approximation.

---

### TENSOR (Tile 4 - Butterfly Core)

Matrix operations.

| Opcode | Description | TriX Implementation |
|--------|-------------|---------------------|
| `HMMA` | Half-precision matrix multiply | Butterfly/Monarch tile |
| `IMMA` | Integer matrix multiply | Butterfly/Monarch tile |
| `DMMA` | Double-precision matrix multiply | Butterfly/Monarch tile |

**Key Insight:** These map to Mesa 6 butterfly structures!

```python
# Monarch matrix decomposition
M = P1 @ B1 @ P2 @ B2

# Same structure as FFT
# O(N√N) instead of O(N²)
```

---

### SYSTEM (System Tile)

System and data movement.

| Opcode | Description | TriX Implementation |
|--------|-------------|---------------------|
| `S2R` | System register to register | Thread ID lookup |
| `S2UR` | System to uniform register | Block ID lookup |
| `CS2R` | Constant system to register | Constant lookup |
| `MOV` | Move | Register routing |
| `SEL` | Select | MUX atom |
| `SHFL` | Warp shuffle | Permutation routing |

#### S2R Detail

```assembly
S2R R0, SR_TID.X
```

**Semantics:** Read thread ID (X dimension) into R0

**System Registers:**
- `SR_TID.X/Y/Z` - Thread ID
- `SR_CTAID.X/Y/Z` - Block ID
- `SR_NTID.X/Y/Z` - Block dimensions
- `SR_NCTAID.X/Y/Z` - Grid dimensions

---

## Register Classes

| Class | Range | Purpose |
|-------|-------|---------|
| `R0-R255` | GPR | General purpose (per-thread) |
| `UR0-UR63` | Uniform | Shared across warp |
| `P0-P6` | Predicate | Boolean conditions |
| `PT` | Predicate True | Always 1 |
| `RZ` | Zero | Always 0 |

---

## Instruction Format

### Standard Format

```
[predicate] OPCODE[.modifiers] dest, src1, src2, src3 ;
```

### Examples

```assembly
        IADD3 R9, R2, R5, RZ ;           # R9 = R2 + R5 + 0
        IMAD R9, R9, UR4, R0 ;           # R9 = R9 * UR4 + R0
   @P0  EXIT ;                            # Exit if P0 is true
        LDG.E R2, desc[UR4][R2.64] ;     # Load from global memory
        STG.E desc[UR4][R6.64], R9 ;     # Store to global memory
```

### Modifiers

| Modifier | Meaning |
|----------|---------|
| `.E` | Extended addressing |
| `.64` | 64-bit operation |
| `.WIDE` | Wide result |
| `.GE`, `.LT`, etc. | Comparison type |
| `.AND`, `.OR` | Predicate combiner |

---

## Sample Kernel Analysis

### Simple Add: `*c = *a + *b`

```assembly
_Z10add_kernelPiS_S_:
    LDC R1, c[0x0][0x28] ;           # Load stack pointer
    LDC.64 R2, c[0x0][0x210] ;       # Load &a
    ULDC.64 UR4, c[0x0][0x208] ;     # Load descriptor
    LDC.64 R4, c[0x0][0x218] ;       # Load &b
    LDG.E R2, desc[UR4][R2.64] ;     # R2 = *a
    LDG.E R5, desc[UR4][R4.64] ;     # R5 = *b
    LDC.64 R6, c[0x0][0x220] ;       # Load &c
    IADD3 R9, R2, R5, RZ ;           # R9 = *a + *b
    STG.E desc[UR4][R6.64], R9 ;     # *c = R9
    EXIT ;
```

**TriX Execution:**
1. LDC/ULDC → Tile 1 (MEMORY)
2. LDG → Tile 1 (MEMORY)
3. IADD3 → Tile 0 (INTEGER_ALU) → FP4 atoms
4. STG → Tile 1 (MEMORY)
5. EXIT → Tile 2 (CONTROL)

---

## Implementation Status

| Opcode | Category | TriX Status |
|--------|----------|-------------|
| IADD3 | INTEGER_ALU | ✅ Implemented (FP4 atoms) |
| IMAD | INTEGER_ALU | 🔶 Next |
| LOP3 | INTEGER_ALU | 🔶 Next |
| LDG | MEMORY | ✅ Implemented (mocked) |
| STG | MEMORY | ✅ Implemented (mocked) |
| LDC | MEMORY | ✅ Implemented (mocked) |
| EXIT | CONTROL | ✅ Implemented |
| MUFU | SPECIAL | ✅ Via Mesa 5 twiddles |
| HMMA | TENSOR | ✅ Via Mesa 6 butterfly |
| ISETP | PREDICATE | 🔶 Next |

---

## Tools

### nvdisasm

Disassemble CUDA binaries:

```bash
nvcc -cubin -arch=sm_90 kernel.cu -o kernel.cubin
nvdisasm -g kernel.cubin > kernel.sass
```

### cuobjdump

List binary contents:

```bash
cuobjdump -sass kernel.cubin
```

---

*SASS opcodes. TriX tiles. Neural GPU execution.*
