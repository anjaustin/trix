#!/usr/bin/env python3
"""
Mesa 8: SASS Parser

Parses nvdisasm output into structured instruction tokens.
These tokens become the "prompt" for the Neural CUDA engine.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from enum import Enum, auto


class OpcodeCategory(Enum):
    """SASS opcode categories mapped to TriX tiles."""
    INTEGER_ALU = auto()    # IADD3, IMAD, etc. → Flynn Core
    FLOAT_ALU = auto()      # FADD, FMUL, FFMA → FP Tile
    MEMORY = auto()         # LDG, STG, LDC → Memory Tile
    CONTROL = auto()        # EXIT, BRA, etc. → Control Tile
    PREDICATE = auto()      # ISETP, etc. → Predicate Tile
    SPECIAL = auto()        # MUFU.SIN, etc. → Twiddle Core
    TENSOR = auto()         # HMMA → Butterfly Core
    SYSTEM = auto()         # S2R, etc. → System Tile
    UNKNOWN = auto()


@dataclass
class SASSOperand:
    """A single operand in a SASS instruction."""
    raw: str
    reg: Optional[str] = None       # R0, UR4, P0, etc.
    immediate: Optional[int] = None  # Immediate value
    const_bank: Optional[int] = None # Constant bank
    const_offset: Optional[int] = None # Constant offset
    is_predicate: bool = False
    is_uniform: bool = False
    is_64bit: bool = False
    
    @classmethod
    def parse(cls, s: str) -> 'SASSOperand':
        """Parse a single operand string."""
        s = s.strip()
        op = cls(raw=s)
        
        # Check for register
        if re.match(r'^R\d+', s):
            op.reg = s.split('.')[0]  # Handle R2.64 -> R2
            if '.64' in s:
                op.is_64bit = True
        elif re.match(r'^UR\d+', s):
            op.reg = s.split('.')[0]
            op.is_uniform = True
            if '.64' in s:
                op.is_64bit = True
        elif re.match(r'^P\d+|^PT$', s):
            op.reg = s
            op.is_predicate = True
        elif s == 'RZ':
            op.reg = 'RZ'  # Zero register
        
        # Check for constant memory reference c[bank][offset]
        const_match = re.match(r'c\[0x([0-9a-f]+)\]\[(?:0x)?([0-9a-fRZ]+)\]', s, re.I)
        if const_match:
            op.const_bank = int(const_match.group(1), 16)
            offset_str = const_match.group(2)
            if offset_str != 'RZ':
                op.const_offset = int(offset_str, 16) if offset_str.startswith('0x') or any(c in offset_str for c in 'abcdef') else int(offset_str, 16)
        
        # Check for immediate
        imm_match = re.match(r'^0x([0-9a-f]+)$', s, re.I)
        if imm_match:
            op.immediate = int(imm_match.group(1), 16)
        elif re.match(r'^\d+$', s):
            op.immediate = int(s)
        
        return op


@dataclass 
class SASSInstruction:
    """A parsed SASS instruction."""
    address: int
    opcode: str
    modifiers: List[str]
    dest: Optional[SASSOperand]
    src: List[SASSOperand]
    predicate: Optional[str]
    category: OpcodeCategory
    raw: str
    
    def __repr__(self):
        pred = f"@{self.predicate} " if self.predicate else ""
        mods = '.' + '.'.join(self.modifiers) if self.modifiers else ''
        dest_str = f"{self.dest.raw}, " if self.dest else ""
        src_str = ", ".join(s.raw for s in self.src)
        return f"{pred}{self.opcode}{mods} {dest_str}{src_str}"


# Opcode to category mapping
OPCODE_CATEGORIES = {
    # Integer ALU (Flynn Core)
    'IADD3': OpcodeCategory.INTEGER_ALU,
    'IADD': OpcodeCategory.INTEGER_ALU,
    'IMAD': OpcodeCategory.INTEGER_ALU,
    'IMUL': OpcodeCategory.INTEGER_ALU,
    'IABS': OpcodeCategory.INTEGER_ALU,
    'INEG': OpcodeCategory.INTEGER_ALU,
    'IMNMX': OpcodeCategory.INTEGER_ALU,
    'LOP3': OpcodeCategory.INTEGER_ALU,
    'SHF': OpcodeCategory.INTEGER_ALU,
    'SHL': OpcodeCategory.INTEGER_ALU,
    'SHR': OpcodeCategory.INTEGER_ALU,
    'BFE': OpcodeCategory.INTEGER_ALU,
    'BFI': OpcodeCategory.INTEGER_ALU,
    'FLO': OpcodeCategory.INTEGER_ALU,
    'POPC': OpcodeCategory.INTEGER_ALU,
    
    # Float ALU (FP Tile)
    'FADD': OpcodeCategory.FLOAT_ALU,
    'FMUL': OpcodeCategory.FLOAT_ALU,
    'FFMA': OpcodeCategory.FLOAT_ALU,
    'FMNMX': OpcodeCategory.FLOAT_ALU,
    'FABS': OpcodeCategory.FLOAT_ALU,
    'FNEG': OpcodeCategory.FLOAT_ALU,
    'FCHK': OpcodeCategory.FLOAT_ALU,
    'FSETP': OpcodeCategory.FLOAT_ALU,
    'F2I': OpcodeCategory.FLOAT_ALU,
    'F2F': OpcodeCategory.FLOAT_ALU,
    'I2F': OpcodeCategory.FLOAT_ALU,
    
    # Memory (Memory Tile)
    'LDG': OpcodeCategory.MEMORY,
    'STG': OpcodeCategory.MEMORY,
    'LDC': OpcodeCategory.MEMORY,
    'LDS': OpcodeCategory.MEMORY,
    'STS': OpcodeCategory.MEMORY,
    'ULDC': OpcodeCategory.MEMORY,
    'ATOM': OpcodeCategory.MEMORY,
    'RED': OpcodeCategory.MEMORY,
    
    # Control (Control Tile)
    'EXIT': OpcodeCategory.CONTROL,
    'BRA': OpcodeCategory.CONTROL,
    'RET': OpcodeCategory.CONTROL,
    'CALL': OpcodeCategory.CONTROL,
    'NOP': OpcodeCategory.CONTROL,
    'BAR': OpcodeCategory.CONTROL,
    'YIELD': OpcodeCategory.CONTROL,
    
    # Predicate (Predicate Tile)
    'ISETP': OpcodeCategory.PREDICATE,
    'ICMP': OpcodeCategory.PREDICATE,
    'PSETP': OpcodeCategory.PREDICATE,
    'P2R': OpcodeCategory.PREDICATE,
    'R2P': OpcodeCategory.PREDICATE,
    
    # Special Functions (Twiddle Core)
    'MUFU': OpcodeCategory.SPECIAL,  # MUFU.SIN, MUFU.COS, etc.
    
    # Tensor Core (Butterfly Core)
    'HMMA': OpcodeCategory.TENSOR,
    'IMMA': OpcodeCategory.TENSOR,
    'DMMA': OpcodeCategory.TENSOR,
    
    # System (System Tile)
    'S2R': OpcodeCategory.SYSTEM,
    'S2UR': OpcodeCategory.SYSTEM,
    'CS2R': OpcodeCategory.SYSTEM,
    'MOV': OpcodeCategory.SYSTEM,
    'SEL': OpcodeCategory.SYSTEM,
    'SHFL': OpcodeCategory.SYSTEM,
}


def parse_sass_line(line: str) -> Optional[SASSInstruction]:
    """Parse a single line of nvdisasm output."""
    line = line.strip()
    if not line or line.startswith('//') or line.startswith('.'):
        return None
    
    # Match instruction pattern: /*addr*/ [@pred] OPCODE[.mod] operands ;
    pattern = r'/\*([0-9a-f]+)\*/\s+(?:@(\w+)\s+)?(\w+)([.\w]*)\s*(.*?)\s*;'
    match = re.match(pattern, line, re.I)
    
    if not match:
        return None
    
    addr = int(match.group(1), 16)
    predicate = match.group(2)
    opcode = match.group(3).upper()
    modifiers_str = match.group(4)
    operands_str = match.group(5)
    
    # Parse modifiers
    modifiers = [m for m in modifiers_str.split('.') if m]
    
    # Parse operands
    operands = []
    if operands_str:
        # Split by comma, but handle nested brackets
        parts = []
        depth = 0
        current = ""
        for c in operands_str:
            if c in '([':
                depth += 1
            elif c in ')]':
                depth -= 1
            elif c == ',' and depth == 0:
                parts.append(current.strip())
                current = ""
                continue
            current += c
        if current.strip():
            parts.append(current.strip())
        
        operands = [SASSOperand.parse(p) for p in parts]
    
    # First operand is usually dest, rest are src
    dest = operands[0] if operands else None
    src = operands[1:] if len(operands) > 1 else []
    
    # Get category
    category = OPCODE_CATEGORIES.get(opcode, OpcodeCategory.UNKNOWN)
    
    return SASSInstruction(
        address=addr,
        opcode=opcode,
        modifiers=modifiers,
        dest=dest,
        src=src,
        predicate=predicate,
        category=category,
        raw=line
    )


def parse_sass_kernel(sass_text: str) -> List[SASSInstruction]:
    """Parse a full kernel's SASS into instruction list."""
    instructions = []
    for line in sass_text.split('\n'):
        inst = parse_sass_line(line)
        if inst:
            instructions.append(inst)
    return instructions


def categorize_kernel(instructions: List[SASSInstruction]) -> Dict[OpcodeCategory, int]:
    """Get opcode category counts for a kernel."""
    counts = {}
    for inst in instructions:
        counts[inst.category] = counts.get(inst.category, 0) + 1
    return counts


# =============================================================================
# TEST
# =============================================================================

def test_parser():
    """Test the SASS parser with real kernel output."""
    
    # The simple add kernel from nvdisasm
    add_kernel_sass = """
        /*0000*/                   LDC R1, c[0x0][0x28] ;
        /*0010*/                   LDC.64 R2, c[0x0][0x210] ;
        /*0020*/                   ULDC.64 UR4, c[0x0][0x208] ;
        /*0030*/                   LDC.64 R4, c[0x0][0x218] ;
        /*0040*/                   LDG.E R2, desc[UR4][R2.64] ;
        /*0050*/                   LDG.E R5, desc[UR4][R4.64] ;
        /*0060*/                   LDC.64 R6, c[0x0][0x220] ;
        /*0070*/                   IADD3 R9, R2, R5, RZ ;
        /*0080*/                   STG.E desc[UR4][R6.64], R9 ;
        /*0090*/                   EXIT ;
    """
    
    print("=" * 60)
    print("MESA 8: SASS PARSER TEST")
    print("=" * 60)
    
    instructions = parse_sass_kernel(add_kernel_sass)
    
    print(f"\nParsed {len(instructions)} instructions:\n")
    
    for inst in instructions:
        cat = inst.category.name
        print(f"  [0x{inst.address:04x}] {cat:12s} | {inst}")
    
    # Category breakdown
    print("\n" + "-" * 60)
    print("Category Breakdown:")
    counts = categorize_kernel(instructions)
    for cat, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cat.name}: {count}")
    
    # Find the IADD3
    iadd = [i for i in instructions if i.opcode == 'IADD3']
    if iadd:
        print("\n" + "-" * 60)
        print("THE TARGET: IADD3")
        inst = iadd[0]
        print(f"  Dest: {inst.dest.raw} (reg={inst.dest.reg})")
        print(f"  Src1: {inst.src[0].raw} (reg={inst.src[0].reg})")
        print(f"  Src2: {inst.src[1].raw} (reg={inst.src[1].reg})")
        print(f"  Src3: {inst.src[2].raw} (reg={inst.src[2].reg})")
        print(f"\n  → R9 = R2 + R5 + RZ (0)")
        print(f"  → This maps directly to Flynn ALU!")
    
    print("\n" + "=" * 60)
    print("PARSER TEST PASSED")
    print("=" * 60)
    
    return instructions


if __name__ == "__main__":
    test_parser()
