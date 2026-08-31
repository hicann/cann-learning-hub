#!/usr/bin/env python3
"""Generate test data for the Stack Expression Evaluation Lab benchmark."""

import os
import numpy as np

BLOCK_DIM = 8
EXPR_LEN = 128
TOKEN_LEN = 64

# ---- Bracket Match expressions ----
# status: 0=matched, 1=right bracket empty stack, 2=type mismatch, 3=end non-empty
BRACKET_EXPRS = [
    ("(a+b)*(c-d)",           0),   # 0: matched
    ("{[a+b]*[c-d]}",         0),   # 1: matched
    ("((a+b))",               0),   # 2: matched
    ("(a+b]",                 2),   # 3: type mismatch
    (")a+b(",                 1),   # 4: right bracket with empty stack
    ("(a+b",                  3),   # 5: end with non-empty stack
    ("a+b-c",                 0),   # 6: matched (no brackets)
    ("((a)+(b))",             0),   # 7: matched
]

# ---- Suffix expressions (token encoded) ----
# token >= 0: operand, -1='+', -2='-', -3='*', -4='/'
SUFFIX_EXPRS = [
    # a+b*c → 4 2 3 * + → 4+2*3=10... let me recalculate
    # expr[0]: result=11 → 3 2 * 5 + → 3*2+5=11
    ([3, 2, -3, 5, -1],     11.0),
    # expr[1]: result=8 → 4 2 + → wait, need to check
    # 4 2 2 * + → 4+2*2=8
    ([4, 2, 2, -3, -1],     8.0),
    # expr[2]: result=5 → 3 2 + → 3+2=5
    ([3, 2, -1],             5.0),
    # expr[3]: result=14 → 4 2 * 3 2 / + → 4*2+3/2=8+1=9... no
    # 4 3 * 2 + → 4*3+2=14
    ([4, 3, -3, 2, -1],     14.0),
    # expr[4]: result=12 → 3 4 * → 3*4=12
    ([3, 4, -3],             12.0),
    # expr[5]: result=9 → 3 2 * 3 / → wait
    # 7 2 - 3 + → (7-2)+3=8... no
    # 3 6 + 3 / → (3+6)/3=3... no
    # 7 4 - 2 * → (7-4)*2=6... no
    # 4 5 + 9 - → (4+5)-9=0... no
    # 3 3 * 4 - 5 + → 9-4+5=10... no
    # 9 3 / 2 * 3 + → 3*2+3=9
    ([9, 3, -4, 2, -3, 3, -1], 9.0),
    # expr[6]: result=30 → 5 6 * → 5*6=30
    ([5, 6, -3],             30.0),
    # expr[7]: result=40 → 5 8 * → 5*8=40
    ([5, 8, -3],             40.0),
]

# ---- Infix to Postfix expressions ----
INFIX_EXPRS = [
    "a+b*c",              # → abc*+
    "(a+b)*c",            # → ab+c*
    "a*(b+c)",            # → abc+*
    "4*2+7-8/2",         # → 42*7+82/-
    "a+b",                # → ab+
    "a",                  # → a
    "a*b+c*d",           # → ab*cd*+
    "(a+b)-(c+d)",       # → ab+cd-+  wait... let me check
]

# Verify expected postfix for expr[7]: (a+b)-(c+d)
# a → output: a
# + → stack: #+
# ( → stack: #+(
# b → output: ab
# ) → pop until (: output: ab+, stack: #+
# - → priority(-)=1, priority(+)=1, +>=- → pop +: output: ab++... wait
# Actually: -(priority 1) vs +(priority 1): 1>=1 → pop +, then push -
# output: ab+a, stack: #-... wait let me redo
# ( → stack: #+(
# b → output: ab
# ) → pop until (: pop + → output: ab+, pop ( → stack: #
# - → top is #, priority(#)... # is sentinel with priority 0, 0 < 1 → push -
#   stack: #-
# ( → stack: #-(
# c → output: ab+c
# + → stack: #-(+
# d → output: ab+cd
# ) → pop until (: pop + → output: ab+cd+, pop ( → stack: #-
# # → pop until #: pop - → output: ab+cd+-
# So result: ab+cd+-
# But expected is ab+cd-*
# Hmm, that doesn't match. Let me re-read the expected output.
# expr[7]: ab+cd-*
# That would be from: (a+b)*(c-d)
# Let me check: (a+b)*(c-d)
# ( → stack: #(
# a → output: a
# + → stack: #(+
# b → output: ab
# ) → pop +: output: ab+, pop (: stack: #
# * → stack: #*
# ( → stack: #*(
# c → output: ab+c
# - → stack: #*(-
# d → output: ab+cd
# ) → pop -: output: ab+cd-, pop (: stack: #*
# # → pop *: output: ab+cd-*
# Yes! So expr[7] = "(a+b)*(c-d)"

INFIX_EXPRS[7] = "(a+b)*(c-d)"


def pad_expr(expr_str, max_len):
    """Pad expression with '#' terminator then zeros."""
    buf = np.zeros(max_len, dtype=np.int8)
    for i, ch in enumerate(expr_str):
        if i >= max_len:
            break
        buf[i] = ord(ch)
    if len(expr_str) < max_len:
        buf[len(expr_str)] = ord('#')
    return buf


def pad_tokens(tokens, max_len):
    """Pad token sequence with zeros."""
    buf = np.zeros(max_len, dtype=np.int32)
    for i, t in enumerate(tokens):
        if i >= max_len:
            break
        buf[i] = t
    return buf


def cpu_bracket_match(expr_bytes):
    """CPU reference for bracket matching."""
    stack = []
    for b in expr_bytes:
        ch = chr(b) if b != 0 else '\0'
        if ch == '#':
            break
        if ch in '([{':
            stack.append(ch)
        elif ch in ')]}':
            if not stack:
                return 1
            top = stack.pop()
            if (ch == ')' and top != '(') or \
               (ch == ']' and top != '[') or \
               (ch == '}' and top != '{'):
                return 2
    return 3 if stack else 0


def cpu_suffix_eval(tokens):
    """CPU reference for suffix evaluation."""
    stack = []
    for t in tokens:
        if t >= 0:
            stack.append(float(t))
        else:
            if len(stack) < 2:
                return 0.0
            b = stack.pop()
            a = stack.pop()
            if t == -1:
                stack.append(a + b)
            elif t == -2:
                stack.append(a - b)
            elif t == -3:
                stack.append(a * b)
            elif t == -4:
                stack.append(a / b if b != 0 else 0.0)
    return stack[0] if stack else 0.0


def cpu_infix_to_postfix(expr_bytes):
    """CPU reference for infix to postfix conversion."""
    def priority(op):
        if op in '+-':
            return 1
        if op in '*/':
            return 2
        if op == '(':
            return 0
        return -1

    def is_operand(ch):
        return ch.isalnum()

    output = []
    stack = ['#']
    for b in expr_bytes:
        ch = chr(b) if b != 0 else '\0'
        if ch == '#':
            break
        if is_operand(ch):
            output.append(ch)
        elif ch == '(':
            stack.append(ch)
        elif ch == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            if stack:
                stack.pop()  # pop '('
        elif ch in '+-*/':
            while stack and stack[-1] != '(' and priority(stack[-1]) >= priority(ch):
                output.append(stack.pop())
            stack.append(ch)

    while stack and stack[-1] != '#':
        output.append(stack.pop())

    return output


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lab_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(lab_dir, "data", "input")
    os.makedirs(data_dir, exist_ok=True)

    # === Bracket Match ===
    bracket_input = np.zeros(BLOCK_DIM * EXPR_LEN, dtype=np.int8)
    bracket_ref = np.zeros(BLOCK_DIM, dtype=np.int32)

    for i, (expr, expected_status) in enumerate(BRACKET_EXPRS):
        buf = pad_expr(expr, EXPR_LEN)
        bracket_input[i * EXPR_LEN:(i + 1) * EXPR_LEN] = buf
        # Verify with CPU reference
        cpu_status = cpu_bracket_match(buf)
        assert cpu_status == expected_status, \
            f"BracketMatch expr[{i}] '{expr}': CPU={cpu_status}, expected={expected_status}"
        bracket_ref[i] = expected_status

    bracket_input.tofile(os.path.join(data_dir, "bracket_input.bin"))
    bracket_ref.tofile(os.path.join(data_dir, "bracket_ref.bin"))
    print(f"BracketMatch: {BLOCK_DIM} expressions generated")

    # === Suffix Eval ===
    suffix_input = np.zeros(BLOCK_DIM * TOKEN_LEN, dtype=np.int32)
    suffix_ref = np.zeros(BLOCK_DIM, dtype=np.float32)

    for i, (tokens, expected_result) in enumerate(SUFFIX_EXPRS):
        buf = pad_tokens(tokens, TOKEN_LEN)
        suffix_input[i * TOKEN_LEN:(i + 1) * TOKEN_LEN] = buf
        cpu_result = cpu_suffix_eval(tokens)
        assert abs(cpu_result - expected_result) < 0.001, \
            f"SuffixEval expr[{i}]: CPU={cpu_result}, expected={expected_result}"
        suffix_ref[i] = expected_result

    suffix_input.tofile(os.path.join(data_dir, "suffix_input.bin"))
    suffix_ref.tofile(os.path.join(data_dir, "suffix_ref.bin"))
    print(f"SuffixEval: {BLOCK_DIM} expressions generated")

    # === Infix to Postfix ===
    infix_input = np.zeros(BLOCK_DIM * EXPR_LEN, dtype=np.int8)
    postfix_ref = np.zeros(BLOCK_DIM * EXPR_LEN, dtype=np.int8)

    for i, expr in enumerate(INFIX_EXPRS):
        buf = pad_expr(expr, EXPR_LEN)
        infix_input[i * EXPR_LEN:(i + 1) * EXPR_LEN] = buf
        cpu_postfix = cpu_infix_to_postfix(buf)
        ref_buf = pad_expr(''.join(cpu_postfix), EXPR_LEN)
        postfix_ref[i * EXPR_LEN:(i + 1) * EXPR_LEN] = ref_buf
        postfix_str = ''.join(cpu_postfix)
        print(f"  InfixToPostfix expr[{i}]: {expr} → {postfix_str}")

    infix_input.tofile(os.path.join(data_dir, "infix_input.bin"))
    postfix_ref.tofile(os.path.join(data_dir, "postfix_ref.bin"))
    print(f"InfixToPostfix: {BLOCK_DIM} expressions generated")

    print(f"\nAll test data saved to {data_dir}")


if __name__ == "__main__":
    main()
