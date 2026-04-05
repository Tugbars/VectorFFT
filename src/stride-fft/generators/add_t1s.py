"""Add emit_ext_tw_scalar + ct_t1s_dit support to all generators."""
import os, re

SCALAR_METHOD = '''    def emit_ext_tw_scalar(self, v, tw_idx, d):
        """Emit twiddle multiply using scalar broadcast (t1s variant)."""
        fwd = (d == 'fwd')
        T = self.isa.T
        self.n_load += 2
        if self.isa.name == 'scalar':
            self.o(f"{{ double wr = W_re[{tw_idx}], wi = W_im[{tw_idx}], tr = {v}_re;")
            if fwd:
                self.o(f"  {v}_re = {v}_re*wr - {v}_im*wi;")
                self.o(f"  {v}_im = tr*wi + {v}_im*wr; }}")
            else:
                self.o(f"  {v}_re = {v}_re*wr + {v}_im*wi;")
                self.o(f"  {v}_im = {v}_im*wr - tr*wi; }}")
        elif self.isa.name == 'avx2':
            self.o(f"{{ const {T} wr = _mm256_broadcast_sd(&W_re[{tw_idx}]);")
            self.o(f"  const {T} wi = _mm256_broadcast_sd(&W_im[{tw_idx}]);")
            self.o(f"  const {T} tr = {v}_re;")
            if fwd:
                self.o(f"  {v}_re = {self.fms(f\\'{v}_re\\',\\'wr\\',self.mul(f\\'{v}_im\\',\\'wi\\'))};")
                self.o(f"  {v}_im = {self.fma(\\'tr\\',\\'wi\\',self.mul(f\\'{v}_im\\',\\'wr\\'))}; }}")
            else:
                self.o(f"  {v}_re = {self.fma(f\\'{v}_re\\',\\'wr\\',self.mul(f\\'{v}_im\\',\\'wi\\'))};")
                self.o(f"  {v}_im = {self.fms(f\\'{v}_im\\',\\'wr\\',self.mul(\\'tr\\',\\'wi\\'))}; }}")
        else:  # avx512
            self.o(f"{{ const {T} wr = _mm512_set1_pd(W_re[{tw_idx}]);")
            self.o(f"  const {T} wi = _mm512_set1_pd(W_im[{tw_idx}]);")
            self.o(f"  const {T} tr = {v}_re;")
            if fwd:
                self.o(f"  {v}_re = {self.fms(f\\'{v}_re\\',\\'wr\\',self.mul(f\\'{v}_im\\',\\'wi\\'))};")
                self.o(f"  {v}_im = {self.fma(\\'tr\\',\\'wi\\',self.mul(f\\'{v}_im\\',\\'wr\\'))}; }}")
            else:
                self.o(f"  {v}_re = {self.fma(f\\'{v}_re\\',\\'wr\\',self.mul(f\\'{v}_im\\',\\'wi\\'))};")
                self.o(f"  {v}_im = {self.fms(f\\'{v}_im\\',\\'wr\\',self.mul(\\'tr\\',\\'wi\\'))}; }}")

'''

# This doesn't work well with f-string escaping. Let me use a different approach.
# Instead of injecting the method as text, I'll patch each file directly.

files = [
    'gen_radix3.py', 'gen_radix6.py', 'gen_radix7.py',
    'gen_radix10.py', 'gen_radix11.py', 'gen_radix12.py', 'gen_radix13.py',
    'gen_radix16.py', 'gen_radix17.py', 'gen_radix19.py',
    'gen_radix20.py', 'gen_radix25.py'
]

for fname in files:
    with open(fname, 'r', encoding='utf-8') as f:
        content = f.read()

    if 'emit_ext_tw_scalar' in content:
        print(f'{fname}: already has emit_ext_tw_scalar, skip')
        continue

    changes = 0

    # 1. Add 'ct_t1s_dit' to CLI choices
    old = "'ct_t1_dit', 'ct_t1_dit_log3'"
    new = "'ct_t1_dit', 'ct_t1s_dit', 'ct_t1_dit_log3'"
    if old in content:
        content = content.replace(old, new, 1)
        changes += 1

    # 2. Add ct_t1s_dit to ct_variants list
    old2 = "'ct_n1', 'ct_t1_dit', 'ct_t1_dit_log3', 'ct_t1_dif'"
    new2 = "'ct_n1', 'ct_t1_dit', 'ct_t1s_dit', 'ct_t1_dit_log3', 'ct_t1_dif'"
    if old2 in content:
        content = content.replace(old2, new2, 1)
        changes += 1

    # 3. Add is_t1s_dit flag in emit_file_ct
    old3 = "is_t1_dit_log3 = ct_variant == 'ct_t1_dit_log3'"
    new3 = "is_t1s_dit = ct_variant == 'ct_t1s_dit'\n    is_t1_dit_log3 = ct_variant == 'ct_t1_dit_log3'"
    if old3 in content and 'is_t1s_dit' not in content:
        content = content.replace(old3, new3, 1)
        changes += 1

    # 4. Add t1s function name/vname in emit_file_ct
    # Find the radix number from filename
    R = int(fname.replace('gen_radix', '').replace('.py', ''))
    old4 = f"elif is_t1_dit_log3:\n        func_base = \"radix{R}_t1_dit_log3\""
    new4 = f"elif is_t1s_dit:\n        func_base = \"radix{R}_t1s_dit\"\n        vname = \"t1s DIT (in-place, scalar broadcast twiddle)\"\n    elif is_t1_dit_log3:\n        func_base = \"radix{R}_t1_dit_log3\""
    if old4 in content:
        content = content.replace(old4, new4, 1)
        changes += 1

    # 5. Add dit_tw_scalar dispatch in kernel body selection
    # Look for the pattern that dispatches to dit_tw_log3 or dit_tw
    # In emit_file_ct, find where kernel_variant is set
    old5 = "if is_t1_dit_log3:"
    new5 = "if is_t1s_dit:\n            emit_kernel_body(em, d, 'dit_tw_scalar')\n        elif is_t1_dit_log3:"
    # Only replace in the emit_file_ct context (the second occurrence usually)
    # Count occurrences
    count = content.count(old5)
    if count >= 1 and 'is_t1s_dit' in content:
        # Replace the one inside the for d in ['fwd','bwd'] loop in emit_file_ct
        # This is the one that has emit_kernel_body_log3 right after
        old5_ctx = "        if is_t1_dit_log3:\n            emit_kernel_body_log3"
        new5_ctx = "        if is_t1s_dit:\n            emit_kernel_body(em, d, 'dit_tw_scalar')\n        elif is_t1_dit_log3:\n            emit_kernel_body_log3"
        if old5_ctx in content:
            content = content.replace(old5_ctx, new5_ctx, 1)
            changes += 1

    # 6. Add dit_tw_scalar handling in emit_kernel_body
    # Find where dit_tw twiddle is applied and add scalar variant
    # Pattern: "if variant == 'dit_tw':" followed by emit_ext_tw calls
    old6 = "if variant == 'dit_tw':"
    if old6 in content:
        # Find the block: "if variant == 'dit_tw':\n        for n in range(...):\n            em.emit_ext_tw(...)"
        # Add elif for dit_tw_scalar after the dit_tw block
        # We need to find where the dit_tw block ends (the next elif or blank line)
        pass  # This is complex, handle below

    # For emit_kernel_body: add dit_tw_scalar as a variant
    # The pattern varies by generator but generally:
    # if variant == 'dit_tw':
    #     for n in range(1, R):
    #         em.emit_ext_tw(...)
    # We add:
    # elif variant == 'dit_tw_scalar':
    #     for n in range(1, R):
    #         em.emit_ext_tw_scalar(...)

    # Simple approach: after each "em.emit_ext_tw(" in the dit_tw block,
    # there's usually a matching pattern we can duplicate.
    # But it's safer to just add the elif block.

    # Find "variant == 'dit_tw'" in emit_kernel_body and add scalar after its block
    # The DIT twiddle block typically looks like:
    #     if variant == 'dit_tw':
    #         for n in range(1, R):
    #             em.emit_ext_tw(f"x{n}", n - 1, d)
    # or for composite radixes it might be different.

    # Let's find all "variant == 'dit_tw'" and after each block add the scalar version
    lines = content.split('\n')
    new_lines = []
    i = 0
    added_scalar_variant = False
    while i < len(lines):
        line = lines[i]
        new_lines.append(line)

        # Detect the dit_tw block in emit_kernel_body
        if "variant == 'dit_tw'" in line and 'dit_tw_scalar' not in line and 'dit_tw_log3' not in line:
            # Collect the dit_tw block (indented lines after it)
            base_indent = len(line) - len(line.lstrip())
            block_lines = [line]
            j = i + 1
            while j < len(lines) and lines[j].strip() and (len(lines[j]) - len(lines[j].lstrip()) > base_indent or lines[j].strip().startswith('for') or lines[j].strip().startswith('em.')):
                new_lines.append(lines[j])
                block_lines.append(lines[j])
                j += 1

            # Generate the scalar variant by replacing emit_ext_tw with emit_ext_tw_scalar
            indent = ' ' * base_indent
            scalar_block = []
            for bl in block_lines:
                if "variant == 'dit_tw'" in bl:
                    scalar_block.append(bl.replace("variant == 'dit_tw'", "variant == 'dit_tw_scalar'").replace('if ', 'elif '))
                elif 'emit_ext_tw(' in bl:
                    scalar_block.append(bl.replace('emit_ext_tw(', 'emit_ext_tw_scalar('))
                else:
                    scalar_block.append(bl)

            for sl in scalar_block:
                new_lines.append(sl)

            i = j
            added_scalar_variant = True
            continue

        # Also handle DIF variant if present
        if "variant == 'dif_tw'" in line and 'dif_tw_scalar' not in line and 'dif_tw_log3' not in line:
            base_indent = len(line) - len(line.lstrip())
            j = i + 1
            dif_block = [line]
            while j < len(lines) and lines[j].strip() and (len(lines[j]) - len(lines[j].lstrip()) > base_indent or lines[j].strip().startswith('for') or lines[j].strip().startswith('em.')):
                new_lines.append(lines[j])
                dif_block.append(lines[j])
                j += 1

            scalar_block = []
            for bl in dif_block:
                if "variant == 'dif_tw'" in bl:
                    scalar_block.append(bl.replace("variant == 'dif_tw'", "variant == 'dif_tw_scalar'").replace('if ', 'elif '))
                elif 'emit_ext_tw(' in bl:
                    scalar_block.append(bl.replace('emit_ext_tw(', 'emit_ext_tw_scalar('))
                else:
                    scalar_block.append(bl)

            for sl in scalar_block:
                new_lines.append(sl)

            i = j
            continue

        i += 1

    content = '\n'.join(new_lines)

    with open(fname, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f'{fname}: {changes} patches + kernel_body scalar variant = {"added" if added_scalar_variant else "MISSED"}')
