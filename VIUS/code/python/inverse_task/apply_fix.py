#!/usr/bin/env python3
"""
apply_fix_v3.py — робастное внесение исправлений с regex.
Использует re.sub для многострочных блоков.
"""

import shutil
import os
import re

def fix_file(filepath, rules):
    if not os.path.exists(filepath):
        print(f"[SKIP] Не найден: {filepath}")
        return

    bak = filepath + ".bak"
    shutil.copy2(filepath, bak)

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    for rule in rules:
        find = rule['find']
        repl = rule['replace']
        flags = rule.get('flags', 0)

        new_content, count = re.subn(find, repl, content, flags=flags)

        if count > 0:
            print(f"  [OK] Замена сработала ({count}x): {find[:70]!r}")
            content = new_content
        else:
            print(f"  [WARN] НЕ НАЙДЕНО: {find[:70]!r}")

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[DONE] {filepath}  (бэкап: {bak})")
    else:
        print(f"[NO CHANGES] {filepath}")


# === 1. helpers/inverse_winding_intermediate.py ===
fix_file(
    "helpers/inverse_winding_intermediate.py",
    [
        # 1. Импорт
        {
            'find': r'from helpers\.inverse_method import \(',
            'replace': 'from helpers.inverse_method_fixed import (',
        },
        # 2. Параметрические границы вместо z_min/z_max
        {
            'find': r'z_min, z_max = get_surface_height_bounds\(surface\)',
            'replace': """surf_u_min = getattr(surface, 'u_min', -float('inf'))
    surf_u_max = getattr(surface, 'u_max',  float('inf'))
    surf_v_min = getattr(surface, 'v_min', -float('inf'))
    surf_v_max = getattr(surface, 'v_max',  float('inf'))""",
        },
        # 3. near_boundary по u-границам
        {
            'find': r'near_boundary = \(u_cur < z_min \+ u_margin\) or \(u_cur > z_max - u_margin\)',
            'replace': 'near_boundary = (u_cur < surf_u_min + u_margin) or (u_cur > surf_u_max - u_margin)',
        },
        # 4a. Проверка границ предсказания (условие)
        {
            'find': r'if u_pred < z_min or u_pred > z_max:',
            'replace': 'if u_pred < surf_u_min or u_pred > surf_u_max:',
        },
        # 4b. Проверка границ предсказания (clip)
        {
            'find': r'u_pred = np\.clip\(u_pred, z_min, z_max\)',
            'replace': 'u_pred = np.clip(u_pred, surf_u_min, surf_u_max)',
        },
        # 5. Удаление бага ratio = 1.0 (вся строка с отступами и комментарием)
        {
            'find': r'[ \t]*ratio = 1\.0.*\n?',
            'replace': '',
            'flags': re.MULTILINE,
        },
        # 6. Восстановление ds_ratio в условии
        {
            'find': r'if ratio > jump_threshold or ratio < 1\.0 / jump_threshold:',
            'replace': 'if ds_ratio > jump_threshold or ds_ratio < 1.0 / jump_threshold:',
        },
        # 7. Fallback: три строки → две строки с правильными границами
        {
            'find': r'([ \t]*)u_p = u_s \+ du_s \* dz_sub\s*\n\s*[ \t]*v_p = v_s \+ dv_s \* dz_sub\s*\n\s*[ \t]*u_p = np\.clip\(u_p, z_min, z_max\)',
            'replace': r'\1u_p = np.clip(u_s + du_s * dz_sub, surf_u_min, surf_u_max)\n\1v_p = np.clip(v_s + dv_s * dz_sub, surf_v_min, surf_v_max)',
            'flags': re.DOTALL,
        },
    ]
)


# === 2. predictors/optical_predictor.py ===
fix_file(
    "helpers/optical_predictor.py",
    [
        {
            'find': r't_min=max\(0\.7 \* length, 1e-6\),',
            'replace': 't_min=1e-6,',
        },
    ]
)


print("\n=== Правки завершены. Проверьте git diff перед коммитом. ===")