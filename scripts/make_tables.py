"""Write the tables of Chapter 4 (LaTeX tabular bodies) from the JSON outputs of fvs.evaluate.

python scripts/make_tables.py [--results outputs/results] [--out outputs/tables]
"""
import argparse
import json
from pathlib import Path


def load(d, name):
    return json.load(open(Path(d) / f'{name}.json'))


def fmt(x):
    return f'{x:.1f}'


def diff_cell(d, bold=False):
    m, lo, hi = d
    star = lo > 0 or hi < 0
    p, l, h = f'{m:+.1f}', f'{lo:+.1f}', f'{hi:+.1f}'
    num = rf'\mathbf{{{p}}}' if (bold and star and p.startswith('+')) else p
    top = f'${num}' + ('^{*}' if star else '') + '$'
    return r'\begin{tabular}[t]{@{}c@{}}' + top + r'\\[-1pt]{\scriptsize[$' + l + '$, $' + h + r'$]}\end{tabular}'


def inline_diff(d, flip=False):
    m, lo, hi = d
    if flip:
        m, lo, hi = -m, -hi, -lo
    star = lo > 0 or hi < 0
    return rf'${m:+.1f}$\,[${lo:+.1f}$,\,${hi:+.1f}$]' + ('$^{*}$' if star else '')


def header(cols, width):
    return [rf'\begin{{tabular*}}{{{width}}}{{@{{\extracolsep{{\fill}}}}{cols}@{{}}}}', r'\toprule']


def objects_tables(res, out):
    sp, de = load(res, 'objects_sparse'), load(res, 'objects_dense')
    blocks = [sp['field/pooled'], sp['harris/pooled'], de['field/pooled'], de['harris/pooled']]
    rows = [('random', 'random'), ('co-vis', 'n_co'), ('count', 'cnt'), ('permuted-field', 'field_permuted'),
            ('Harris-matched', 'harris_matched'), ('Harris-sum', 'harris_sum'), ('field-sum', 'field_sum')]
    L = header('lcccc', r'0.8\textwidth') + [
        r'& \multicolumn{2}{c}{Sparse objects} & \multicolumn{2}{c}{Dense objects} \\',
        r'\cmidrule(lr){2-3}\cmidrule(l){4-5}', r'Planner & Field lm. & Harris lm. & Field lm. & Harris lm. \\',
        r'\midrule']
    best = [max(round(b['rates'][k], 1) for _, k in rows if k not in ('field_sum', 'random')) for b in blocks]
    for name, k in rows:
        vals = [rf'\textbf{{{fmt(b["rates"][k])}}}' if (k == 'field_sum' and round(b['rates'][k], 1) > m)
                else fmt(b['rates'][k]) for b, m in zip(blocks, best)]
        L.append(rf'\meth{{{name}}} & ' + ' & '.join(vals) + r' \\')
    L += [r'\midrule', r'\meth{oracle} & ' + ' & '.join(fmt(b['rates']['oracle']) for b in blocks) + r' \\',
          r'\bottomrule', r'\end{tabular*}']
    (out / 'tab_c2_obj.tex').write_text('\n'.join(L) + '\n')

    cols = ['field_sum - cnt', 'field_sum - harris_sum', 'field_sum - field_permuted', 'field_sum - harris_matched',
            'harris_matched - harris_sum']
    settings = [(r'\multicolumn{6}{@{}l}{\emph{Sparse objects}} \\', None),
                (r'(i) Field lm., $h$ pooled$^{\dagger}$', sp['field/pooled']),
                (r'(ii) Field lm., $h=0.02$', sp['field/h0.02']),
                (r'(iii) Field lm., $h=0.04$', sp['field/h0.04']),
                (r'(iv) Harris lm., $h$ pooled', sp['harris/pooled']),
                (r'\midrule\multicolumn{6}{@{}l}{\emph{Dense objects}} \\', None),
                (r'(v) Field lm.$^{\dagger}$', de['field/pooled']),
                (r'(vi) Harris lm.', de['harris/pooled'])]
    L = [r'\setlength{\tabcolsep}{2pt}\renewcommand{\arraystretch}{1.1}'] + header('lccccc', r'\textwidth') + [
        r'& \multicolumn{4}{c}{\meth{field-sum} minus} & \\', r'\cmidrule(lr){2-5}',
        r'Setting & \meth{count} & \meth{Harris-sum} & \meth{permuted-field} & \meth{Harris-matched}'
        r' & \begin{tabular}[b]{@{}c@{}}\meth{Harris-matched}\\$-$\,\meth{Harris-sum}\end{tabular} \\', r'\midrule']
    for label, b in settings:
        if b is None:
            L.append(label)
            continue
        L.append(label + ' & ' + ' & '.join(diff_cell(b['diffs'][c], bold=c.startswith('field_sum'))
                                            for c in cols) + r' \\[5pt]')
    L += [r'\bottomrule', r'\end{tabular*}']
    (out / 'tab_c2_obj_effects.tex').write_text('\n'.join(L) + '\n')


def indoor_tables(res, out):
    te, de, th = load(res, 'test_sparse_field'), load(res, 'dev_sparse_field'), load(res, 'test_sparse_harris')
    rows = [('meth', 'random', 'random'), ('meth', 'count', 'cnt'), ('meth', 'co-vis', 'n_co'),
            ('meth', 'angle', 'angle'), ('meth', 'Harris-sum', 'harris_sum'), ('meth', 'field-sum', 'field_sum'), None,
            ('arm', 'G', 'u_G'), ('arm', 'G+H', 'u_G+H'), ('arm', 'G+C9', 'u_G+C9'), ('arm', 'G+F', 'u_G+F'),
            ('arm', 'G+F+C9', 'u_G+F+C9'), None, ('meth', 'oracle', 'oracle')]
    srcs = [(te, 'rates'), (te, 'relative'), (de, 'rates'), (de, 'relative'), (th, 'rates')]
    ours = {'field_sum', 'u_G+F', 'u_G+F+C9'}
    groups = [[r for r in rows[:6] if r[2] != 'random'], rows[7:12]]
    best = {}
    for g in groups:
        for ci, (src, part) in enumerate(srcs):
            m = max(round(src[part][r[2]], 1) for r in g if r[2] not in ours)
            for r in g:
                if r[2] in ours and round(src[part][r[2]], 1) > m:
                    best[(r[2], ci)] = True
    L = header('lccccc', r'\textwidth') + [
        r'& \multicolumn{2}{c}{Test, field lm.} & \multicolumn{2}{c}{Development, field lm.} & Test, Harris lm. \\',
        r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-6}', r'Planner & top-1 & Rel. & top-1 & Rel. & top-1 \\',
        r'\midrule']
    for r in rows:
        if r is None:
            L.append(r'\midrule')
            continue
        mac, name, k = r
        vals = []
        for ci, (src, part) in enumerate(srcs):
            v = '--' if (k == 'oracle' and part == 'relative') else fmt(src[part][k])
            vals.append(rf'\textbf{{{v}}}' if best.get((k, ci)) else v)
        L.append(rf'\{mac}{{{name}}} & ' + ' & '.join(vals) + r' \\')
    L += [r'\bottomrule', r'\end{tabular*}']
    (out / 'tab_c2_planners.tex').write_text('\n'.join(L) + '\n')

    cols = ['u_G+F - u_G', 'u_G+F - u_G+H', 'u_G+F - u_G+C9', 'u_G+F+C9 - u_G+C9', 'u_G - cnt']
    settings = [(r'\multicolumn{6}{@{}l}{\emph{Test rooms, sparse tier}} \\', None),
                (r'(a) Field lm., both $h$$^{\dagger}$', te),
                (r'(b) Field lm., $h=10$\,cm$^{\ddagger}$', load(res, 'test_sparse_field_h100')),
                (r'(c) Field lm., $h=5$\,cm', load(res, 'test_sparse_field_h50')),
                (r'(d) Harris lm., both $h$', th),
                (r'\midrule\multicolumn{6}{@{}l}{\emph{Development rooms}} \\', None),
                (r'(e) Sparse, field lm.', de),
                (r'(f) Dense', load(res, 'dev_dense')),
                (r'\midrule\multicolumn{6}{@{}l}{\emph{Test rooms, dense tier}} \\', None),
                (r'(g) Dense', load(res, 'test_dense'))]
    L = [r'\setlength{\tabcolsep}{2pt}\renewcommand{\arraystretch}{1.1}'] + header('lccccc', r'\textwidth') + [
        r'Setting & \begin{tabular}[b]{@{}c@{}}H1\\\arm{G+F}$-$\arm{G}\end{tabular}'
        r' & \begin{tabular}[b]{@{}c@{}}H2\\\arm{G+F}$-$\arm{G+H}\end{tabular}'
        r' & \begin{tabular}[b]{@{}c@{}}H3\\\arm{G+F}$-$\arm{G+C9}\end{tabular}'
        r' & \begin{tabular}[b]{@{}c@{}}\arm{G+F+C9}\\$-$\,\arm{G+C9}\end{tabular}'
        r' & \begin{tabular}[b]{@{}c@{}}\arm{G}\\$-$\,\meth{count}\end{tabular} \\', r'\midrule']
    for label, b in settings:
        if b is None:
            L.append(label)
            continue
        L.append(label + ' & ' + ' & '.join(diff_cell(b['diffs'][c], bold=c.startswith('u_G+F'))
                                            for c in cols) + r' \\[5pt]')
    L += [r'\bottomrule', r'\end{tabular*}']
    (out / 'tab_c2_effects.tex').write_text('\n'.join(L) + '\n')

    dv, ts = load(res, 'dev_dense'), load(res, 'test_dense')
    rows = [('meth', 'random', 'random'), ('meth', 'count', 'cnt'), ('meth', 'co-vis', 'n_co'), (None, 'FIF', 'fif'),
            ('meth', 'angle', 'angle'), ('meth', 'Harris-sum', 'harris_sum'), ('meth', 'field-sum', 'field_sum'), None,
            ('arm', 'G', 'u_G'), ('arm', 'G+H', 'u_G+H'), ('arm', 'G+C9', 'u_G+C9'), ('arm', 'G+F', 'u_G+F'),
            ('arm', 'G+F+C9', 'u_G+F+C9'), None, ('meth', 'oracle', 'oracle')]
    groups = [rows[1:7], rows[8:13]]
    bold = {}
    for g in groups:
        for ci, src in enumerate([dv, ts]):
            m = max(round(src['rates'][r[2]], 1) for r in g if r[2] not in ours)
            for r in g:
                if r[2] in ours and round(src['rates'][r[2]], 1) > m:
                    bold[(r[2], ci)] = True
    L = header('lcc', r'0.6\textwidth') + [r'Planner & Development & Test \\', r'\midrule']
    for r in rows:
        if r is None:
            L.append(r'\midrule')
            continue
        mac, name, k = r
        lab = rf'\{mac}{{{name}}}' if mac else name
        vals = [rf'\textbf{{{fmt(s["rates"][k])}}}' if bold.get((k, ci)) else fmt(s['rates'][k])
                for ci, s in enumerate([dv, ts])]
        L.append(lab + ' & ' + ' & '.join(vals) + r' \\')
    L += [r'\bottomrule', r'\end{tabular*}']
    (out / 'tab_c2_dense.tex').write_text('\n'.join(L) + '\n')

    dlm, tlm = load(res, 'dev_dense_lwl_localizer'), load(res, 'test_dense_lwl_localizer')
    rows = [('Dense', r'LWL, all points$^{\dagger}$', dv, ts, 'u_lwl - cnt', False),
            ('Dense', 'LWL, localizer lm.', dlm, tlm, 'u_lwl - cnt', False),
            ('Dense', 'FIF', dv, ts, 'cnt - fif', True), None,
            ('Sparse', r'LWL, all points$^{\dagger}$', de, te, 'u_lwl - cnt', False),
            ('Sparse', 'FIF', de, te, 'cnt - fif', True)]
    L = header('llccc', r'0.9\textwidth') + [r'& & Development & \multicolumn{2}{c}{Test} \\', r'\cmidrule(l){4-5}',
                                             r'Tier & Planner & top-1 & top-1 & $-$\,\meth{count} \\', r'\midrule']
    for r in rows:
        if r is None:
            L.append(r'\midrule')
            continue
        tier, name, d, t, pair, flip = r
        k = 'fif' if flip else 'u_lwl'
        L.append(f'{tier} & {name} & {fmt(d["rates"][k])} & {fmt(t["rates"][k])} & '
                 f'{inline_diff(t["diffs"][pair], flip)} \\\\')
    L += [r'\bottomrule', r'\end{tabular*}']
    (out / 'tab_c2_competitors.tex').write_text('\n'.join(L) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results', default='outputs/results')
    ap.add_argument('--out', default='outputs/tables')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    objects_tables(a.results, out)
    indoor_tables(a.results, out)
    print('->', out)


if __name__ == '__main__':
    main()
