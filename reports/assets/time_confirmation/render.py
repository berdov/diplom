"""Render one paired VALID plot as SVG and PNG from verified source JSON.

Requires Pillow, but no scientific model or plotting framework.
"""

import argparse
from html import escape
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from report import HERE, build


def render(font_path):
    data = build()
    width, height, scale = 960, 560, 2
    image = Image.new('RGB', (width * scale, height * scale), 'white')
    draw = ImageDraw.Draw(image)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<title>Парные результаты shared/separate: лучший VALID NDCG@10</title>',
           '<rect width="100%" height="100%" fill="white"/>', '<g font-family="Arial, sans-serif">']

    def text(x, y, value, size=16, anchor='start', color='#222222'):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{color}">{escape(str(value))}</text>')
        font = ImageFont.truetype(str(font_path), round(size * scale))
        draw.text((x * scale, y * scale), str(value), font=font, fill=color,
                  anchor={'start': 'ls', 'middle': 'ms', 'end': 'rs'}[anchor])

    def line(x1, y1, x2, y2, color, stroke=1):
        svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{stroke}"/>')
        draw.line((x1 * scale, y1 * scale, x2 * scale, y2 * scale), fill=color, width=round(stroke * scale))

    def point(x, y, color, square=False, attrs=''):
        r = 5
        box = ((x - r) * scale, (y - r) * scale, (x + r) * scale, (y + r) * scale)
        if square:
            svg.append(f'<rect x="{x-r}" y="{y-r}" width="10" height="10" fill="{color}" {attrs}/>')
            draw.rectangle(box, fill=color)
        else:
            svg.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{color}" {attrs}/>')
            draw.ellipse(box, fill=color)

    blue, red = '#0072B2', '#B33B30'
    text(60, 37, 'Shared и separate: лучший VALID NDCG@10', 25)
    text(60, 66, 'KuaiRand, полный каталог; каждый отрезок соединяет одинаковый seed', 16)
    point(67, 97, blue)
    text(81, 103, 'shared', 16)
    point(187, 97, red, True)
    text(201, 103, 'separate', 16)
    Y = lambda value: 413 - (value - .0600) / .0040 * 280
    for i in range(9):
        value = .0600 + i * .0005
        y = Y(value)
        line(105, y, 905, y, '#E1E4E8')
        text(95, y + 5, f'{value:.4f}', 14, 'end', '#555555')
    for i, row in enumerate(data['pairs']):
        x = 180 + 160 * i
        a, b = row['shared'], row['separate']
        line(x - 16, Y(a), x + 16, Y(b), '#818891', 2)
        for mode, value, pos, color in [('shared', a, x - 16, blue), ('separate', b, x + 16, red)]:
            attrs = f'data-seed="{row["seed"]}" data-mode="{mode}" data-value="{value}"'
            point(pos, Y(value), color, mode == 'separate', attrs)
            text(pos, Y(value) + (22 if mode == 'shared' else -12), f'{value:.4f}', 14, 'middle', color)
        text(x, 449, row['seed'], 18, 'middle')
        text(x, 473, 'исходный' if row['seed'] == 2026 else 'новый', 14, 'middle', '#555555')
    text(60, 514, '2026: сохранённые исходные запуски; 2027–2030: подтверждающая серия.', 16)
    text(60, 542, 'Показаны реальные точки, не доверительные интервалы. Новых TEST нет.', 16)
    svg.append('</g></svg>')
    (HERE / 'paired_ndcg10.svg').write_text('\n'.join(svg) + '\n')
    image.save(HERE / 'paired_ndcg10.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', type=Path, help='Arial or another Cyrillic-capable TrueType font')
    args = parser.parse_args()
    font = args.font or next((p for p in [Path('/System/Library/Fonts/Supplemental/Arial.ttf'),
                                          Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')]
                              if p.exists()), None)
    if font is None:
        parser.error('Pass --font with an installed Cyrillic-capable font')
    render(font)
