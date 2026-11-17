#!/usr/bin/env python3
"""semfocus : mesure le focus sémantique d'un site (idée de siteFocusScore / siteRadius).

Installation : pip install requests sentence-transformers
Usage        : python semfocus.py https://exemple.fr
Options      : --max 300 (pages analysées)
               --model intfloat/multilingual-e5-base (plus précis, plus lent)
"""
import argparse, csv, gzip, html, os, re, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse, urldefrag
import xml.etree.ElementTree as ET

import numpy as np
import requests

SKIP_EXT = re.compile(r"\.(jpe?g|png|gif|webp|avif|svg|pdf|zip|rar|mp4|mp3|css|js|json|xml|txt|ico|woff2?)$", re.I)
SKIP_TAGS = {"script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "form", "iframe", "template", "button", "select"}
BLOCKS = {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "td", "th", "blockquote", "pre", "div", "section", "dd", "dt", "figcaption", "br"}
MAIN = ("main", "article")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # avertissement sans conséquence sous Windows
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

http = requests.Session()
http.headers["User-Agent"] = "Mozilla/5.0 (compatible; semfocus/1.0)"
failures = Counter()  # pages non récupérées, par motif


def get(url):
    """GET poli : patiente et réessaie si le site limite le débit (429, 503)."""
    for attempt in range(4):
        try:
            r = http.get(url, timeout=15)
        except requests.RequestException:
            failures["réseau"] += 1
            return None
        if r.status_code in (429, 503) and attempt < 3:
            wait = r.headers.get("Retry-After", "")
            time.sleep(min(int(wait), 60) if wait.isdigit() else 10 * (attempt + 1))
            continue
        if r.status_code != 200:
            failures[f"HTTP {r.status_code}"] += 1
            return None
        return r
    return None


def host(url):
    return urlparse(url).netloc.lower().removeprefix("www.")


def clean(url):
    return urldefrag(url)[0].rstrip("/")


class Page(HTMLParser):
    """Texte par blocs, sans menus, en-têtes, pieds de page ni scripts ; <main> ou <article> en priorité.
    Les blocs faits surtout de liens (listes de sujets, nuages de tags, fils d'Ariane) sont écartés."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip = self.main = self.link = 0
        self.buf, self.blocks, self.main_blocks, self.links = [], [], [], []
        self.title, self.in_title, self.noindex, self.lang = "", False, False, ""

    def flush(self):
        text = " ".join("".join(t for t, _ in self.buf).split())
        linked = len("".join(t for t, in_link in self.buf if in_link).strip())
        if text and linked <= 0.5 * len(text):
            self.blocks.append(text)
            if self.main:
                self.main_blocks.append(text)
        self.buf = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "html":
            self.lang = (a.get("lang") or "")[:2].lower()
        if tag == "a" and a.get("href"):
            self.links.append(a["href"])
        if tag == "meta" and (a.get("name") or "").lower() == "robots" and "noindex" in (a.get("content") or "").lower():
            self.noindex = True
        if tag in BLOCKS or tag in SKIP_TAGS or tag in MAIN:
            self.flush()
        self.in_title |= tag == "title"
        self.skip += tag in SKIP_TAGS
        self.main += tag in MAIN
        self.link += tag == "a"

    def handle_endtag(self, tag):
        if tag in BLOCKS or tag in SKIP_TAGS or tag in MAIN:
            self.flush()
        if tag == "title":
            self.in_title = False
        if tag == "a" and self.link:
            self.link -= 1
        if tag in SKIP_TAGS and self.skip:
            self.skip -= 1
        if tag in MAIN and self.main:
            self.main -= 1

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        elif not self.skip:
            self.buf.append((data, self.link > 0))


def fetch(url):
    r = get(url)
    if not r or "html" not in r.headers.get("Content-Type", ""):
        return None
    p = Page()
    p.feed(r.text)
    p.flush()
    blocks = p.main_blocks if sum(len(b.split()) for b in p.main_blocks) >= 80 else p.blocks
    links = [clean(urljoin(r.url, l)) for l in p.links]
    return {"url": clean(r.url), "title": " ".join(p.title.split()), "blocks": blocks, "links": links, "noindex": p.noindex, "lang": p.lang}


def sitemap_urls(root):
    robots = get(urljoin(root, "/robots.txt"))
    queue = re.findall(r"(?im)^\s*sitemap:\s*(\S+)", robots.text) if robots else []
    queue = queue or [urljoin(root, "/sitemap.xml"), urljoin(root, "/sitemap_index.xml")]
    seen, pages = set(), []
    while queue and len(seen) < 100:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        r = get(url)
        if not r:
            continue
        data = gzip.decompress(r.content) if r.content[:2] == b"\x1f\x8b" else r.content
        try:
            tree = ET.fromstring(data)
        except ET.ParseError:
            continue
        index = tree.tag.endswith("sitemapindex")
        for entry in tree:
            for child in entry:
                if child.tag.split("}")[-1] == "loc" and child.text:
                    (queue if index else pages).append(child.text.strip())
    return list(dict.fromkeys(pages))


def collect(root, limit):
    site = host(root)
    urls = [u for u in sitemap_urls(root) if host(u) == site and not SKIP_EXT.search(urlparse(u).path)]
    pages, seen = [], set()
    with ThreadPoolExecutor(4) as pool:
        if urls:
            print(f"Sitemap : {len(urls)} adresses", end="")
            if len(urls) > limit:  # échantillon réparti sur tout le sitemap
                urls = [urls[int(i * len(urls) / limit)] for i in range(limit)]
                print(f", échantillon de {limit}", end="")
            print()
            pages = [p for p in pool.map(fetch, urls) if p]
        else:
            print("Pas de sitemap : exploration des liens internes")
            todo = [clean(root)]
            seen.add(todo[0])
            while todo and len(pages) < limit:
                batch, todo = todo[:16], todo[16:]
                for p in pool.map(fetch, batch):
                    if not p or len(pages) >= limit:
                        continue
                    pages.append(p)
                    for l in p["links"]:
                        if host(l) == site and l not in seen and not SKIP_EXT.search(urlparse(l).path):
                            seen.add(l)
                            todo.append(l)
    if failures:
        print("Pages non récupérées : " + ", ".join(f"{n} ({k})" for k, n in failures.items()))
    unique = {}
    for p in pages:
        if not p["noindex"]:
            unique.setdefault(p["url"], p)
    return list(unique.values())


def strip_boilerplate(pages):
    """Retire les blocs répétés sur plus de 30 % des pages (menus, encarts, mentions)
    et les fragments de moins de 5 mots (dates, compteurs, boutons, pseudos)."""
    for p in pages:
        p["blocks"] = [b for b in p["blocks"] if len(b.split()) >= 5]
    count = {}
    for p in pages:
        for b in set(p["blocks"]):
            count[b] = count.get(b, 0) + 1
    limit = max(2, 0.3 * len(pages))
    for p in pages:
        p["text"] = " ".join(b for b in p["blocks"] if count[b] <= limit)


def short(url, root):
    u = urlparse(url)
    return (u.path or "/") + ("?" + u.query if u.query else "")


def write_svg(path, C, verdicts, pages, focus, root):
    """Carte radiale : distance au centre = 1 - focus (mise à l'échelle), angle = ACP des vecteurs centrés (approximatif).
    Le cercle est le rayon moyen du site, et les pages loin du centre sont dehors."""
    _, _, vt = np.linalg.svd(C, full_matrices=False)
    ang = np.arctan2(C @ vt[1], C @ vt[0])
    dist = 1 - focus
    dist = (dist - dist.min()) / max(np.ptp(dist), 1e-9)  # mise à l'échelle : de la page la plus centrale à la plus éloignée
    w, h, m = 900, 700, 60
    k = min((w - 2 * m) / 2, (h - 2 * m - 60) / 2)
    cx, cy = w / 2, (h - 60) / 2 + 10
    xy = np.c_[np.cos(ang) * dist, np.sin(ang) * dist]
    r = dist.mean() * k
    colors = {"cœur": "#ECC46A", "à surveiller": "#8E959F", "loin du centre": "#F0645A"}
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="Segoe UI, Arial, sans-serif">',
           f'<rect width="{w}" height="{h}" fill="#18191C"/>',
           f'<text x="24" y="34" fill="#CDD1D6" font-size="16">semfocus · {html.escape(host(root))} · {len(pages)} pages (survolez un point)</text>',
           f'<circle cx="{cx}" cy="{cy}" r="{r:.1f}" fill="#DCAB4C" fill-opacity=".07" stroke="#DCAB4C" stroke-dasharray="6 5"/>']
    for i in np.argsort(-focus):  # les pages éloignées sont dessinées en dernier, au-dessus
        tip = html.escape(f"{pages[i]['title']}\n{pages[i]['url']}\nfocus {focus[i]:.3f} · {verdicts[i]}")
        out.append(f'<circle cx="{cx + xy[i, 0] * k:.1f}" cy="{cy - xy[i, 1] * k:.1f}" r="5" fill="{colors[verdicts[i]]}"><title>{tip}</title></circle>')
    out.append(f'<path d="M{cx - 9} {cy}h18M{cx} {cy - 9}v18" stroke="#F2F3F5" stroke-width="2"/>')
    y, x = h - 26, 24
    for label, col in colors.items():
        out.append(f'<circle cx="{x + 6}" cy="{y - 5}" r="6" fill="{col}"/><text x="{x + 18}" y="{y}" fill="#CDD1D6" font-size="14">{label}</text>')
        x += 140
    out.append(f'<text x="{x}" y="{y}" fill="#A3A9B2" font-size="13">+ centre · cercle = rayon moyen · distance à l’échelle, angle approximatif</text></svg>')
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Focus sémantique d'un site")
    ap.add_argument("url")
    ap.add_argument("--max", type=int, default=300)
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    args = ap.parse_args()
    root = args.url if "://" in args.url else "https://" + args.url

    pages = collect(root, args.max)
    strip_boilerplate(pages)
    pages = [p for p in pages if len(p["text"].split()) >= 50]
    if len(pages) < 5:
        sys.exit(f"Seulement {len(pages)} pages avec assez de texte : impossible de calculer un centre.")
    print(f"{len(pages)} pages indexables d'au moins 50 mots\n")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    e5 = "e5" in args.model
    docs = [("passage: " if e5 else "") + (p["title"] + ". " + p["text"])[:3000] for p in pages]
    E = model.encode(docs, normalize_embeddings=True, batch_size=16, show_progress_bar=True)

    # Un centre par langue (5 pages au moins) : une page traduite se compare aux pages de sa langue
    langs = np.array([p["lang"] for p in pages])
    counts = Counter(langs)
    if len(counts) > 1:
        print("Langues : " + ", ".join(f"{l or '?'} {n}" for l, n in counts.most_common()))
    groups = [np.where(langs == l)[0] for l, n in counts.items() if n >= 5]
    rest = np.where([counts[l] < 5 for l in langs])[0]
    focus, C = np.zeros(len(pages)), np.zeros_like(E)
    for g in groups + ([rest] if len(rest) else []):
        mean = E[g].mean(0) if len(g) >= 5 else E.mean(0)
        focus[g] = E[g] @ (mean / np.linalg.norm(mean))
        C[g] = E[g] - mean
    p10, p25 = np.percentile(focus, [10, 25])
    # Proximité entre pages sur les vecteurs centrés : sans le "fond commun" du site, les écarts redeviennent lisibles
    Cn = C / np.linalg.norm(C, axis=1, keepdims=True)
    S = Cn @ Cn.T
    np.fill_diagonal(S, -1)
    S[langs[:, None] != langs[None, :]] = -1  # une page et sa traduction se ressemblent forcément
    near = S.argmax(1)
    order = np.argsort(focus)

    print(f"\nFocus médian : {np.median(focus):.3f}   rayon moyen (1 - focus) : {np.mean(1 - focus):.3f}")
    print("Les valeurs dépendent du modèle : comparez les pages entre elles, pas à un seuil absolu.\n")
    print("== Cœur du site (les plus proches du centre) ==")
    for i in order[::-1][:5]:
        print(f"  {focus[i]:.3f}  {short(pages[i]['url'], root)}  {pages[i]['title'][:60]}")
    print("\n== Loin du centre (10 % les plus bas) : supprimer, rattacher ou sortir du site ==")
    for i in order[: max(3, len(pages) // 10)]:
        print(f"  {focus[i]:.3f}  {short(pages[i]['url'], root)}  {pages[i]['title'][:60]}")

    iu = np.triu_indices(len(pages), 1)
    pairs = S[iu]
    print("\n== Paires les plus proches : doublons ou cannibalisation à vérifier ==")
    for k in np.argsort(pairs)[::-1][:10]:
        i, j = iu[0][k], iu[1][k]
        print(f"  {pairs[k]:.3f}  {short(pages[i]['url'], root)}  <->  {short(pages[j]['url'], root)}")

    verdicts = ["loin du centre" if f <= p10 else "à surveiller" if f <= p25 else "cœur" for f in focus]
    out = f"semfocus-{host(root)}.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["url", "titre", "focus", "verdict", "page la plus proche", "similarité"])
        for i in order:
            w.writerow([pages[i]["url"], pages[i]["title"], f"{focus[i]:.4f}", verdicts[i], pages[near[i]]["url"], f"{S[i, near[i]]:.4f}"])
    write_svg(f"semfocus-{host(root)}.svg", C, verdicts, pages, focus, root)
    print(f"\nTableau complet : {out}")
    print(f"Carte des pages : semfocus-{host(root)}.svg (à ouvrir dans le navigateur)")


if __name__ == "__main__":
    main()
