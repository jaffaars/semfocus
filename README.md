# semfocus

Measure the semantic focus of a website, in the spirit of `siteFocusScore` and `siteRadius`, two attributes revealed by
the Google Content Warehouse leak (2024). Type an address: semfocus reads the site, turns every page into a vector
(embedding) and tells you which pages sit at the heart of the site, which ones drift away from it, and which pairs are
too close to each other.

Everything runs locally and for free: no API key, no account.

![semfocus map](docs/semfocus.png)

## What it does

1. **Collects the pages** from the sitemap (index and `.gz` included, evenly sampled on large sites), or by following
   internal links when there is no sitemap. Pages in `noindex` are skipped, rate limits (429, 503) are respected.
2. **Keeps the main content**: `<main>` or `<article>` when present, without menus, headers, footers, scripts, blocks
   made mostly of links, short fragments (dates, counters, buttons) and blocks repeated on more than 30 % of the pages.
3. **Embeds every page** with a free multilingual model that runs on your machine (`intfloat/multilingual-e5-small`).
4. **Computes the centre** of the site (one centre per language on multilingual sites) and, for every page,
   `focus = cos(page, centre)`. The radius is `mean(1 - focus)`.
5. **Reports**, in French:
   - the core of the site and the 10 % of pages furthest from the centre;
   - the closest pairs of pages (duplicates or cannibalisation to check), always within the same language;
   - `semfocus-<site>.csv`: every page with its score, verdict and closest page (opens in Excel);
   - `semfocus-<site>.svg`: a map of the pages, to open in a browser (hover a point to see the page).

## Getting started

Python 3.9 or later.

```bash
pip install -r requirements.txt
python semfocus.py https://example.com
```

The first run downloads the model (about 470 MB); PyTorch, installed by `pip`, weighs about 1 GB. Then everything runs
offline. About one minute for 60 to 150 pages on a regular computer.

Options:

```bash
python semfocus.py https://example.com --max 500                                  # pages analysed (default 300)
python semfocus.py https://example.com --model intfloat/multilingual-e5-base      # more accurate, slower
```

## Reading the results

- **Far from the centre, no traffic**: remove, stop linking to it, or `noindex`.
- **Far from the centre, with traffic**: tie it back to the core through internal links and context.
- **Close to the centre but ranking poorly**: reinforce its content and internal links.
- **Pairs too close**: merge with a 301, or make the two pages really different.

## Limits

- The leak gives the names of the attributes, not their weight in ranking. This is a diagnostic, not a ranking formula.
- Scores depend on the model: compare pages of the same site with each other, not with an absolute threshold.
- On the map, the distance to the centre comes from the score (scaled from the most central page to the furthest);
  only the angle is an approximation (principal component analysis).
- JavaScript is not run: content injected by scripts is not seen.

## License

[MIT](LICENSE)
