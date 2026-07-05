#!/usr/bin/env python3
"""Build fars-index.csv from the FARS corpus JSONs (paper id, title, dates, scores, review stats)."""
import json, csv, os, statistics

BASE = os.path.join(os.path.dirname(__file__), '..', 'data', 'fars-a-reviews', 'data')
OUT = os.path.join(os.path.dirname(__file__), '..', 'data', 'fars-index.csv')

rows = []
for pid in sorted(os.listdir(BASE)):
    d = os.path.join(BASE, pid)
    if not os.path.isdir(d):
        continue
    row = {'paper_id': pid, 'title': '', 'submission_date': '', 'venue': '',
           'llm_score': '', 'n_human_reviews': 0, 'human_rating_mean': '', 'human_ratings': ''}
    pr = os.path.join(d, 'paperreview.json')
    if os.path.exists(pr):
        j = json.load(open(pr))
        row['title'] = j.get('title', '')
        row['submission_date'] = j.get('submission_date', '')
        row['venue'] = j.get('venue', '')
        row['llm_score'] = j.get('numerical_score', '')
    hr = os.path.join(d, 'human_review.json')
    if os.path.exists(hr):
        j = json.load(open(hr))
        revs = j.get('reviews', [])
        ratings = []
        for r in revs:
            v = r.get('rating') or r.get('scores', {}).get('rating')
            if v is not None:
                try: ratings.append(float(v))
                except (TypeError, ValueError): pass
        row['n_human_reviews'] = len(revs)
        if ratings:
            row['human_rating_mean'] = round(statistics.mean(ratings), 2)
            row['human_ratings'] = ';'.join(str(x) for x in ratings)
    rows.append(row)

with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

n_rev = sum(1 for r in rows if r['n_human_reviews'] > 0)
means = [r['human_rating_mean'] for r in rows if r['human_rating_mean'] != '']
print(f"{len(rows)} papers indexed; {n_rev} with human reviews; "
      f"corpus human-rating mean {round(statistics.mean(means),2) if means else 'NA'}")
dates = sorted(r['submission_date'][:10] for r in rows if r['submission_date'])
print(f"submission dates: {dates[0]} .. {dates[-1]}" if dates else "no dates")
