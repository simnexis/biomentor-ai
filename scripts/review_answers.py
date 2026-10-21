"""Summarise a human-reviewed answer sheet (exported from the Disease Explorer page).

    python scripts/review_answers.py data/evaluation/answer_reviews/<file>.csv
"""

import json
import sys

from _common import ROOT  # noqa: F401
from biomentor.evaluation.answer_review import summarize_reviews

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(json.dumps(summarize_reviews(sys.argv[1]), indent=2))
