"""CLI do portao real (draw/selflearn/gate.py).

Uso:
  python tools/selflearn/judge_gate.py \
      --candidate candidate_real.json --baseline baseline_real.json \
      [--dibco dibco.json] [--min-sessions N] [--min-pages N]
      [--promote-margin F] [--worst-group-tolerance F]
      [--accepted-drift accepted_drift.json]

Os JSON de replay real trazem primary, worst_group e, no
candidato, holdout_sessions/holdout_pages. O JSON DIBCO traz iou
(e baseline_iou) e sai impresso como diagnostico. O
accepted_drift.json da rodada (train_round.py) entra como sinal ao
juizo: ATTENTION/DRIFT pedem revisao humana, sem mudar o veredito.

Codigo de saida: 0 = PROMOTE, 1 = REJECT, 2 = INCONCLUSIVE.
Promocao continua sendo decisao humana: esta ferramenta julga,
nao troca modelo.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from draw.selflearn import gate  # noqa: E402

_EXIT = {gate.PROMOTE: 0, gate.REJECT: 1, gate.INCONCLUSIVE: 2}


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--dibco", default=None)
    parser.add_argument("--min-sessions", type=int, default=gate.MIN_HOLDOUT_SESSIONS)
    parser.add_argument("--min-pages", type=int, default=gate.MIN_HOLDOUT_PAGES)
    parser.add_argument("--promote-margin", type=float, default=0.0)
    parser.add_argument("--worst-group-tolerance", type=float, default=0.0)
    parser.add_argument("--accepted-drift", default=None)
    args = parser.parse_args(argv)

    result = gate.judge_candidate(
        _load(args.candidate),
        _load(args.baseline),
        dibco=_load(args.dibco) if args.dibco else None,
        min_sessions=args.min_sessions,
        min_pages=args.min_pages,
        promote_margin=args.promote_margin,
        worst_group_tolerance=args.worst_group_tolerance,
        accepted_drift=(
            _load(args.accepted_drift)["status"] if args.accepted_drift else None
        ),
    )
    print(gate.format_verdict(result))
    return _EXIT[result["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
