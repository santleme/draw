"""CLI da revisao do holdout selado (draw/selflearn/review.py).

Uso:
  # placar: revisadas/pendentes por sessao, n do portao
  python tools/selflearn/review_holdout.py status [--pool captures/pool]

  # o humano olhou a pagina e o rascunho esta certo
  python tools/selflearn/review_holdout.py mark SESSAO GERACAO --accept-draft

  # o humano corrigiu o rascunho num editor e salvou a mascara
  python tools/selflearn/review_holdout.py mark SESSAO GERACAO --label corrigido.png

  # holdout das sessoes completas no formato de train_round.py
  python tools/selflearn/review_holdout.py export --out dados/holdout

Nada aqui revisa sozinho nem promove modelo: marcar exige a prova
explicita, e o placar so informa se o n do portao foi atingido.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from draw.selflearn import review  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default="captures/pool")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    mark = sub.add_parser("mark")
    mark.add_argument("session")
    mark.add_argument("generation", type=int)
    proof = mark.add_mutually_exclusive_group(required=True)
    proof.add_argument("--accept-draft", action="store_true")
    proof.add_argument("--label")
    mark.add_argument("--reviewer")
    export = sub.add_parser("export")
    export.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if args.command == "status":
        print(review.format_status(review.status(args.pool)))
    elif args.command == "mark":
        entry = review.mark_reviewed(
            args.pool,
            args.session,
            args.generation,
            accept_draft=args.accept_draft,
            label_path=args.label,
            reviewer=args.reviewer,
        )
        print(json.dumps(entry, indent=2, sort_keys=True))
        print(review.format_status(review.status(args.pool)))
    else:
        print(json.dumps(review.export_holdout(args.pool, args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
