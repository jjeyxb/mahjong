import json
import multiprocessing
import os
from pathlib import Path
import sys

multiprocessing.freeze_support()
import bot

if "--self-check" in sys.argv:
    resources = Path(os.environ.get("MIA_RESOURCE_ROOT", Path(sys.executable).resolve().parents[2]))
    bot._load(str(resources / "models/mortal_298k.pth"), 0, bot.UPSTREAM)
    print(json.dumps(dict(ok=True, backend="cpu")))
else:
    raise SystemExit(bot.main())
