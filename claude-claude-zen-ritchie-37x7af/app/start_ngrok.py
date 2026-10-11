"""Uruchamia aplikację i wystawia ją publicznie przez ngrok (link do pokazania na prezentacji).

    export NGROK_AUTHTOKEN=...        # token z https://dashboard.ngrok.com/get-started/your-authtoken
    python start_ngrok.py             # opcjonalnie: --auth login:haslo
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from pyngrok import conf, ngrok

ap = argparse.ArgumentParser()
ap.add_argument('--port', type=int, default=8501)
ap.add_argument('--auth', help='login:hasło — zabezpiecza link podstawowym logowaniem')
args = ap.parse_args()

token = os.environ.get('NGROK_AUTHTOKEN')
if not token:
    sys.exit('Ustaw zmienną NGROK_AUTHTOKEN (token z https://dashboard.ngrok.com/get-started/your-authtoken).')
conf.get_default().auth_token = token

app_dir = Path(__file__).parent
if not (app_dir / 'models' / 'bundle.joblib').exists():
    sys.exit('Brak models/bundle.joblib — uruchom najpierw train_models.py.')

proc = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'app.py', '--server.port', str(args.port)],
                        cwd=app_dir)
time.sleep(4)
opts = {'basic_auth': [args.auth]} if args.auth else {}
tunnel = ngrok.connect(args.port, 'http', **opts)
print('\n' + '=' * 60)
print(f'🌍 Publiczny link do aplikacji: {tunnel.public_url}')
print('=' * 60 + '\nCtrl+C kończy.\n')
try:
    proc.wait()
except KeyboardInterrupt:
    pass
finally:
    ngrok.kill()
    proc.terminate()
