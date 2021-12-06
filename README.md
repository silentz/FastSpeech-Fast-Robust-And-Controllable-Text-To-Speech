## FastSpeech TTS model

Third assignment on DLA (Deep learning in audio) HSE course.

### Reproduce model

1. Clone repository:
```bash
git clone git@github.com:silentz/tts.git
```

2. Cd into repository root:
```bash
cd tts
```

3. Create and activate virtualenv:
```bash
virtualenv --python=python3 venv
source venv/bin/activate
```

4. Install required packages:
```bash
pip install -r requirements.txt
```

5. Run init script (quite long operation):
```bash
./init.sh
```

6. Train model:
```bash
./train.sh
```

7. Test model:
```bash
./test.sh
```
