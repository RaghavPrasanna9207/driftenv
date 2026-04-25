---
title: driftenv
sdk: docker
app_port: 7860
---

# driftenv

Minimal Hugging Face Docker Space setup for the FastAPI app in `server.py`.

The container runs:

```bash
uvicorn server:app --host 0.0.0.0 --port 7860
```
