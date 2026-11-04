import time, requests
from finclaim.verifiers import gemini_generate

def make_gen(model):
    if not model.startswith("ollama:"):
        return gemini_generate(model)
    name = model.split(":", 1)[1]
    def g(prompt):
        for a in range(4):
            try:
                r = requests.post("http://localhost:11434/api/generate", timeout=300,
                    json={"model": name, "prompt": prompt, "stream": False,
                          "options": {"temperature": 0}})
                r.raise_for_status()
                return r.json()["response"]
            except Exception as e:
                print(f"  Ollama error (attempt {a+1}/4): {str(e)[:200]}", flush=True)
                time.sleep(5)
        raise RuntimeError("Ollama failed repeatedly; is the app running?")
    return g
