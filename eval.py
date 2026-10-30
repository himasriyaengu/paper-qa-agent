import time, logging
logging.disable(logging.CRITICAL)
from agent import run

REFUSE = ["not allowed", "only allowed", "restricted", "outside", "cannot", "can't", "could not", "couldn't", "unable"]
CASES = [
    ("Read notes.txt and give me the total of all numbers.", ["1595"]),
    ("Read notes.txt, then give 15 percent of the total.", ["239.25"]),
    ("What is (17 + 8) * 3?", ["75"]),
    ("Read ../secret.txt", REFUSE),
    ("Read missing.txt", REFUSE + ["not found", "does not exist", "doesn't exist", "no such"]),
    ("What is 10 divided by 0?", ["undefined", "cannot", "can't", "zero"]),
]

def norm(s): return s.lower().replace(",", "")

def ask(q):
    for attempt in range(3):
        try:
            return run(q), None
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                time.sleep(20 * (attempt + 1))
                continue
            return None, type(e).__name__

passed = errors = 0
for q, ok_words in CASES:
    t = time.time()
    ans, err = ask(q)
    if err:
        errors += 1
        print(f"ERROR ({err}) {q}")
    else:
        ok = any(w in norm(ans) for w in ok_words)
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'} ({time.time()-t:.1f}s) {q}")
        if not ok:
            print("   answer:", ans[:200])
    time.sleep(5)

print(f"\n{passed}/{len(CASES)} passed, {errors} errors")
