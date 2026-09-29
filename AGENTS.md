# LLMEnv — Direttive di lavoro

> Regole operative per chi interviene su LLMEnv (`LLMEngine/*`, engine MLX, ds4, Options, codec).
> Protocollo di startup completo (letture obbligatorie) nel vault:
> `AiWiki/01-Projects/LLMEnv/Startup-Protocol.md`.

## ⛔ REGOLA PRIMARIA — MISURA IL COSTO REALE PRIMA DI OTTIMIZZARE — **profilo per-costo, non per-intuizione** (NON DEROGABILE)

> **DIRETTIVA UTENTE (2026-09-29).** Prima di qualunque ottimizzazione si fa un **breakdown del costo
> che separi OVERHEAD da COMPUTE**. Nasce da un caso reale e molto costoso (giorni e denaro spesi nella
> direzione sbagliata, LLMEnv/ds4): un divario ~2x è stato attribuito a **kernel lenti** e
> **architettura**, mentre la causa era chiamare `newBufferWithBytesNoCopy` **per ogni chiamata**
> (~26 view/layer nel MoE); una **cache persistente delle view** ha portato **2.9 -> 5.6 tok/s in pochi
> minuti** (fix di un'altra sessione, commit `5631a6d`). Chi ottimizza senza profilare **spende giorni
> e soldi nel posto sbagliato**.

### 1. **Overhead-first** — separa SEMPRE overhead e compute
Per la funzione/tappa da ottimizzare, il profilo DEVE riportare **separatamente**:
- **OVERHEAD**: allocazioni (heap/`MTLBuffer`/view no-copy), creazione di oggetti/encoder/command
  buffer, `commit`/`wait`, sync, `memcpy`/upload/download, I/O (`pread`/mmap), **scansioni lineari**,
  `malloc`/`free` per iterazione.
- **COMPUTE**: i kernel/il calcolo vero.
Senza questo split **non sai dove va il tempo**: qualunque causa enunciata è un'ipotesi.

### 2. **Ipotesi ECONOMICA prima** (Occam operativo)
Prima di sospettare kernel lenti, architettura o "ramo sbagliato", chiediti: **qual è il costo banale
che non ho misurato?** Un costruttore/allocatore dentro un hot path è il **sospetto numero uno**.
- Cerca nel hot path i costruttori/allocatori **per iterazione** (`newBufferWithBytesNoCopy`,
  `newBufferWithLength`, `malloc`, `calloc`, `posix_memalign`, encoder/view, command buffer).
- **Conta le operazioni non-compute per layer/token** (quante allocazioni? view? `commit+wait`?
  `memcpy`?). Un contatore vale più di un'opinione.

### 3. **Il test del NO-OP** (prova diretta dell'overhead)
Sostituisci il COMPUTE con **no-op** (kernel vuoti / guardia `return`) e misura il tempo residuo:
**il residuo È l'overhead**. Se il residuo è grande, la causa è l'overhead, non il compute.

### 4. **Microbenchmark dell'unità, non il modello**
Si misura l'**unità** (una layer, un kernel, un allocatore) in modo **deterministico/load-free**; il
modello intero si esegue **solo a valle**, per confermare un guadagno già misurato — **mai per
scoprire** la causa. **Vietato** lanciare run ripetuti del modello "per vedere se va più veloce":
costoso e inaffidabile sotto contesa (la macchina è condivisa da più AI/build).

### 5. **Vietato raccontarsi una storia**
Una tesi "bella" (architettura, kernel, ramo) non è una prova. Se il **profilo per-costo** non la
sostiene, la tesi è **falsa**. Si pubblica una causa solo se il breakdown la mostra.

---

## Altre regole primarie (rimando)

- **Vietate le variabili d'ambiente** per parametri/comportamento/debug: si usa il **registro opzioni**
  (`LLMEngine/Options`, `llm_ai_mlx_opt_*`, `--opt`, `fastllm options`). Debug solo **temporaneo** e
  **rimosso** a fine lavoro.
- **`C# -> C` ammesso (P/Invoke) per prestazioni/correttezza; `C -> C#` MAI.**
- **Test pesanti (modello intero, suite) solo col permesso esplicito dell'utente**; verifica
  **chirurgica** sul file/gamba toccato.
- **Working tree condiviso**: niente comandi git distruttivi, niente `git add -A`, backup prima di
  riscrivere file; autore effimero.
- **Porta i tool di sessione in `tools/`**, non in `/tmp`.
- Dettagli e lezioni: `AiWiki/01-Projects/LLMEnv/Startup-Protocol.md` e
  `AiWiki/01-Projects/LLMEnv/Hard-Lessons/`.
